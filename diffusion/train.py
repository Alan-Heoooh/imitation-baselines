import os
import json
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm
import MinkowskiEngine as ME
from diffusers.optimization import get_cosine_schedule_with_warmup

# from utils import load_data # data functions
from dataset.rh100t import RH100TDataset as RH100TPretrain, collate_fn
from dataset.rh100t_real import RH100TDataset as RH100TReal
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from models.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy

torch.multiprocessing.set_sharing_strategy('file_system')

# Prepare distributed training
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
RANK = int(os.environ['RANK'])
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
print(WORLD_SIZE, RANK, LOCAL_RANK)
# a bug in 6025/6026
os.environ['NCCL_P2P_DISABLE'] = '1'

def main(args):
    set_seed(1)
    # command line parameters
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    chunk_size = args['chunk_size']
    # freq = args['freq']
    save_epoch = args['save_epoch']
    resume_ckpt = args['resume_ckpt']
    # itw = args['in_the_wild']
    autoregressive_bins = args['autoregressive_bins']

    # get task parameters
    # from constants import TASK_CONFIGS
    # task_config = TASK_CONFIGS[task_name]
    # dataset_dir = task_config['dataset_dir_itw'] if itw else task_config['dataset_dir']
    camera_names = ['top'] #task_config['camera_names']
    # state_dim = 4 #task_config['state_dim']
    # state_dim = 8 #task_config['state_dim']
    state_dim = 10 # for rot 6d
    # stats_dir = task_config['stats_dir']
    # norm_stats = task_config['norm_stats']

    # if not os.path.exists(stats_dir):
    #     os.makedirs(stats_dir)

    # fixed parameters
    lr_backbone = 3e-4
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': chunk_size,
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'state_dim': state_dim,
                         'autoregressive_bins': autoregressive_bins
                         }
    elif policy_class == 'Diffusion':
        policy_config = {'lr': args['lr'],
                         'num_queries': chunk_size,
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'enc_layers': 4,
                         'dec_layers': 4,
                         'nheads': 8,
                         'state_dim': state_dim
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names, 'state_dim': state_dim}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'camera_names': camera_names,
        # 'norm_stats': norm_stats,
        'save_epoch': save_epoch,
        'resume_ckpt': resume_ckpt
    }

    # Init distributed training
    dist.init_process_group(backend='nccl', init_method='env://', world_size=WORLD_SIZE, rank=RANK)

    dataset_root_train = '/data'
    task_descriptions = json.load(open('dataset/task_description.json'))
    task_config_train = ['RH100T_cfg1','RH100T_cfg2','RH100T_cfg3','RH100T_cfg4','RH100T_cfg5','RH100T_cfg6','RH100T_cfg7']
    train_dataset = RH100TPretrain(dataset_root_train, task_config_train, 'train', task_descriptions, num_input=1, horizon=1+chunk_size, timestep=-1, from_cache=True, augmentation=True, top_down_view=True, rot_6d=True)
    task_config_test = ['RH100T_cfg1']
    dataset_root_val = '/data/chenxi/dataset/real_data_sampled'
    val_dataset = RH100TReal(dataset_root_val, task_config_test, 'train', task_descriptions, num_input=1, horizon=1+chunk_size, timestep=-1, augmentation=False, rot_6d=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=False, num_workers=24, collate_fn=collate_fn, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=24, collate_fn=collate_fn, sampler=val_sampler)

    # train_dataloader, val_dataloader = load_data(dataset_dir, task_name, camera_names, batch_size_train, batch_size_val, chunk_size, norm_stats, freq = freq, itw = itw)

    if RANK == 0 and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, train_sampler, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    if RANK == 0:
        torch.save(best_state_dict, ckpt_path)
        print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError

    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.module.configure_optimizers() 
    elif policy_class == 'CNNMLP':
        optimizer = policy.module.configure_optimizers() 
    elif policy_class == 'Diffusion':
        optimizer = policy.module.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def forward_pass(data, policy, device):
    cloud_coords = data['input_coords_list']
    cloud_feats = data['input_feats_list']
    qpos_data = data['input_frame_tcp_normalized']
    action_data = data['target_frame_tcp_normalized']
    is_pad = data['padding_mask']
    language_embed = data['language_embed']
    language_mask = data['language_mask']
    # convert to device
    cloud_feats, cloud_coords = cloud_feats.to(device), cloud_coords.to(device)
    language_embed, language_mask = language_embed.to(device), language_mask.to(device)
    qpos_data, action_data, is_pad = qpos_data.to(device), action_data.to(device), is_pad.to(device)
    # process cloud input
    cloud_data = ME.SparseTensor(cloud_feats, cloud_coords)
    return policy(qpos_data, cloud_data, language_embed, language_mask, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, train_sampler, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)    
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy = make_policy(policy_class, policy_config)
    policy = nn.parallel.DistributedDataParallel(policy, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)

    if config['resume_ckpt'] is not None:
        policy.module.load_state_dict(torch.load(config['resume_ckpt'], map_location = device))
        if RANK == 0:
            print('Loaded checkpoint from %s' % (config['resume_ckpt']))
    
    optimizer = make_optimizer(policy_class, policy)
    lr_scheduler = get_cosine_schedule_with_warmup(
                        optimizer=optimizer,
                        num_warmup_steps=2000,
                        num_training_steps=len(train_dataloader)*num_epochs,
                        last_epoch=-1)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in range(num_epochs):
        if RANK == 0: print(f'\nEpoch {epoch}')
        train_sampler.set_epoch(epoch)
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            pbar = tqdm(val_dataloader) if RANK == 0 else val_dataloader
            for data in pbar:
                forward_dict = forward_pass(data, policy, device)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            epoch_summary = synchronize_summary_dict(epoch_summary, device)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.module.state_dict()))
        if RANK == 0: print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v:.3f} '
        if RANK == 0: print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        num_steps = len(train_dataloader)
        pbar = tqdm(train_dataloader) if RANK == 0 else train_dataloader
        for data in pbar:
            forward_dict = forward_pass(data, policy, device)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[num_steps*epoch:num_steps*(epoch+1)])
        epoch_summary = synchronize_summary_dict(epoch_summary, device)
        epoch_train_loss = epoch_summary['loss']
        if RANK == 0: print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v:.3f} '
        if RANK == 0:
            print(summary_string)
            if epoch % config["save_epoch"] == 0:
                ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
                torch.save(policy.module.state_dict(), ckpt_path)
                plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    if RANK == 0: torch.save(policy.module.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    if RANK == 0:
        torch.save(best_state_dict, ckpt_path)
        print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    if RANK == 0:
        plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key] for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def synchronize_summary_dict(summary_dict, device, keys=['loss']):
    t = [summary_dict[k] for k in keys]
    t = torch.tensor(t, dtype=torch.float64, device=device)
    dist.barrier()
    dist.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
    t = t.tolist()
    for k,v in zip(keys, t):
        summary_dict[k] = v
    return summary_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--save_epoch', action='store', type=int, help='save frequency (epoch)', default=10, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    # parser.add_argument('--freq', action='store', type=float, help='frequency', required=True)
    parser.add_argument('--resume_ckpt', action='store', type=str, help='checkpoint to resume training', default=None, required=False)
    parser.add_argument('--in_the_wild', action='store_true', help='In the wild configuration', required=False)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--autoregressive_bins', action='store', type=int, help='autoregressive bins', default=1, required=False)
    
    main(vars(parser.parse_args()))
