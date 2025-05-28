import os
os.environ["PYTHONUNBUFFERED"] = "1"
import warnings
warnings.filterwarnings("ignore", message="The default value of the antialias parameter of all the resizing transforms")

import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup

from dataset.rh100t_real import RH100TDataset as RH100TReal, collate_fn
from utils import compute_dict_mean, set_seed, detach_dict
from models.policy import ACTPolicy

def main(args):
    set_seed(1)
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    task_name = args['task_name']
    batch_size = args['batch_size']
    num_epochs = args['num_epochs']
    chunk_size = args['chunk_size']
    save_epoch = args['save_epoch']
    resume_ckpt = args['resume_ckpt']
    autoregressive_bins = args['autoregressive_bins']

    camera_names = ['top']
    state_dim = 13 # 7 for robot joints, 6 for hand qpos
    lr_backbone = 5e-5
    backbone = 'resnet18'

    if policy_class == 'ACT':
        policy_config = {
            'lr': args['lr'],
            'num_queries': chunk_size,
            'kl_weight': args['kl_weight'],
            'hidden_dim': args['hidden_dim'],
            'dim_feedforward': args['dim_feedforward'],
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': camera_names,
            'state_dim': state_dim,
            'autoregressive_bins': autoregressive_bins
        }
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
        'save_epoch': save_epoch,
        'resume_ckpt': resume_ckpt
    }

    dataset_root = args['dataset_root']
    task_config = ['RH100T_cfg1']
    train_dataset = RH100TReal(dataset_root, task_config, 'train', num_input=1, horizon=1+chunk_size, timestep=-1, voxel_size=0.005, augmentation=True, num_sample='all')
    val_dataset = RH100TReal(dataset_root, task_config, 'train', num_input=1, horizon=1+chunk_size, timestep=-1, voxel_size=0.005, augmentation=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch {best_epoch}')

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        return ACTPolicy(policy_config)
    else:
        raise NotImplementedError

def make_optimizer(policy):
    return policy.configure_optimizers()

def forward_pass(data, policy, device):
    # top_image, wrist_image = data['top_image_list'].to(device), data['wrist_image_list'].to(device)
    top_image = data['top_image_list'].to(device)
    wrist_image = None
    
    qpos_data, action_data = data['input_frame_action_normalized'].to(device), data['target_frame_action_normalized'].to(device)
    is_pad = data['padding_mask'].to(device)
    return policy(qpos_data, top_image, wrist_image, action_data, is_pad)
    # return policy(qpos_data, top_image, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = make_policy(config['policy_class'], config['policy_config']).to(device)
    if config['resume_ckpt']:
        policy.load_state_dict(torch.load(config['resume_ckpt'], map_location=device))
        print(f'Loaded checkpoint from {config["resume_ckpt"]}')

    optimizer = make_optimizer(policy)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=len(train_dataloader)*config['num_epochs'])

    train_history = []
    min_val_loss = float('inf')
    best_ckpt_info = None

    for epoch in range(config['num_epochs']):
        print(f'\nEpoch {epoch}')

        policy.train()
        for data in tqdm(train_dataloader):
            output = forward_pass(data, policy, device)
            loss = output['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_history.append(detach_dict(output))

        epoch_summary = compute_dict_mean(train_history[-len(train_dataloader):])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        if epoch_train_loss < min_val_loss:
            min_val_loss = epoch_train_loss
            best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

        if epoch % config['save_epoch'] == 0:
            ckpt_path = os.path.join(config['ckpt_dir'], f'policy_epoch_{epoch}_seed_{config["seed"]}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, epoch, config['ckpt_dir'], config['seed'])

    torch.save(policy.state_dict(), os.path.join(config['ckpt_dir'], f'policy_last.ckpt'))
    plot_history(train_history, config['num_epochs'], config['ckpt_dir'], config['seed'])
    return best_ckpt_info

def plot_history(history, num_epochs, ckpt_dir, seed):
    for key in history[0]:
        path = os.path.join(ckpt_dir, f'train_{key}_seed_{seed}.png')
        plt.figure()
        values = [x[key].item() for x in history]
        plt.plot(np.linspace(0, num_epochs-1, len(history)), values, label='train')
        plt.legend()
        plt.title(key)
        plt.tight_layout()
        plt.savefig(path)
    print(f'Saved plots to {ckpt_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--policy_class', type=str, required=True)
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--save_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--kl_weight', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, required=True)
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--dim_feedforward', type=int, required=True)
    parser.add_argument('--autoregressive_bins', type=int, default=1)

    main(vars(parser.parse_args()))
