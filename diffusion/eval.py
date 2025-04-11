import os
import json
import time
import torch
import argparse
import numpy as np
import cv2
import torchvision.transforms as T
from multiprocessing import shared_memory

from PIL import Image
from transforms3d.quaternions import mat2quat, quat2mat
# from easydict import EasyDict as edict
# from easyrobot.robot.api import get_robot
# from easyrobot.camera.api import get_rgbd_camera

from utils import get_stats, set_seed
from models.policy import DiffusionPolicy
from dataset.rh100t import encode_gripper_width, decode_gripper_width, create_point_cloud_from_depth_image

from dataset.HIReal.flexiv_sdk.flexiv_api import FlexivApi
from dataset.HIReal.gripper_sdk.gripper_dahuan_modbus import DahuanModbusGripper
from dataset.HIReal.rh100t_sdk.configurations import load_conf
from dataset.HIReal.rh100t_sdk.rh100t_online import RH100TOnline

ROT6D = True
SCALE = np.array([1, 1, 1], dtype=np.float32)
OFFSET = np.array([0, 0, 0], dtype=np.float32)
# OFFSET = np.array([0, 0, -0.03], dtype=np.float32)
# SCALE = np.array([1.2, 0.95, 1], dtype=np.float32)
# OFFSET = np.array([0.05, 0.1, -0.02], dtype=np.float32)
# OFFSET = np.array([0.1, -0.05, -0.04], dtype=np.float32)

def rotation6d_to_quaternion(rot6d):
    print(rot6d)
    a1, a2 = rot6d[:3], rot6d[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-6)
    b2 = a2 - (b1 * a2).sum() * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-6)
    b3 = np.cross(b1, b2)
    matrix = np.stack([b1, b2, b3], axis=-2)
    quat = mat2quat(matrix)
    return quat

def quaternion_to_rotation6d(quat):
    matrix = quat2mat(quat)
    rot6d = matrix[:2].reshape(6)
    return rot6d

def rot_diff(quat1, quat2):
    mat1 = quat2mat(quat1)
    mat2 = quat2mat(quat2)
    diff = np.matmul(mat1, mat2.T)
    diff = np.diag(diff).sum()
    diff = min(max((diff-1)/2, -1), 1)
    diff = np.arccos(diff)
    return diff

def discrete_rotation(start_quat, end_quat, step=np.pi/24):
    start_6d = quaternion_to_rotation6d(start_quat)
    end_6d = quaternion_to_rotation6d(end_quat)
    diff = rot_diff(start_quat, end_quat)
    n_step = int(np.max(diff // step)) + 1
    quat_list = []
    for i in range(n_step):
        rot6d_i = start_6d * (n_step-1-i) + end_6d * (i+1)
        rot6d_i /= n_step
        quat_i = rotation6d_to_quaternion(rot6d_i)
        quat_list.append(quat_i)
    return quat_list

def get_observation(top_serial, wrist_serial, gripper):
    existing_shm_tcp = shared_memory.SharedMemory(name = 'tcp')
    top_shm_color = shared_memory.SharedMemory(name = '{}_color'.format(top_serial))
    wrist_shm_color = shared_memory.SharedMemory(name = '{}_color'.format(wrist_serial))
    tcp_raw = np.copy(np.ndarray((13, ), dtype = np.float64, buffer = existing_shm_tcp.buf))
    # tcp_raw[:3] = (tcp_raw[:3] - OFFSET) / SCALE
    top_image = np.copy(np.ndarray((720, 1280, 3), dtype = np.uint8, buffer = top_shm_color.buf))
    wrist_image = np.copy(np.ndarray((720, 1280, 3), dtype = np.uint8, buffer = wrist_shm_color.buf))
    # print('raw', tcp_raw)
    # print('prj', projected_tcp_7d)
    while True:
        try:
            gripper_info = gripper.get_info()
            break
        except Exception:
            pass
    gripper_width = decode_gripper_width(gripper_info, 'RH100T_cfg1')
    # tcp = np.array(list(tcp_raw[:7]) + [0]).astype(np.float32)  
    tcp = np.array(list(tcp_raw[:7]) + [gripper_width]).astype(np.float32)  
    # tcp = np.zeros((8))  
    # tcp[5] = 1

    return top_image, wrist_image, tcp

def pre_process(tcp, max_gripper_width=0.11):
    trans_min, trans_max = np.array([0, -0.5, 0]), np.array([1.0, 0.5, 0.3])
    tcp[...,:3] = (tcp[...,:3] - trans_min) / (trans_max - trans_min) * 2 - 1
    # tcp[...,-1] = tcp[...,-1] / max_gripper_width * 2 - 1
    if ROT6D:
        rot6d = quaternion_to_rotation6d(tcp[3:7])
        tcp = np.concatenate([tcp[:3], rot6d, tcp[-1:]])
    return tcp

def post_process(tcp, max_gripper_width=0.11):
    trans_min, trans_max = np.array([0, -0.5, 0]), np.array([1.0, 0.5, 0.3])
    tcp[...,:3] = (tcp[...,:3] + 1) / 2.0 * (trans_max - trans_min) + trans_min
    # tcp[...,-1] = (tcp[...,-1] + 1) / 2.0 * max_gripper_width
    if ROT6D:
        quat = rotation6d_to_quaternion(tcp[3:9])
        tcp = np.concatenate([tcp[:3], quat, tcp[-1:]])
    return tcp

def main(args):
    set_seed(1)

    # command line parameters
    ckpt = args['ckpt']
    # robot_cfgs = args['robot_cfgs']
    policy_class = args['policy_class']
    task_name = args['task_name']
    control_freq = args['control_freq']

    # get task parameters
    episode_len = 100 # task_config['episode_len']
    camera_names = ['top', 'wrist'] #task_config['camera_names']
    chunk_size = args['chunk_size']

    # fixed parameters
    # state_dim = 4
    state_dim = 10 if ROT6D else 8
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'num_queries': chunk_size,
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'state_dim': state_dim
                         }
    elif policy_class == 'Diffusion':
        policy_config = {'lr': 0,
                         'num_queries': chunk_size,
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'enc_layers': 4,
                        #  'dec_layers': 4,
                         'dec_layers': 7,
                         'nheads': 8,
                         'state_dim': state_dim
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names, 'state_dim': state_dim}
    else:
        raise NotImplementedError

    config = {
        'ckpt': ckpt,
        # 'robot_cfgs': robot_cfgs,
        'control_freq': control_freq,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'policy_class': policy_class,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        # 'dataset_dir': dataset_dir,
        # 'norm_stats': norm_stats
    }

    eval_bc(config)
    

def make_policy(policy_class, policy_config):
    if policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def eval_bc(config):
    global OFFSET
    set_seed(config["seed"])
    
    task_name = config["task_name"]
    
    # initialize robots and grippers
    print('Init Flexiv arm')
    ready_pose = np.array([0.4, 0.0, 0.17, 0.0, 0.0, 1.0, 0.0])
    # ready_pose = np.array([0.6, 0.10, 0.17, 0.0, 0.0, 1.0, 0.0])
    robot = FlexivApi(robot_ip_address = "192.168.2.100", pc_ip_address = "192.168.2.35", with_streaming = True)
    current_pose = robot.get_tcp_pose()
    current_pose[2] = 0.17
    robot.send_tcp_pose(current_pose)
    robot.streaming()
    time.sleep(0.5)
    robot.send_tcp_pose(ready_pose)
    # initialize grippers
    print('Init Gripper')
    gripper = DahuanModbusGripper("/dev/ttyUSB0", with_streaming = True)
    gripper.streaming()
    gripper.set_force(30)
    gripper.set_width(0)

    # initialize camera(s)
    print('Init camera')
    top_serial = '750612070851'
    wrist_serial = '043322070878'
    os.system('cd dataset/HIReal/camera_sdk/ && bash camera_shm_run.sh &')
    time.sleep(5)

    # # initialize RH100T data transform
    # print('Init RH100T data transform')
    # confs = load_conf("dataset/HIReal/configs.json")
    # calib_folder = '/home/ubuntu/data/calib/1708166065375'
    # conf = confs[0] # when testing with conf 1, simply select the first item
    # projector = RH100TOnline(calib_folder, conf, serial)
    
    # preparation for load policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    ckpt = config['ckpt']
    state_dim = config['state_dim']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    max_timesteps = config['episode_len']
    temporal_agg = config['temporal_agg']
    # norm_stats_file = config['norm_stats']

    # load policy
    policy = make_policy(policy_class, policy_config)
    policy.load_state_dict(torch.load(ckpt, map_location = device))
    policy.to(device)
    policy.eval()
    print(f'Policy loaded: {ckpt}')

    # load max timesteps
    max_timesteps = int(max_timesteps * 20) # may increase for real-world tasks

    # temporal aggregation
    query_frequency = policy_config['num_queries']
    if temporal_agg:
        # query_frequency = 1
        query_frequency = 10
        num_queries = policy_config['num_queries']
        all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).to(device)

    tf = T.Compose([
        T.ToTensor(),
        T.Resize((180, 320)),
        T.CenterCrop([162, 288]),
        T.Resize((180, 320)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # step time
    step_time = 1.0 / config["control_freq"]
    # step_time = 0.3

    fixed_rot = None
    prev_width = None
    prev_rot = None
    with torch.inference_mode():
        for t in range(max_timesteps):
            start_time = time.time()
            # fetch images
            top_image, wrist_image, tcp = get_observation(top_serial, wrist_serial, gripper)
            top_image = tf(Image.fromarray(top_image)).unsqueeze(0).to(device)
            wrist_image = tf(Image.fromarray(wrist_image)).unsqueeze(0).to(device)
            center = tcp[:3].copy()
            ''' max_pos: 800 for voxel_size=0.005, 2000 for voxel_size=0.002
                max_num_token: 200 for voxel_size=0.005, 400 for voxel_size=0.002
            '''
            
            # fetch robot states
            if fixed_rot is None:
                fixed_rot = tcp[3:7]
            # qpos = np.concatenate([tcp[:3], tcp[-1:]], axis=0)
            qpos = tcp
            qpos = torch.from_numpy(pre_process(qpos))
            qpos = qpos.float().unsqueeze(0)
            qpos = qpos.to(device)

            # query policy
            if config['policy_class'] == "ACT" or config['policy_class'] == "Diffusion":
                if t % query_frequency == 0:
                    all_actions = policy(top_image, wrist_image)
                    # all_actions[:,:,:3] -= all_actions[0,0,:3] - qpos[0,:3]
                    # print("***** ", all_actions)
                if temporal_agg:
                    # ########################
                    # # try to ignore outlier
                    # current_traj = all_time_actions[[t], t: t + num_queries]
                    # current_traj[:,-1] = current_traj[:,-2]
                    # if t > 0 and torch.abs(all_actions[:,:-1]-current_traj[:,:-1]).mean() > 0.05:
                    #     all_actions = current_traj
                    # ########################
                    all_time_actions[[t], t: t + num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1).to(device)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]
                # do not z and smooth rotation
                raw_action[:, 2:] = all_actions[:, t % query_frequency, 2:]
            else:
                raise NotImplementedError

            # post-process actions
            raw_action = raw_action.squeeze(0).cpu()
            action = post_process(raw_action.numpy())

            # calculate real actions for robot
            # action_tcp = np.concatenate([action[:3], fixed_rot], axis=0)
            # action[3:7] /= np.linalg.norm(action[3:7])
            action_tcp = action[:7]
            # print(action_tcp[:3])
            action_width = encode_gripper_width(action[-1], 'RH100T_cfg1')
            
            # action
            action_tcp[:3] = action_tcp[:3] * SCALE + OFFSET
            action_tcp[2] = max(action_tcp[2], 0.002)
            print(action_tcp)
            if prev_rot is not None and rot_diff(prev_rot, action_tcp[3:7]) > np.pi/24:
                quat_list = discrete_rotation(prev_rot, action_tcp[3:7])
                for quat in quat_list:
                    action_tcp[3:7] = quat
                    robot.send_tcp_pose(action_tcp)
                    time.sleep(0.4)
            else:
                robot.send_tcp_pose(action_tcp)
            prev_rot = action_tcp[3:7]
            # action_width = min(action_width*1.5, 1000)
            action_width = min(action_width, 1000)
            if prev_width is None or abs(prev_width-action_width) > 200:
                prev_width = action_width
                try:
                    gripper.set_width(int(action_width))
                    time.sleep(0.5)
                except Exception as e:
                    print('[Gripper] Error in main:', e)
                    pass
            duration = time.time() - start_time
            print(duration)
            if duration < step_time:
                time.sleep(step_time - duration)

    robot.stop_streaming()
    gripper.stop_streaming()
    robot.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', action='store', type=str, help='checkpoint file', required=True)
    # parser.add_argument('--robot_cfgs', action='store', type=str, help='real-robot evaluation config file', required=True)
    parser.add_argument('--control_freq', action='store', type=float, help='control frequency', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    try:
        os.system("kill -9 `ps -ef | grep camera_shm | grep -v grep | awk '{print $2}'`")
        os.system("kill -9 `ps -ef | grep get_highfreq | grep -v grep | awk '{print $2}'`")
        os.system('rm -f /dev/shm/*')
        os.system('udevadm trigger')
    
        main(vars(parser.parse_args()))

    except InterruptedError:
        pass

    os.system("kill -9 `ps -ef | grep camera_shm | grep -v grep | awk '{print $2}'`")
    os.system("kill -9 `ps -ef | grep get_highfreq | grep -v grep | awk '{print $2}'`")
    os.system('rm -f /dev/shm/*')
