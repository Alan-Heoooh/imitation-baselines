import os
import json
import numpy as np
# import open3d as o3d
from PIL import Image
from tqdm import tqdm
from transforms3d.euler import quat2euler
from transforms3d.quaternions import quat2mat, mat2quat

import torch
import torchvision
# from torch._six import container_abcs
import collections.abc as container_abcs
from torch.utils.data import Dataset
# import clip

# TO_TENSOR_KEYS = ['language_embed', 'language_mask', 'top_image_list', 'wrist_image_list', 'input_frame_tcp_normalized', 'target_frame_tcp_normalized', 'padding_mask']

TO_TENSOR_KEYS = ['top_image_list', 'wrist_image_list', 'input_frame_action_normalized', 'target_frame_action_normalized', 'padding_mask']

class RH100TDataset(Dataset):
    def __init__(
        self, 
        root, 
        task_config_list, 
        split='train', 
        num_input=1, 
        horizon=1+20, 
        timestep=-1, 
        filter_thresh=None, 
        image_mean=[0.485, 0.456, 0.406], 
        image_std=[0.229, 0.224, 0.225], 
        voxel_size=0.005, 
        augmentation=False, 
        frame_sample_step=1, 
        centralize_gripper=False, 
        tcp_mode='camera', 
        num_sample='all', 
        rot_6d=False, 
        wrist_cam_id=None
        ):
        assert split in ['train', 'val', 'all']
        assert tcp_mode in ['base', 'camera']
        assert num_sample in ['all', '10', '39']

        self.root = root
        self.split = split
        self.num_input = num_input
        self.horizon = horizon
        self.filter_thresh = filter_thresh if filter_thresh is not None else\
                            {'translation': 0.005, 'rotation': np.pi/12, 'gripper_width': 0.005}
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.voxel_size = voxel_size
        self.augmentation = augmentation
        self.centralize_gripper = centralize_gripper
        self.tcp_mode = tcp_mode
        self.rot_6d = rot_6d
        self.wrist_cam_id = wrist_cam_id
        
        self.input_task_ids = []
        self.input_cam_ids = []
        self.input_task_configs = []
        self.target_frame_ids = []
        self.wrist_frame_ids = []
        self.padding_mask_list = []
        
        self.task_ids, self.cam_ids = load_real_tasks(root, task_config_list, split)


        if split == 'val':
            self.task_ids = self.task_ids[::10]
            self.cam_ids = self.cam_ids[::10]
            self.task_configs = self.task_configs[::10]
            self.split = 'train'
        # if num_sample == '10':
        #     selected_task_indices = [2,3,20,24,32,33,37,40,49,55]
        #     # self.task_ids = self.task_ids[::8]
        #     # self.cam_ids = self.cam_ids[::8]
        #     # self.task_configs = self.task_configs[::8]
        #     self.task_ids = [self.task_ids[i] for i in selected_task_indices]
        #     self.cam_ids = [self.cam_ids[i] for i in selected_task_indices]
        #     self.task_configs = [self.task_configs[i] for i in selected_task_indices]
        # elif num_sample == '39':
        #     self.task_ids = self.task_ids[::2]
        #     self.cam_ids = self.cam_ids[::2]
        #     self.task_configs = self.task_configs[::2]

        num_tasks = len(self.task_ids)
        print(1234567)
        print(num_tasks)
        print('#tasks:', num_tasks)

        unique_cam_ids = []
        for i in tqdm(range(num_tasks), desc='loading data samples...'):
            task_id, cam_id = self.task_ids[i], self.cam_ids[i]

            frame_ids = sorted(os.listdir(os.path.join(self.root, task_id, cam_id)))
            
            # print(frame_ids)

            frame_ids = [int(x.split('.')[0].split('_')[-1]) for x in frame_ids]

            target_frame_ids, padding_mask_list = self._get_input_output_frame_id_lists(frame_ids, num_input=num_input, horizon=horizon, timestep=timestep, frame_sample_step=frame_sample_step)
            self.target_frame_ids += target_frame_ids
            self.padding_mask_list += padding_mask_list
            self.input_task_ids += [task_id] * len(target_frame_ids)
            self.input_cam_ids += [cam_id] * len(target_frame_ids)
            # self.input_task_configs += [task_config] * len(target_frame_ids)
            if cam_id not in unique_cam_ids:
                unique_cam_ids.append(cam_id)

    def __len__(self):
        return len(self.target_frame_ids)

    def _get_frames(self, task_id, cam_id, task_config, finish_time=None):
        color_dir = os.path.join(self.root, task_config, task_id, cam_id, 'color')
        frame_ids = sorted(os.listdir(color_dir))
        frame_ids = [int(x.split('.')[0]) for x in frame_ids]

        if finish_time is not None:
            frame_ids = [x for x in frame_ids if x <= finish_time]
        
        return frame_ids

    def _get_scene_gripper_poses(self, task_id, cam_id, task_config):
        # load tcps
        tcp_path = os.path.join(self.root, task_config, task_id, 'transformed', 'tcp.npy')
        print(tcp_path)
        tcp_list = np.load(tcp_path, allow_pickle=True)[()][cam_id[4:]]
        tcps = [x['tcp'].astype(np.float32) for x in tcp_list]
        tcps = np.array(tcps)

        # # load grippers (only for checking invalid scenes)
        # gripper_path = os.path.join(self.root, task_config, task_id, 'transformed', 'gripper.npy')
        # gripper_list = np.load(gripper_path, allow_pickle=True)[()][cam_id[4:]]
        # grippers = [gripper_list[x]['gripper_command'].astype(np.float32) for x in gripper_list]
        # grippers = np.array(grippers)

        # load gripper widths
        gripper_widths = []
        gripper_info_root = os.path.join(self.root, task_config, task_id, cam_id, 'gripper_info')
        gripper_info_path_list = sorted(os.listdir(gripper_info_root))
        for gripper_info_path in gripper_info_path_list:
            gripper_info_path = os.path.join(gripper_info_root, gripper_info_path)
            gripper_info = np.load(gripper_info_path)
            gripper_width = decode_gripper_width(gripper_info, task_config)
            gripper_widths.append(gripper_width)
        gripper_widths = np.array(gripper_widths)

        return tcps, gripper_widths

    def _filter_scene_frames(self, tcps, gripper_widths):
        kept_frame_indices = [0]
        id1 = 0
        while True:
            for id2 in range(id1 + 1, len(tcps)):
                if self._diff(tcps, gripper_widths, id1, id2):
                    id1 = id2
                    kept_frame_indices.append(id2)
                    break
            if id2 == len(tcps) - 1:
                break
        return kept_frame_indices

    def _diff(self, tcps, gripper_widths, id1, id2):
        if id1 == id2:
            return False
        if id1 > id2:
            id1, id2 = id2, id1

        def _diff_translation(sequence, delta):
            if np.any(np.abs(sequence[id1]-sequence[id2]) > delta):
                return True
            return False
            
        def _diff_rotation(quat_sequence, delta):
            mat1 = quat2mat(quat_sequence[id1])
            mat2 = quat2mat(quat_sequence[id2])
            rot_diff = np.matmul(mat1, mat2.T)
            rot_diff = np.diag(rot_diff).sum()
            rot_diff = min(max((rot_diff-1)/2, -1), 1)
            rot_diff = np.arccos(rot_diff)
            return rot_diff > delta

        if _diff_translation(tcps[:,:3], self.filter_thresh['translation']):
            return True
        if _diff_translation(gripper_widths, self.filter_thresh['gripper_width']):
            return True
        if _diff_rotation(tcps[:,3:7], self.filter_thresh['rotation']):
            return True

        return False
    
    def _get_input_output_frame_id_lists(self, frame_id_list, num_input=2, horizon=16, timestep=300, frame_sample_step=1):
        # num_frame = num_input + 1
        target_frame_ids = []
        padding_mask_list = []

        if len(frame_id_list) < horizon:
            # padding
            frame_id_list = frame_id_list + frame_id_list[-1:] * (horizon-len(frame_id_list))

        if timestep <= 0:
            # ignore timestep
            # padding for the first (num_input-1) frames
            # for i in range(1, num_input):
            #     if i >= len(frame_id_list):
            #         break
            #     target_frame_ids.append(frame_id_list[:1]*(num_input-i) + frame_id_list[:i+horizon-num_input])
            frame_id_list = frame_id_list[0:1] * (num_input-1) * frame_sample_step + frame_id_list
            # for i in range(len(frame_id_list) - horizon):
            #     target_frame_ids.append(frame_id_list[i:i+horizon])
            for i in range(len(frame_id_list)-int(num_input*frame_sample_step)):
                cur_target_frame_ids = frame_id_list[i:i+horizon*frame_sample_step:frame_sample_step]
                padding_mask = np.zeros(horizon, dtype=bool)
                if len(cur_target_frame_ids) < horizon:
                    padding_mask[len(cur_target_frame_ids):] = 1
                    cur_target_frame_ids += [frame_id_list[-1]] * (horizon - len(cur_target_frame_ids))
                target_frame_ids.append(cur_target_frame_ids)
                padding_mask_list.append(padding_mask)
                

        else:
            # flip frame list
            # frame_id_list = frame_id_list[::-1]
            # for i, target_frame_id in enumerate(frame_id_list[:-1-num_latency]):
            #     curr_frame_id_list = [target_frame_id]
            #     curr_frame_ptr = i + 1
            #     while curr_frame_ptr < len(frame_id_list):
            #         if curr_frame_id_list[-1] - frame_id_list[curr_frame_ptr] >= timestep:
            #             curr_frame_id_list.append(frame_id_list[curr_frame_ptr])
            #             if len(curr_frame_id_list) == num_frame:
            #                 break
            #         curr_frame_ptr += 1

            #     # padding
            #     if len(curr_frame_id_list) < num_frame:
            #         curr_frame_id_list += frame_id_list[-1:] * (num_frame - len(curr_frame_id_list))

            #     input_frame_id_lists.append(curr_frame_id_list[1+num_latency:][::-1])
            #     target_frame_ids.append(target_frame_id)
            assert False # not implemented

        return target_frame_ids, padding_mask_list

    # def _clip_tcp(self, tcp_list):
    #     ''' tcp_list: [T, 8]'''
    #     tcp_list[:,0] = np.clip(tcp_list[:,0], 0, 1)
    #     tcp_list[:,1] = np.clip(tcp_list[:,1], -0.5, 0.5)
    #     tcp_list[:,2] = np.clip(tcp_list[:,2], 0, 0.3)
    #     tcp_list[:,7] = np.clip(tcp_list[:,7], 0, 0.11)
    #     return tcp_list

    # def _normalize_tcp(self, tcp_list):
    #     ''' tcp_list: [T, 8]'''
    #     trans_min, trans_max = np.array([0, -0.5, 0]), np.array([1.0, 0.5, 0.3]) # for base frame
    #     max_gripper_width = 0.11 # meter
    #     tcp_list[:,:3] = (tcp_list[:,:3] - trans_min) / (trans_max - trans_min) * 2 - 1
    #     tcp_list[:,7] = tcp_list[:,7] / max_gripper_width * 2 - 1
    #     return tcp_list

    def _normalize_action(self, action_list):
        ''' action_list: [T, 13]'''
        joint_min = np.array([-2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        joint_max = np.array([2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        action_list[:,:7] = (action_list[:,:7] - joint_min) / (joint_max - joint_min) * 2 - 1
        return action_list

    def load_image(self, task_id, cam_id, frame_id):
        color_path = os.path.join(self.root, task_id, cam_id, 'frame_%03d.png'%frame_id)
        colors = np.array(Image.open(color_path), dtype=np.float32) / 255.0
        # imagenet normalization
        colors = (colors - self.image_mean) / self.image_std
        colors = colors.transpose([2, 0, 1])
        colors = torch.from_numpy(colors.astype(np.float32))
        return colors
    
    def resize_image(self, image, image_size=[360,640]):
        h, w = image_size
        resize = torchvision.transforms.Resize([h, w])
        image = resize(image)
        return image

    def __getitem__(self, index):
        task_id = self.input_task_ids[index]
        # input_frame_ids = self.input_frame_id_lists[index]
        target_frame_ids = self.target_frame_ids[index]
        padding_mask = self.padding_mask_list[index]
        cam_id = self.input_cam_ids[index]
        # task_config = self.input_task_configs[index]
        # tcp_list = self.tcp_dicts[task_id][cam_id[4:]]
        # gripper_list = self.gripper_dicts[task_id][cam_id[4:]]

        # gripper_info_dir = os.path.join(self.root, self.split, task_id, cam_id, 'gripper_info')
        # gripper_command_dir = os.path.join(self.root, self.split, task_id, cam_id, 'gripper_command')
        # tcp_dir = os.path.join(self.root, self.split, task_id, cam_id, 'tcp')
        
        states = torch.load(os.path.join(self.root, task_id, 'states.pt'))
        

        # load input and target gripper pose and gripper width
        # target_frame_tcp_list = []
        # target_gripper_width_list = []
        states_traj = states['traj_0']
        states_traj_actions = states_traj['actions']
        states_traj_actions_right_arm = states_traj_actions['right_arm']

        target_joint_list = []
        target_hand_qpos_list = []
        for i,fid in enumerate(target_frame_ids):
            # tcp_path = os.path.join(tcp_dir, '%d.npy'%fid)
            # target_frame_tcp_list.append(np.load(tcp_path))
            if fid == states_traj_actions_right_arm.shape[0]:
                fid = states_traj_actions_right_arm.shape[0] - 1
            target_joint_list.append(states_traj_actions_right_arm[fid][:7])
            # if i < self.num_input:
            #     gripper_path = os.path.join(gripper_info_dir, '%d.npy'%fid)
            # else:
            #     gripper_path = os.path.join(gripper_command_dir, '%d.npy'%fid)
            # gripper_state = np.load(gripper_path)
            # gripper_width = decode_gripper_width(gripper_state, task_config)
            # target_gripper_width_list.append(gripper_width)
            target_hand_qpos_list.append(states_traj_actions_right_arm[fid][7:])
        # target_frame_tcp_list = np.array(target_frame_tcp_list, dtype=np.float32)
        target_joint_list = np.array(target_joint_list, dtype=np.float32)
        target_hand_qpos_list = np.array(target_hand_qpos_list, dtype=np.float32)


        # load input rgbs
        top_image_list = []
        wrist_image_list = []
        for i, input_frame_id in enumerate(target_frame_ids[:self.num_input]):
            top_image = self.load_image(task_id, cam_id, input_frame_id)
            top_image = self.resize_image(top_image)
            top_image_list.append(top_image)
            if self.wrist_cam_id is not None:
                wrist_frame_id = self.wrist_frame_ids[index]
                wrist_image = self.load_image(task_id, self.wrist_cam_id, wrist_frame_id)
                wrist_image = self.resize_image(wrist_image)
                wrist_image_list.append(wrist_image)
        top_image_list = np.stack(top_image_list, axis=0)

        # # visualization
        # points_vis = input_cloud_list[0]
        # tcp_vis = target_frame_tcp_list[0][:3]
        # tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(tcp_vis)
        # cloud_vis = o3d.geometry.PointCloud()
        # cloud_vis.points = o3d.utility.Vector3dVector(points_vis[:,:3])
        # cloud_vis.colors = o3d.utility.Vector3dVector(points_vis[:,3:]*self.image_std+self.image_mean)
        # o3d.visualization.draw_geometries([cloud_vis.voxel_down_sample(0.005), tcp_vis])

        # target_gripper_width_list = np.array(target_gripper_width_list, dtype=np.float32)[:,np.newaxis]
        # target_frame_tcp_list = np.concatenate([target_frame_tcp_list, target_gripper_width_list], axis=-1)

        target_action_list = np.concatenate([target_joint_list, target_hand_qpos_list], axis=-1)

        # get normalized tcp
        # target_frame_tcp_list = np.array(target_frame_tcp_list, dtype=np.float32)
        target_action_list = np.array(target_action_list, dtype=np.float32)
        # target_frame_tcp_list = self._clip_tcp(target_frame_tcp_list)
        # target_frame_tcp_normalized = self._normalize_tcp(target_frame_tcp_list.copy())
        target_action_normalized = self._normalize_action(target_action_list.copy())
        # transform quaternion to 6d rotation
        # if self.rot_6d:
        #     target_frame_rotation_6d = batch_quaternion_to_rotation6d(target_frame_tcp_normalized[:,3:7])
        #     target_frame_tcp_normalized = np.concatenate([target_frame_tcp_normalized[:,0:3], target_frame_rotation_6d, target_frame_tcp_normalized[:,-1:]], axis=-1)

        # split data
        # input_frame_tcp_list = target_frame_tcp_list[:self.num_input]
        # target_frame_tcp_list = target_frame_tcp_list[self.num_input:]
        input_frame_action_list = target_action_list[:self.num_input]
        target_frame_action_list = target_action_list[self.num_input:]

        # input_frame_tcp_normalized = target_frame_tcp_normalized[:self.num_input]
        # target_frame_tcp_normalized = target_frame_tcp_normalized[self.num_input:]
        input_frame_action_normalized = target_action_normalized[:self.num_input]
        target_frame_action_normalized = target_action_normalized[self.num_input:]
        padding_mask = padding_mask[self.num_input:]

        # convert to torch
        # input_frame_tcp_normalized = torch.from_numpy(input_frame_tcp_normalized)
        # target_frame_tcp_normalized = torch.from_numpy(target_frame_tcp_normalized)
        input_frame_action_normalized = torch.from_numpy(input_frame_action_normalized)
        target_frame_action_normalized = torch.from_numpy(target_frame_action_normalized)
        padding_mask = torch.from_numpy(padding_mask)
        # language_embed = torch.from_numpy(language_embed)
        # language_mask = torch.from_numpy(language_mask)

        if self.num_input == 1:
            top_image_list = top_image_list[0]
            # wrist_image_list = wrist_image_list[0]
            input_frame_action_list = input_frame_action_list[0]
            input_frame_action_normalized = input_frame_action_normalized[0]

        ret_dict = {
                    # 'instruction': instruction,
                    # 'language_embed': language_embed,
                    # 'language_mask': language_mask,
                    'top_image_list': top_image_list,
                    # 'wrist_image_list': wrist_image_list,
                    # 'input_frame_tcp_list': input_frame_tcp_list,
                    # 'input_frame_tcp_normalized': input_frame_tcp_normalized,
                    # 'target_frame_tcp_list': target_frame_tcp_list,
                    # 'target_frame_tcp_normalized': target_frame_tcp_normalized,
                    'input_frame_action_list': input_frame_action_list,
                    'target_frame_action_list': target_frame_action_list,
                    'input_frame_action_normalized': input_frame_action_normalized,
                    'target_frame_action_normalized': target_frame_action_normalized,
                    'padding_mask': padding_mask,
                    'task_id': task_id,
                    'target_frame_ids': target_frame_ids,
                    'cam_id': cam_id}

        return ret_dict
        

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        # return [torch.from_numpy(sample) for b in batch for sample in b]
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))

def convert_data_to_gpu(batch, device_id=0):
    for key in batch.keys():
        if key not in TO_TENSOR_KEYS:
            continue
        batch[key] = batch[key].to(device_id)
    return batch

def load_selected_tasks(task_list_path):
    task_dict = json.load(open(task_list_path))
    task_ids = []
    cam_ids = []
    task_configs = []
    for task_config in task_dict:
        for task_id in task_dict[task_config]:
            cur_cam_ids = task_dict[task_config][task_id]
            cur_cam_ids = [x for x in cur_cam_ids if x != 'cam_043322070878']
            task_ids += [task_id] * len(cur_cam_ids)
            cam_ids += cur_cam_ids
            task_configs += [task_config] * len(cur_cam_ids)
    return task_ids, cam_ids, task_configs

def load_real_tasks(task_root, task_configs, split='train'):
    assert split in ['train', 'val', 'all']
    task_ids = []
    cam_ids = []
    # task_root = os.path.join(task_root, split)
    unique_task_ids = sorted(os.listdir(task_root))
    ##########################
    # # for grasping experiments
    # unique_task_ids = [x for x in unique_task_ids if 'task_050' in x or 'task_0510' in x]
    ##########################
    ##########################
    # for iros experiments (511,512,513,514,515,516,517,518,519,520,521)
    # if unique_task_ids:
    #     # Extract the first task ID from the first directory name
    #     # E.g., from 'task_0703_user_0099_scene_0001_cfg_0001', extract 'task_0703'
    #     target_task_id = unique_task_ids[0].split('_user_')[0]
    #     print(f"Dynamically selected target task ID: {target_task_id}")
    # else:
    #     raise ValueError(f"No task directories found in {task_root}")
    # print(target_task_id)
    # unique_task_ids = [x for x in unique_task_ids if target_task_id in x]

    print(unique_task_ids)

    for task_id in unique_task_ids:
        task_dir = os.path.join(task_root, task_id)
        cur_cam_ids = os.listdir(task_dir)
        cur_cam_ids = ['cam_3']
        for cam_id in cur_cam_ids:
            task_ids.append(task_id)
            cam_ids.append(cam_id)
    return task_ids, cam_ids

def load_all_tasks(task_root, task_configs, split='train', cache_dict=None, inference_mode=False):
    assert split in ['train', 'val', 'all']
    task_ids = []
    cam_ids = []
    config_ids = []
    # if split == 'train':
    #     selected_scenes = list(range(1,10))
    # elif split == 'val':
    #     selected_scenes = [10]
    # else:
    #     selected_scenes = list(range(1,11))

    def _validate_scene(scene_dir, cam_id):
        color_dir = os.path.join(scene_dir, cam_id, 'color')
        gripper_info_dir = os.path.join(scene_dir, cam_id, 'gripper_info')
        if not os.path.exists(color_dir):
            # print(scene_dir, cam_id)
            return False
        if not os.path.exists(gripper_info_dir):
            # print(scene_dir, cam_id)
            return False
        if len(os.listdir(color_dir)) <= 10:
            return False
        if len(os.listdir(color_dir)) != len(os.listdir(gripper_info_dir)):
            # print(scene_dir, cam_id)
            return False
        return True

    def _search_cam_ids(scene_dir):
        cur_cam_ids = os.listdir(scene_dir)
        cur_cam_ids = [x for x in cur_cam_ids if x[:4] == 'cam_']
        return cur_cam_ids

    # def _get_scene_id(task_id):
    #     scene_id = int(task_id.split('_')[5])
    #     return scene_id

    def _get_user_id(task_id):
        return task_id[10:19]

    def _search_user_ids(task_ids):
        user_ids = {_get_user_id(tid) for tid in task_ids}
        user_ids = sorted(list(user_ids))
        return user_ids

    def _get_val_user_ids(task_ids):
        val_user_ids = {}
        for tid in task_ids:
            if tid[:9] not in val_user_ids:
                val_user_ids[tid[:9]] = _get_user_id(tid)
            else:
                val_user_ids[tid[:9]] = max(val_user_ids[tid[:9]], _get_user_id(tid))
        return val_user_ids

    def _split_train_val_task_ids(task_ids, split='train'):
        assert split in ['train', 'val']
        val_user_ids = _get_val_user_ids(task_ids)
        if split == 'train':
            split_task_ids = [tid for tid in task_ids if val_user_ids[tid[:9]]!=_get_user_id(tid)]
        else:
            split_task_ids = [tid for tid in task_ids if val_user_ids[tid[:9]]==_get_user_id(tid)]
        return split_task_ids

    if cache_dict is not None:
        for task_config in task_configs:
            cur_task_ids = sorted(cache_dict[task_config].keys())

            # split data by user_ids, the last user is used as val
            if split != 'all':
                if not inference_mode:
                    cur_user_ids = _search_user_ids(cur_task_ids)
                    cur_user_ids = cur_user_ids[:-1] if split == 'train' else cur_user_ids[-1:]
                    cur_task_ids = [tid for tid in cur_task_ids if _get_user_id(tid) in cur_user_ids]
                else:
                    cur_task_ids = _split_train_val_task_ids(cur_task_ids, split)

            for task_id in cur_task_ids:
                # if task_id[:9] in IGNORED_TASKS:
                if task_id[:9] not in PICK_PLACE_TASKS:
                # if task_id[:9] not in FINETUNE_TASKS:
                    continue
                # if _get_scene_id(task_id) not in selected_scenes:
                #     continue
                cur_cam_ids = sorted(cache_dict[task_config][task_id])
                for cam_id in cur_cam_ids:
                    task_ids.append(task_id)
                    cam_ids.append(cam_id)
                    config_ids.append(task_config)
        return task_ids, cam_ids, config_ids

    for task_config in task_configs:
        cur_task_ids = os.listdir(os.path.join(task_root, task_config))
        cur_task_ids = [x for x in cur_task_ids if x[:4] == 'task' and 'human' not in x]
        cur_task_ids = sorted(cur_task_ids)

        # split data by user_ids, the last user is used as val
        if split != 'all':
            cur_user_ids = _search_user_ids(cur_task_ids)
            cur_user_ids = cur_user_ids[:-1] if split == 'train' else cur_user_ids[-1:]
            cur_task_ids = [tid for tid in cur_task_ids if _get_user_id(tid) in cur_user_ids]

        for task_id in tqdm(cur_task_ids, desc=task_config):
            if task_id[:9] in IGNORED_TASKS:
                continue
            # if _get_scene_id(task_id) not in selected_scenes:
            #     # print(task_id)
            #     continue
            scene_dir = os.path.join(task_root, task_config, task_id)
            cur_cam_ids = _search_cam_ids(scene_dir)
            for cam_id in cur_cam_ids:
                if not _validate_scene(scene_dir, cam_id):
                    # print(scene_dir, cam_id)
                    continue
                task_ids.append(task_id)
                cam_ids.append(cam_id)
                config_ids.append(task_config)

    return task_ids, cam_ids, config_ids

def create_point_cloud_from_depth_image(depths, camera_intrinsics, scale):
    xmap = np.arange(depths.shape[1])
    ymap = np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    fx, fy = camera_intrinsics[0,0], camera_intrinsics[1,1]
    cx, cy = camera_intrinsics[0,2], camera_intrinsics[1,2]
    points_z = depths.astype(np.float32) / scale
    # points_z = np.clip(points_z, minval, maxval)
    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    return cloud.astype(np.float32)

def decode_gripper_width(gripper_state, task_config):
    ''' process gripper state to real width (in meters), used for gripper_info and gripper_command
        gripper states in transformed folder do not need this function
    '''
    assert task_config in ['RH100T_cfg%d' % x for x in range(1,8)]

    gripper_width = gripper_state[0]
    if task_config in ['RH100T_cfg1', 'RH100T_cfg2']:
        # Dahuan AG-95
        gripper_width = gripper_width / 1000. * 0.095
    elif task_config in ['RH100T_cfg3']:
        # WSG-50
        gripper_width = gripper_width / 100.
    elif task_config in ['RH100T_cfg4', 'RH100T_cfg6', 'RH100T_cfg7']:
        # Robotiq 2F-85
        gripper_width = (255. - gripper_width) / 255. * 0.085
    elif task_config in ['RH100T_cfg5']:
        # Panda
        gripper_width = gripper_width / 100000.
    else:
        print('wrong')

    return gripper_width

def encode_gripper_width(gripper_width, task_config):
    ''' process real gripper width (in meters) to command, used for gripper_info and gripper_command
        gripper states in transformed folder do not need this function
    '''
    assert task_config in ['RH100T_cfg%d' % x for x in range(1,8)]

    if task_config in ['RH100T_cfg1', 'RH100T_cfg2']:
        # Dahuan AG-95
        gripper_width = gripper_width / 0.095 * 1000.
        gripper_width = max(0, min(1000, gripper_width))
    elif task_config in ['RH100T_cfg3']:
        # WSG-50
        gripper_width = gripper_width * 100.
    elif task_config in ['RH100T_cfg4', 'RH100T_cfg6', 'RH100T_cfg7']:
        # Robotiq 2F-85
        gripper_width = 255. - gripper_width / 0.085 * 255.
        gripper_width = max(0, min(255, gripper_width))
    elif task_config in ['RH100T_cfg5']:
        # Panda
        gripper_width = gripper_width * 100000.
    else:
        print('wrong')

    return gripper_width

def parse_action_preds(action_preds, max_gripper_width=0.11, tcp_in_center=False, center=None, tcp_mode='camera'):
    ''' logits: numpy.ndarray, [B,T,8]
    '''
    # if tcp_in_center:
    #     trans_min, trans_max = np.array([-0.15, -0.15, -0.15]), np.array([0.15, 0.15, 0.15]) # TODO: need adjustification
    # else:
    #     trans_min, trans_max = np.array([-0.64, -0.64, 0]), np.array([0.64, 0.64, 1.28]) # TODO: need adjustification
    assert tcp_mode in ['base', 'camera']
    if tcp_mode == 'camera':
        trans_min, trans_max = np.array([-0.35, -0.35, 0]), np.array([0.35, 0.35, 0.7]) # for camera frame
    else:
        trans_min, trans_max = np.array([0.25, -0.35, 0]), np.array([0.8, 0.35, 0.3]) # for base frame

    trans_preds = action_preds[...,0:3]
    trans_preds = (trans_preds + 1) / 2.0
    trans_preds = trans_preds * (trans_max - trans_min) + trans_min
    if tcp_in_center and center is not None:
        trans_preds = trans_preds + center

    # quat_preds = action_preds[...,3:7]
    # quat_preds /=  np.linalg.norm(quat_preds, axis=2, keepdims=True) + 1e-6

    gripper_width_preds = action_preds[...,-1:]
    gripper_width_preds = (gripper_width_preds + 1) / 2.0
    gripper_width_preds = gripper_width_preds * max_gripper_width

    # action_preds = np.concatenate([trans_preds, quat_preds, gripper_width_preds], axis=-1)
    action_preds = np.concatenate([trans_preds, gripper_width_preds], axis=-1)

    return action_preds

def compute_action_error(action_preds, action_labels):
    trans_error = np.linalg.norm(action_preds[...,:3]-action_labels[...,:3], axis=-1).mean()
    width_error = np.abs(action_preds[...,-1]-action_labels[...,-1]).mean()
    return trans_error, width_error

def batch_quaternion_to_rotation6d(batch_quat):
    ''' batch_quat: [N,4]
    '''
    # quaternion to matrix
    batch_matrix = np.zeros([*batch_quat.shape[:-1],3,3], dtype=np.float32)
    for i in range(batch_quat.shape[0]):
        batch_matrix[i] = quat2mat(batch_quat[i])

    # matrix to 6d
    batch_dim = batch_matrix.shape[:-2]
    batch_rot6d = batch_matrix[...,:2,:].reshape(*batch_dim, 6)

    return batch_rot6d

def batch_rotation6d_to_quaternion(batch_rot6d):
    ''' batch_rot6d: [N,6]
    '''
    # 6d to matrix
    a1, a2 = batch_dim[..., :3], batch_dim[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-6)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    batch_matrix = np.stack([b1,b2,b3], axis=-2)

    # matrix to quat
    batch_quat = np.zeros([batch_matrix.shape[0],4], dtype=np.float32)
    for i in range(batch_matrix.shape[0]):
        batch_quat[i] = mat2quat(batch_matrix[i])

    return batch_quat


if __name__ == "__main__":
    dataset_root = '/aidata'
    dataset_root = '/zihao-fast-vol/vr_data'
    # task_config_list = ['RH100T_cfg1','RH100T_cfg2','RH100T_cfg3','RH100T_cfg4','RH100T_cfg5','RH100T_cfg6','RH100T_cfg7']
    task_config_list = ['RH100T_cfg1']
    dataset = RH100TDataset(dataset_root, task_config_list, 'train', num_input=1, horizon=16, timestep=-1, augmentation=False, tcp_mode='base')
    print(len(dataset))
    xmin, xmax, ymin, ymax, zmin, zmax = 1, -1, 1, -1, 1, -1
    for i in tqdm(range(0,len(dataset))):
        batch = dataset[i]
        input_frame_action_list = batch['input_frame_action_list']
        action_list = batch['target_frame_action_list']
        top_image_list = batch['top_image_list']
        print(input_frame_action_list.shape)
        print(action_list.shape)
        print(top_image_list.shape)


        