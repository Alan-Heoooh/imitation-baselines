import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
# from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import quat2euler
from transforms3d.quaternions import quat2mat, mat2quat
# import open3d as o3d

import torch
# from torch._six import container_abcs
import collections.abc as container_abcs
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import clip

TO_TENSOR_KEYS = ['language_embed', 'language_mask', 'input_coords_list', 'input_feats_list', 'input_frame_tcp_normalized', 'target_frame_tcp_normalized', 'padding_mask']
PROBLEM_TASKS = [('task_0091_user_0015_scene_0003_cfg_0001', 'cam_038522063145'),
                 ('task_0100_user_0007_scene_0006_cfg_0002', 'cam_037522062165'),
                 ('task_0001_user_0016_scene_0001_cfg_0003', 'cam_038522062288'),
                 ('task_0001_user_0016_scene_0001_cfg_0003', 'cam_104422070011'),
                 ('task_0054_user_0010_scene_0001_cfg_0003', 'cam_038522062288'),
                 ('task_0054_user_0010_scene_0001_cfg_0003', 'cam_104422070011'),
                 ('task_0054_user_0010_scene_0002_cfg_0003', 'cam_038522062288'),
                 ('task_0054_user_0010_scene_0002_cfg_0003', 'cam_104422070011'),
                 ('task_0054_user_0010_scene_0004_cfg_0003', 'cam_038522062288'),
                 ('task_0054_user_0010_scene_0004_cfg_0003', 'cam_104422070011'),
                 ('task_0054_user_0010_scene_0005_cfg_0003', 'cam_038522062288'),
                 ('task_0054_user_0010_scene_0005_cfg_0003', 'cam_104422070011'),
                 ('task_0054_user_0010_scene_0006_cfg_0003', 'cam_038522062288'),
                 ('task_0054_user_0010_scene_0006_cfg_0003', 'cam_104422070011'),
                 ('task_0054_user_0010_scene_0007_cfg_0003', 'cam_038522062288'),
                 ('task_0054_user_0010_scene_0007_cfg_0003', 'cam_104422070011'),
                 ('task_0054_user_0010_scene_0008_cfg_0003', 'cam_038522062288'),
                 ('task_0054_user_0010_scene_0008_cfg_0003', 'cam_104422070011'),
                 ('task_0054_user_0010_scene_0009_cfg_0003', 'cam_038522062288'),
                 ('task_0054_user_0010_scene_0009_cfg_0003', 'cam_104422070011'),
                 ('task_0054_user_0010_scene_0010_cfg_0003', 'cam_038522062288'),
                 ('task_0054_user_0010_scene_0010_cfg_0003', 'cam_104422070011'),
                 ('task_0066_user_0010_scene_0001_cfg_0003', 'cam_038522062288'),
                 ('task_0066_user_0010_scene_0001_cfg_0003', 'cam_104422070011'),
                 ('task_0066_user_0010_scene_0004_cfg_0003', 'cam_038522062288'),
                 ('task_0066_user_0010_scene_0004_cfg_0003', 'cam_104422070011'),
                 ('task_0066_user_0010_scene_0005_cfg_0003', 'cam_038522062288'),
                 ('task_0066_user_0010_scene_0005_cfg_0003', 'cam_104422070011'),
                 ('task_0066_user_0010_scene_0006_cfg_0003', 'cam_038522062288'),
                 ('task_0066_user_0010_scene_0006_cfg_0003', 'cam_104422070011'),
                 ('task_0066_user_0010_scene_0007_cfg_0003', 'cam_038522062288'),
                 ('task_0066_user_0010_scene_0007_cfg_0003', 'cam_104422070011'),
                 ('task_0066_user_0010_scene_0008_cfg_0003', 'cam_038522062288'),
                 ('task_0066_user_0010_scene_0008_cfg_0003', 'cam_104422070011'),
                 ('task_0066_user_0010_scene_0009_cfg_0003', 'cam_038522062288'),
                 ('task_0066_user_0010_scene_0009_cfg_0003', 'cam_104422070011'),
                 ('task_0066_user_0010_scene_0010_cfg_0003', 'cam_038522062288'),
                 ('task_0066_user_0010_scene_0010_cfg_0003', 'cam_104422070011'),
                 ('task_0067_user_0010_scene_0001_cfg_0003', 'cam_038522062288'),
                 ('task_0067_user_0010_scene_0001_cfg_0003', 'cam_104422070011'),
                 ('task_0067_user_0010_scene_0002_cfg_0003', 'cam_038522062288'),
                 ('task_0067_user_0010_scene_0002_cfg_0003', 'cam_104422070011'),
                 ('task_0067_user_0010_scene_0003_cfg_0003', 'cam_038522062288'),
                 ('task_0067_user_0010_scene_0003_cfg_0003', 'cam_104422070011'),
                 ('task_0067_user_0010_scene_0004_cfg_0003', 'cam_038522062288'),
                 ('task_0067_user_0010_scene_0004_cfg_0003', 'cam_104422070011'),
                 ('task_0067_user_0010_scene_0005_cfg_0003', 'cam_038522062288'),
                 ('task_0067_user_0010_scene_0005_cfg_0003', 'cam_104422070011'),
                 ('task_0067_user_0010_scene_0006_cfg_0003', 'cam_038522062288'),
                 ('task_0067_user_0010_scene_0006_cfg_0003', 'cam_104422070011'),
                 ('task_0067_user_0010_scene_0007_cfg_0003', 'cam_038522062288'),
                 ('task_0067_user_0010_scene_0007_cfg_0003', 'cam_104422070011'),
                 ('task_0067_user_0010_scene_0008_cfg_0003', 'cam_038522062288'),
                 ('task_0067_user_0010_scene_0008_cfg_0003', 'cam_104422070011'),
                 ('task_0067_user_0010_scene_0009_cfg_0003', 'cam_038522062288'),
                 ('task_0067_user_0010_scene_0009_cfg_0003', 'cam_104422070011'),
                 ('task_0067_user_0010_scene_0010_cfg_0003', 'cam_038522062288'),
                 ('task_0067_user_0010_scene_0010_cfg_0003', 'cam_104422070011'),
                 ('task_0076_user_0010_scene_0001_cfg_0003', 'cam_038522062288'),
                 ('task_0076_user_0010_scene_0001_cfg_0003', 'cam_104422070011'),
                 ('task_0076_user_0010_scene_0002_cfg_0003', 'cam_038522062288'),
                 ('task_0076_user_0010_scene_0002_cfg_0003', 'cam_104422070011'),
                 ('task_0076_user_0010_scene_0003_cfg_0003', 'cam_038522062288'),
                 ('task_0076_user_0010_scene_0003_cfg_0003', 'cam_104422070011'),
                 ('task_0076_user_0010_scene_0004_cfg_0003', 'cam_038522062288'),
                 ('task_0076_user_0010_scene_0004_cfg_0003', 'cam_104422070011'),
                 ('task_0076_user_0010_scene_0005_cfg_0003', 'cam_038522062288'),
                 ('task_0076_user_0010_scene_0005_cfg_0003', 'cam_104422070011'),
                 ('task_0076_user_0010_scene_0006_cfg_0003', 'cam_038522062288'),
                 ('task_0076_user_0010_scene_0006_cfg_0003', 'cam_104422070011'),
                 ('task_0076_user_0010_scene_0007_cfg_0003', 'cam_038522062288'),
                 ('task_0076_user_0010_scene_0007_cfg_0003', 'cam_104422070011'),
                 ('task_0076_user_0010_scene_0008_cfg_0003', 'cam_038522062288'),
                 ('task_0076_user_0010_scene_0008_cfg_0003', 'cam_104422070011'),
                 ('task_0076_user_0010_scene_0009_cfg_0003', 'cam_038522062288'),
                 ('task_0076_user_0010_scene_0009_cfg_0003', 'cam_104422070011'),
                 ('task_0076_user_0010_scene_0010_cfg_0003', 'cam_038522062288'),
                 ('task_0076_user_0010_scene_0010_cfg_0003', 'cam_104422070011'),
                 ('task_0077_user_0010_scene_0001_cfg_0003', 'cam_038522062288'),
                 ('task_0077_user_0010_scene_0001_cfg_0003', 'cam_104422070011'),
                 ('task_0077_user_0010_scene_0002_cfg_0003', 'cam_038522062288'),
                 ('task_0077_user_0010_scene_0002_cfg_0003', 'cam_104422070011'),
                 ('task_0077_user_0010_scene_0003_cfg_0003', 'cam_038522062288'),
                 ('task_0077_user_0010_scene_0003_cfg_0003', 'cam_104422070011'),
                 ('task_0077_user_0010_scene_0004_cfg_0003', 'cam_038522062288'),
                 ('task_0077_user_0010_scene_0004_cfg_0003', 'cam_104422070011'),
                 ('task_0077_user_0010_scene_0005_cfg_0003', 'cam_038522062288'),
                 ('task_0077_user_0010_scene_0005_cfg_0003', 'cam_104422070011'),
                 ('task_0077_user_0010_scene_0006_cfg_0003', 'cam_038522062288'),
                 ('task_0077_user_0010_scene_0006_cfg_0003', 'cam_104422070011'),
                 ('task_0077_user_0010_scene_0007_cfg_0003', 'cam_038522062288'),
                 ('task_0077_user_0010_scene_0007_cfg_0003', 'cam_104422070011'),
                 ('task_0077_user_0010_scene_0008_cfg_0003', 'cam_038522062288'),
                 ('task_0077_user_0010_scene_0008_cfg_0003', 'cam_104422070011'),
                 ('task_0077_user_0010_scene_0009_cfg_0003', 'cam_038522062288'),
                 ('task_0077_user_0010_scene_0009_cfg_0003', 'cam_104422070011'),
                 ('task_0077_user_0010_scene_0010_cfg_0003', 'cam_038522062288'),
                 ('task_0077_user_0010_scene_0010_cfg_0003', 'cam_104422070011'),
                 ('task_0091_user_0010_scene_0001_cfg_0003', 'cam_038522062288'),
                 ('task_0091_user_0010_scene_0001_cfg_0003', 'cam_104422070011'),
                 ('task_0091_user_0010_scene_0002_cfg_0003', 'cam_038522062288'),
                 ('task_0091_user_0010_scene_0002_cfg_0003', 'cam_104422070011'),
                 ('task_0091_user_0010_scene_0003_cfg_0003', 'cam_038522062288'),
                 ('task_0091_user_0010_scene_0003_cfg_0003', 'cam_104422070011'),
                 ('task_0091_user_0010_scene_0004_cfg_0003', 'cam_038522062288'),
                 ('task_0091_user_0010_scene_0004_cfg_0003', 'cam_104422070011'),
                 ('task_0091_user_0010_scene_0005_cfg_0003', 'cam_038522062288'),
                 ('task_0091_user_0010_scene_0005_cfg_0003', 'cam_104422070011'),
                 ('task_0091_user_0010_scene_0006_cfg_0003', 'cam_038522062288'),
                 ('task_0091_user_0010_scene_0006_cfg_0003', 'cam_104422070011'),
                 ('task_0091_user_0010_scene_0007_cfg_0003', 'cam_038522062288'),
                 ('task_0091_user_0010_scene_0007_cfg_0003', 'cam_104422070011'),
                 ('task_0091_user_0010_scene_0009_cfg_0003', 'cam_038522062288'),
                 ('task_0091_user_0010_scene_0009_cfg_0003', 'cam_104422070011'),
                 ('task_0091_user_0010_scene_0010_cfg_0003', 'cam_038522062288'),
                 ('task_0091_user_0010_scene_0010_cfg_0003', 'cam_104422070011'),
                 ('task_0092_user_0010_scene_0001_cfg_0003', 'cam_038522062288'),
                 ('task_0092_user_0010_scene_0001_cfg_0003', 'cam_104422070011'),
                 ('task_0092_user_0010_scene_0002_cfg_0003', 'cam_038522062288'),
                 ('task_0092_user_0010_scene_0002_cfg_0003', 'cam_104422070011'),
                 ('task_0092_user_0010_scene_0003_cfg_0003', 'cam_038522062288'),
                 ('task_0092_user_0010_scene_0003_cfg_0003', 'cam_104422070011'),
                 ('task_0092_user_0010_scene_0004_cfg_0003', 'cam_038522062288'),
                 ('task_0092_user_0010_scene_0004_cfg_0003', 'cam_104422070011'),
                 ('task_0092_user_0010_scene_0005_cfg_0003', 'cam_038522062288'),
                 ('task_0092_user_0010_scene_0005_cfg_0003', 'cam_104422070011'),
                 ('task_0092_user_0010_scene_0006_cfg_0003', 'cam_038522062288'),
                 ('task_0092_user_0010_scene_0006_cfg_0003', 'cam_104422070011'),
                 ('task_0092_user_0010_scene_0007_cfg_0003', 'cam_038522062288'),
                 ('task_0092_user_0010_scene_0007_cfg_0003', 'cam_104422070011'),
                 ('task_0092_user_0010_scene_0008_cfg_0003', 'cam_038522062288'),
                 ('task_0092_user_0010_scene_0008_cfg_0003', 'cam_104422070011'),
                 ('task_0092_user_0010_scene_0009_cfg_0003', 'cam_038522062288'),
                 ('task_0092_user_0010_scene_0009_cfg_0003', 'cam_104422070011'),
                 ('task_0092_user_0010_scene_0010_cfg_0003', 'cam_038522062288'),
                 ('task_0092_user_0010_scene_0010_cfg_0003', 'cam_104422070011'),
                 ('task_0002_user_0010_scene_0005_cfg_0004', 'cam_104422070011'),
                 ('task_0003_user_0010_scene_0005_cfg_0004', 'cam_036422060909'),
                 ('task_0005_user_0010_scene_0002_cfg_0004', 'cam_036422060909'),
                 ('task_0007_user_0010_scene_0003_cfg_0004', 'cam_104122062823'),
                 ('task_0007_user_0010_scene_0007_cfg_0004', 'cam_104122062823'),
                 ('task_0044_user_0014_scene_0010_cfg_0004', 'cam_f0172289'),
                 ('task_0045_user_0014_scene_0008_cfg_0004', 'cam_036422060909'),
                 ('task_0011_user_0007_scene_0009_cfg_0005', 'cam_036422060215'),
                 ('task_0011_user_0007_scene_0009_cfg_0005', 'cam_037522062165'),
                 ('task_0011_user_0007_scene_0009_cfg_0005', 'cam_104122061850'),
                 ('task_0011_user_0007_scene_0009_cfg_0005', 'cam_104122063678'),
                 ('task_0011_user_0007_scene_0009_cfg_0005', 'cam_104422070044'),
                 ('task_0011_user_0007_scene_0009_cfg_0005', 'cam_105422061350'),
                 ('task_0034_user_0007_scene_0010_cfg_0005', 'cam_037522062165'),
                 ('task_0034_user_0007_scene_0010_cfg_0005', 'cam_104122063678'),
                 ('task_0034_user_0007_scene_0010_cfg_0005', 'cam_104422070044'),
                 ('task_0034_user_0007_scene_0010_cfg_0005', 'cam_105422061350'),
                 ('task_0034_user_0010_scene_0005_cfg_0005', 'cam_036422060215'),
                 ('task_0034_user_0010_scene_0005_cfg_0005', 'cam_037522062165'),
                 ('task_0034_user_0010_scene_0005_cfg_0005', 'cam_104122061850'),
                 ('task_0034_user_0010_scene_0005_cfg_0005', 'cam_104122063678'),
                 ('task_0034_user_0010_scene_0005_cfg_0005', 'cam_104422070044'),
                 ('task_0034_user_0010_scene_0005_cfg_0005', 'cam_105422061350'),
                 ('task_0034_user_0010_scene_0005_cfg_0005', 'cam_f0461559'),
                 ('task_0035_user_0010_scene_0001_cfg_0005', 'cam_036422060215'),
                 ('task_0035_user_0010_scene_0001_cfg_0005', 'cam_037522062165'),
                 ('task_0035_user_0010_scene_0001_cfg_0005', 'cam_104122061850'),
                 ('task_0035_user_0010_scene_0001_cfg_0005', 'cam_104122063678'),
                 ('task_0035_user_0010_scene_0001_cfg_0005', 'cam_104422070044'),
                 ('task_0035_user_0010_scene_0001_cfg_0005', 'cam_105422061350'),
                 ('task_0035_user_0010_scene_0001_cfg_0005', 'cam_f0461559'),
                 ('task_0035_user_0010_scene_0002_cfg_0005', 'cam_036422060215'),
                 ('task_0035_user_0010_scene_0002_cfg_0005', 'cam_037522062165'),
                 ('task_0035_user_0010_scene_0002_cfg_0005', 'cam_104122061850'),
                 ('task_0035_user_0010_scene_0002_cfg_0005', 'cam_104122063678'),
                 ('task_0035_user_0010_scene_0002_cfg_0005', 'cam_104422070044'),
                 ('task_0035_user_0010_scene_0002_cfg_0005', 'cam_105422061350'),
                 ('task_0035_user_0010_scene_0002_cfg_0005', 'cam_f0461559'),
                 ('task_0035_user_0010_scene_0003_cfg_0005', 'cam_036422060215'),
                 ('task_0035_user_0010_scene_0003_cfg_0005', 'cam_037522062165'),
                 ('task_0035_user_0010_scene_0003_cfg_0005', 'cam_104122061850'),
                 ('task_0035_user_0010_scene_0003_cfg_0005', 'cam_104122063678'),
                 ('task_0035_user_0010_scene_0003_cfg_0005', 'cam_104422070044'),
                 ('task_0035_user_0010_scene_0003_cfg_0005', 'cam_105422061350'),
                 ('task_0035_user_0010_scene_0003_cfg_0005', 'cam_f0461559'),
                 ('task_0035_user_0010_scene_0004_cfg_0005', 'cam_036422060215'),
                 ('task_0035_user_0010_scene_0004_cfg_0005', 'cam_037522062165'),
                 ('task_0035_user_0010_scene_0004_cfg_0005', 'cam_104122061850'),
                 ('task_0035_user_0010_scene_0004_cfg_0005', 'cam_104122063678'),
                 ('task_0035_user_0010_scene_0004_cfg_0005', 'cam_104422070044'),
                 ('task_0035_user_0010_scene_0004_cfg_0005', 'cam_105422061350'),
                 ('task_0035_user_0010_scene_0004_cfg_0005', 'cam_f0461559'),
                 ('task_0035_user_0010_scene_0005_cfg_0005', 'cam_036422060215'),
                 ('task_0035_user_0010_scene_0005_cfg_0005', 'cam_037522062165'),
                 ('task_0035_user_0010_scene_0005_cfg_0005', 'cam_104122061850'),
                 ('task_0035_user_0010_scene_0005_cfg_0005', 'cam_104122063678'),
                 ('task_0035_user_0010_scene_0005_cfg_0005', 'cam_104422070044'),
                 ('task_0035_user_0010_scene_0005_cfg_0005', 'cam_105422061350'),
                 ('task_0035_user_0010_scene_0005_cfg_0005', 'cam_f0461559'),
                 ('task_0035_user_0010_scene_0007_cfg_0005', 'cam_036422060215'),
                 ('task_0035_user_0010_scene_0007_cfg_0005', 'cam_037522062165'),
                 ('task_0035_user_0010_scene_0007_cfg_0005', 'cam_104122061850'),
                 ('task_0035_user_0010_scene_0007_cfg_0005', 'cam_104122063678'),
                 ('task_0035_user_0010_scene_0007_cfg_0005', 'cam_104422070044'),
                 ('task_0035_user_0010_scene_0007_cfg_0005', 'cam_105422061350'),
                 ('task_0035_user_0010_scene_0007_cfg_0005', 'cam_f0461559'),
                 ('task_0035_user_0010_scene_0008_cfg_0005', 'cam_036422060215'),
                 ('task_0035_user_0010_scene_0008_cfg_0005', 'cam_037522062165'),
                 ('task_0035_user_0010_scene_0008_cfg_0005', 'cam_104122061850'),
                 ('task_0035_user_0010_scene_0008_cfg_0005', 'cam_104122063678'),
                 ('task_0035_user_0010_scene_0008_cfg_0005', 'cam_104422070044'),
                 ('task_0035_user_0010_scene_0008_cfg_0005', 'cam_105422061350'),
                 ('task_0035_user_0010_scene_0008_cfg_0005', 'cam_f0461559'),
                 ('task_0035_user_0010_scene_0009_cfg_0005', 'cam_036422060215'),
                 ('task_0035_user_0010_scene_0009_cfg_0005', 'cam_037522062165'),
                 ('task_0035_user_0010_scene_0009_cfg_0005', 'cam_104122061850'),
                 ('task_0035_user_0010_scene_0009_cfg_0005', 'cam_104122063678'),
                 ('task_0035_user_0010_scene_0009_cfg_0005', 'cam_104422070044'),
                 ('task_0035_user_0010_scene_0009_cfg_0005', 'cam_105422061350'),
                 ('task_0035_user_0010_scene_0009_cfg_0005', 'cam_f0461559'),
                 ('task_0035_user_0010_scene_0010_cfg_0005', 'cam_036422060215'),
                 ('task_0035_user_0010_scene_0010_cfg_0005', 'cam_037522062165'),
                 ('task_0035_user_0010_scene_0010_cfg_0005', 'cam_104122061850'),
                 ('task_0035_user_0010_scene_0010_cfg_0005', 'cam_104122063678'),
                 ('task_0035_user_0010_scene_0010_cfg_0005', 'cam_104422070044'),
                 ('task_0035_user_0010_scene_0010_cfg_0005', 'cam_105422061350'),
                 ('task_0035_user_0010_scene_0010_cfg_0005', 'cam_f0461559'),
                 ('task_0036_user_0010_scene_0001_cfg_0005', 'cam_036422060215'),
                 ('task_0036_user_0010_scene_0001_cfg_0005', 'cam_037522062165'),
                 ('task_0036_user_0010_scene_0001_cfg_0005', 'cam_104122061850'),
                 ('task_0036_user_0010_scene_0001_cfg_0005', 'cam_104122063678'),
                 ('task_0036_user_0010_scene_0001_cfg_0005', 'cam_104422070044'),
                 ('task_0036_user_0010_scene_0001_cfg_0005', 'cam_105422061350'),
                 ('task_0036_user_0010_scene_0001_cfg_0005', 'cam_f0461559'),
                 ('task_0036_user_0010_scene_0002_cfg_0005', 'cam_036422060215'),
                 ('task_0036_user_0010_scene_0002_cfg_0005', 'cam_037522062165'),
                 ('task_0036_user_0010_scene_0002_cfg_0005', 'cam_104122061850'),
                 ('task_0036_user_0010_scene_0002_cfg_0005', 'cam_104122063678'),
                 ('task_0036_user_0010_scene_0002_cfg_0005', 'cam_104422070044'),
                 ('task_0036_user_0010_scene_0002_cfg_0005', 'cam_105422061350'),
                 ('task_0036_user_0010_scene_0002_cfg_0005', 'cam_f0461559'),
                 ('task_0036_user_0010_scene_0003_cfg_0005', 'cam_036422060215'),
                 ('task_0036_user_0010_scene_0003_cfg_0005', 'cam_037522062165'),
                 ('task_0036_user_0010_scene_0003_cfg_0005', 'cam_104122061850'),
                 ('task_0036_user_0010_scene_0003_cfg_0005', 'cam_104122063678'),
                 ('task_0036_user_0010_scene_0003_cfg_0005', 'cam_104422070044'),
                 ('task_0036_user_0010_scene_0003_cfg_0005', 'cam_105422061350'),
                 ('task_0036_user_0010_scene_0003_cfg_0005', 'cam_f0461559'),
                 ('task_0036_user_0010_scene_0004_cfg_0005', 'cam_036422060215'),
                 ('task_0036_user_0010_scene_0004_cfg_0005', 'cam_037522062165'),
                 ('task_0036_user_0010_scene_0004_cfg_0005', 'cam_104122061850'),
                 ('task_0036_user_0010_scene_0004_cfg_0005', 'cam_104122063678'),
                 ('task_0036_user_0010_scene_0004_cfg_0005', 'cam_104422070044'),
                 ('task_0036_user_0010_scene_0004_cfg_0005', 'cam_105422061350'),
                 ('task_0036_user_0010_scene_0004_cfg_0005', 'cam_f0461559'),
                 ('task_0036_user_0010_scene_0006_cfg_0005', 'cam_036422060215'),
                 ('task_0036_user_0010_scene_0006_cfg_0005', 'cam_037522062165'),
                 ('task_0036_user_0010_scene_0006_cfg_0005', 'cam_104122061850'),
                 ('task_0036_user_0010_scene_0006_cfg_0005', 'cam_104122063678'),
                 ('task_0036_user_0010_scene_0006_cfg_0005', 'cam_104422070044'),
                 ('task_0036_user_0010_scene_0006_cfg_0005', 'cam_105422061350'),
                 ('task_0036_user_0010_scene_0006_cfg_0005', 'cam_f0461559'),
                 ('task_0036_user_0010_scene_0007_cfg_0005', 'cam_036422060215'),
                 ('task_0036_user_0010_scene_0007_cfg_0005', 'cam_037522062165'),
                 ('task_0036_user_0010_scene_0007_cfg_0005', 'cam_104122061850'),
                 ('task_0036_user_0010_scene_0007_cfg_0005', 'cam_104122063678'),
                 ('task_0036_user_0010_scene_0007_cfg_0005', 'cam_104422070044'),
                 ('task_0036_user_0010_scene_0007_cfg_0005', 'cam_105422061350'),
                 ('task_0036_user_0010_scene_0007_cfg_0005', 'cam_f0461559'),
                 ('task_0036_user_0010_scene_0008_cfg_0005', 'cam_036422060215'),
                 ('task_0036_user_0010_scene_0008_cfg_0005', 'cam_037522062165'),
                 ('task_0036_user_0010_scene_0008_cfg_0005', 'cam_104122061850'),
                 ('task_0036_user_0010_scene_0008_cfg_0005', 'cam_104122063678'),
                 ('task_0036_user_0010_scene_0008_cfg_0005', 'cam_104422070044'),
                 ('task_0036_user_0010_scene_0008_cfg_0005', 'cam_105422061350'),
                 ('task_0036_user_0010_scene_0009_cfg_0005', 'cam_036422060215'),
                 ('task_0036_user_0010_scene_0009_cfg_0005', 'cam_037522062165'),
                 ('task_0036_user_0010_scene_0009_cfg_0005', 'cam_104122061850'),
                 ('task_0036_user_0010_scene_0009_cfg_0005', 'cam_104122063678'),
                 ('task_0036_user_0010_scene_0009_cfg_0005', 'cam_104422070044'),
                 ('task_0036_user_0010_scene_0009_cfg_0005', 'cam_105422061350'),
                 ('task_0036_user_0010_scene_0010_cfg_0005', 'cam_036422060215'),
                 ('task_0036_user_0010_scene_0010_cfg_0005', 'cam_037522062165'),
                 ('task_0036_user_0010_scene_0010_cfg_0005', 'cam_104122061850'),
                 ('task_0036_user_0010_scene_0010_cfg_0005', 'cam_104122063678'),
                 ('task_0036_user_0010_scene_0010_cfg_0005', 'cam_104422070044'),
                 ('task_0036_user_0010_scene_0010_cfg_0005', 'cam_105422061350'),
                 ('task_0070_user_0007_scene_0007_cfg_0005', 'cam_036422060215'),
                 ('task_0070_user_0007_scene_0007_cfg_0005', 'cam_037522062165'),
                 ('task_0101_user_0007_scene_0001_cfg_0005', 'cam_036422060215'),
                 ('task_0101_user_0007_scene_0001_cfg_0005', 'cam_037522062165'),
                 ('task_0101_user_0007_scene_0001_cfg_0005', 'cam_104122063678'),
                 ('task_0101_user_0007_scene_0001_cfg_0005', 'cam_104422070044'),
                 ('task_0101_user_0007_scene_0001_cfg_0005', 'cam_f0461559'),
                 ('task_0105_user_0007_scene_0008_cfg_0005', 'cam_036422060215'),
                 ('task_0105_user_0007_scene_0008_cfg_0005', 'cam_037522062165'),
                 ('task_0105_user_0007_scene_0008_cfg_0005', 'cam_104122061850'),
                 ('task_0105_user_0007_scene_0008_cfg_0005', 'cam_104122063678'),
                 ('task_0105_user_0007_scene_0008_cfg_0005', 'cam_104422070044'),
                 ('task_0105_user_0007_scene_0008_cfg_0005', 'cam_105422061350'),
                 ('task_0105_user_0007_scene_0009_cfg_0005', 'cam_036422060215'),
                 ('task_0105_user_0007_scene_0009_cfg_0005', 'cam_037522062165'),
                 ('task_0105_user_0007_scene_0009_cfg_0005', 'cam_104122061850'),
                 ('task_0105_user_0007_scene_0009_cfg_0005', 'cam_104122063678'),
                 ('task_0105_user_0007_scene_0009_cfg_0005', 'cam_104422070044'),
                 ('task_0105_user_0007_scene_0009_cfg_0005', 'cam_105422061350'),
                 ('task_0105_user_0007_scene_0010_cfg_0005', 'cam_036422060215'),
                 ('task_0105_user_0007_scene_0010_cfg_0005', 'cam_037522062165'),
                 ('task_0105_user_0007_scene_0010_cfg_0005', 'cam_104122061850'),
                 ('task_0105_user_0007_scene_0010_cfg_0005', 'cam_104122063678'),
                 ('task_0105_user_0007_scene_0010_cfg_0005', 'cam_104422070044'),
                 ('task_0105_user_0007_scene_0010_cfg_0005', 'cam_105422061350'),
                 ('task_0106_user_0007_scene_0001_cfg_0005', 'cam_036422060215'),
                 ('task_0106_user_0007_scene_0001_cfg_0005', 'cam_037522062165'),
                 ('task_0106_user_0007_scene_0001_cfg_0005', 'cam_104122061850'),
                 ('task_0106_user_0007_scene_0001_cfg_0005', 'cam_104122063678'),
                 ('task_0106_user_0007_scene_0001_cfg_0005', 'cam_104422070044'),
                 ('task_0106_user_0007_scene_0001_cfg_0005', 'cam_105422061350'),
                 ('task_0106_user_0007_scene_0002_cfg_0005', 'cam_036422060215'),
                 ('task_0106_user_0007_scene_0002_cfg_0005', 'cam_037522062165'),
                 ('task_0106_user_0007_scene_0002_cfg_0005', 'cam_104122061850'),
                 ('task_0106_user_0007_scene_0002_cfg_0005', 'cam_104122063678'),
                 ('task_0106_user_0007_scene_0002_cfg_0005', 'cam_104422070044'),
                 ('task_0106_user_0007_scene_0002_cfg_0005', 'cam_105422061350'),
                 ('task_0106_user_0007_scene_0003_cfg_0005', 'cam_036422060215'),
                 ('task_0106_user_0007_scene_0003_cfg_0005', 'cam_037522062165'),
                 ('task_0106_user_0007_scene_0003_cfg_0005', 'cam_104122061850'),
                 ('task_0106_user_0007_scene_0003_cfg_0005', 'cam_104122063678'),
                 ('task_0106_user_0007_scene_0003_cfg_0005', 'cam_104422070044'),
                 ('task_0106_user_0007_scene_0003_cfg_0005', 'cam_105422061350'),
                 ('task_0106_user_0007_scene_0004_cfg_0005', 'cam_036422060215'),
                 ('task_0106_user_0007_scene_0004_cfg_0005', 'cam_037522062165'),
                 ('task_0106_user_0007_scene_0004_cfg_0005', 'cam_104122061850'),
                 ('task_0106_user_0007_scene_0004_cfg_0005', 'cam_104122063678'),
                 ('task_0106_user_0007_scene_0004_cfg_0005', 'cam_104422070044'),
                 ('task_0106_user_0007_scene_0004_cfg_0005', 'cam_105422061350'),
                 ('task_0106_user_0007_scene_0005_cfg_0005', 'cam_104422070044'),
                 ('task_0106_user_0007_scene_0008_cfg_0005', 'cam_036422060215'),
                 ('task_0106_user_0007_scene_0008_cfg_0005', 'cam_037522062165'),
                 ('task_0106_user_0007_scene_0008_cfg_0005', 'cam_104122061850'),
                 ('task_0106_user_0007_scene_0008_cfg_0005', 'cam_104122063678'),
                 ('task_0106_user_0007_scene_0008_cfg_0005', 'cam_104422070044'),
                 ('task_0106_user_0007_scene_0008_cfg_0005', 'cam_105422061350'),
                 ('task_0106_user_0007_scene_0009_cfg_0005', 'cam_036422060215'),
                 ('task_0106_user_0007_scene_0009_cfg_0005', 'cam_037522062165'),
                 ('task_0106_user_0007_scene_0009_cfg_0005', 'cam_104122061850'),
                 ('task_0106_user_0007_scene_0009_cfg_0005', 'cam_104122063678'),
                 ('task_0106_user_0007_scene_0009_cfg_0005', 'cam_104422070044'),
                 ('task_0106_user_0007_scene_0009_cfg_0005', 'cam_105422061350'),
                 ('task_0106_user_0007_scene_0010_cfg_0005', 'cam_036422060215'),
                 ('task_0106_user_0007_scene_0010_cfg_0005', 'cam_037522062165'),
                 ('task_0106_user_0007_scene_0010_cfg_0005', 'cam_104122061850'),
                 ('task_0106_user_0007_scene_0010_cfg_0005', 'cam_104122063678'),
                 ('task_0106_user_0007_scene_0010_cfg_0005', 'cam_104422070044'),
                 ('task_0106_user_0007_scene_0010_cfg_0005', 'cam_105422061350'),
                 ('task_0107_user_0007_scene_0001_cfg_0005', 'cam_036422060215'),
                 ('task_0107_user_0007_scene_0001_cfg_0005', 'cam_037522062165'),
                 ('task_0107_user_0007_scene_0001_cfg_0005', 'cam_104122061850'),
                 ('task_0107_user_0007_scene_0001_cfg_0005', 'cam_104122063678'),
                 ('task_0107_user_0007_scene_0001_cfg_0005', 'cam_104422070044'),
                 ('task_0107_user_0007_scene_0001_cfg_0005', 'cam_105422061350'),
                 ('task_0107_user_0007_scene_0002_cfg_0005', 'cam_036422060215'),
                 ('task_0107_user_0007_scene_0002_cfg_0005', 'cam_037522062165'),
                 ('task_0107_user_0007_scene_0002_cfg_0005', 'cam_104122061850'),
                 ('task_0107_user_0007_scene_0002_cfg_0005', 'cam_104122063678'),
                 ('task_0107_user_0007_scene_0002_cfg_0005', 'cam_104422070044'),
                 ('task_0107_user_0007_scene_0002_cfg_0005', 'cam_105422061350'),
                 ('task_0107_user_0007_scene_0003_cfg_0005', 'cam_036422060215'),
                 ('task_0107_user_0007_scene_0003_cfg_0005', 'cam_037522062165'),
                 ('task_0107_user_0007_scene_0003_cfg_0005', 'cam_104122061850'),
                 ('task_0107_user_0007_scene_0003_cfg_0005', 'cam_104122063678'),
                 ('task_0107_user_0007_scene_0003_cfg_0005', 'cam_104422070044'),
                 ('task_0107_user_0007_scene_0003_cfg_0005', 'cam_105422061350'),
                 ('task_0107_user_0007_scene_0004_cfg_0005', 'cam_036422060215'),
                 ('task_0107_user_0007_scene_0004_cfg_0005', 'cam_037522062165'),
                 ('task_0107_user_0007_scene_0004_cfg_0005', 'cam_104122061850'),
                 ('task_0107_user_0007_scene_0004_cfg_0005', 'cam_104122063678'),
                 ('task_0107_user_0007_scene_0004_cfg_0005', 'cam_104422070044'),
                 ('task_0107_user_0007_scene_0004_cfg_0005', 'cam_105422061350'),
                 ('task_0107_user_0007_scene_0005_cfg_0005', 'cam_036422060215'),
                 ('task_0107_user_0007_scene_0005_cfg_0005', 'cam_037522062165'),
                 ('task_0107_user_0007_scene_0005_cfg_0005', 'cam_104122061850'),
                 ('task_0107_user_0007_scene_0005_cfg_0005', 'cam_104122063678'),
                 ('task_0107_user_0007_scene_0005_cfg_0005', 'cam_104422070044'),
                 ('task_0107_user_0007_scene_0005_cfg_0005', 'cam_105422061350'),
                 ('task_0107_user_0007_scene_0006_cfg_0005', 'cam_036422060215'),
                 ('task_0107_user_0007_scene_0006_cfg_0005', 'cam_037522062165'),
                 ('task_0107_user_0007_scene_0006_cfg_0005', 'cam_104122061850'),
                 ('task_0107_user_0007_scene_0006_cfg_0005', 'cam_104122063678'),
                 ('task_0107_user_0007_scene_0006_cfg_0005', 'cam_104422070044'),
                 ('task_0107_user_0007_scene_0006_cfg_0005', 'cam_105422061350'),
                 ('task_0107_user_0007_scene_0007_cfg_0005', 'cam_036422060215'),
                 ('task_0107_user_0007_scene_0007_cfg_0005', 'cam_037522062165'),
                 ('task_0107_user_0007_scene_0007_cfg_0005', 'cam_104122061850'),
                 ('task_0107_user_0007_scene_0007_cfg_0005', 'cam_104122063678'),
                 ('task_0107_user_0007_scene_0007_cfg_0005', 'cam_104422070044'),
                 ('task_0107_user_0007_scene_0007_cfg_0005', 'cam_105422061350'),
                 ('task_0107_user_0007_scene_0008_cfg_0005', 'cam_036422060215'),
                 ('task_0107_user_0007_scene_0008_cfg_0005', 'cam_037522062165'),
                 ('task_0107_user_0007_scene_0008_cfg_0005', 'cam_104122061850'),
                 ('task_0107_user_0007_scene_0008_cfg_0005', 'cam_104122063678'),
                 ('task_0107_user_0007_scene_0008_cfg_0005', 'cam_104422070044'),
                 ('task_0107_user_0007_scene_0008_cfg_0005', 'cam_105422061350'),
                 ('task_0107_user_0007_scene_0009_cfg_0005', 'cam_036422060215'),
                 ('task_0107_user_0007_scene_0009_cfg_0005', 'cam_037522062165'),
                 ('task_0107_user_0007_scene_0009_cfg_0005', 'cam_104122061850'),
                 ('task_0107_user_0007_scene_0009_cfg_0005', 'cam_104122063678'),
                 ('task_0107_user_0007_scene_0009_cfg_0005', 'cam_104422070044'),
                 ('task_0107_user_0007_scene_0009_cfg_0005', 'cam_105422061350'),
                 ('task_0107_user_0007_scene_0010_cfg_0005', 'cam_036422060215'),
                 ('task_0107_user_0007_scene_0010_cfg_0005', 'cam_037522062165'),
                 ('task_0107_user_0007_scene_0010_cfg_0005', 'cam_104122061850'),
                 ('task_0107_user_0007_scene_0010_cfg_0005', 'cam_104122063678'),
                 ('task_0107_user_0007_scene_0010_cfg_0005', 'cam_104422070044'),
                 ('task_0107_user_0007_scene_0010_cfg_0005', 'cam_105422061350'),
                 ('task_0108_user_0007_scene_0001_cfg_0005', 'cam_036422060215'),
                 ('task_0108_user_0007_scene_0001_cfg_0005', 'cam_037522062165'),
                 ('task_0108_user_0007_scene_0001_cfg_0005', 'cam_104122061850'),
                 ('task_0108_user_0007_scene_0001_cfg_0005', 'cam_104122063678'),
                 ('task_0108_user_0007_scene_0001_cfg_0005', 'cam_104422070044'),
                 ('task_0108_user_0007_scene_0001_cfg_0005', 'cam_105422061350'),
                 ('task_0108_user_0007_scene_0002_cfg_0005', 'cam_036422060215'),
                 ('task_0108_user_0007_scene_0002_cfg_0005', 'cam_037522062165'),
                 ('task_0108_user_0007_scene_0002_cfg_0005', 'cam_104122061850'),
                 ('task_0108_user_0007_scene_0002_cfg_0005', 'cam_104122063678'),
                 ('task_0108_user_0007_scene_0002_cfg_0005', 'cam_105422061350'),
                 ('task_0108_user_0007_scene_0003_cfg_0005', 'cam_036422060215'),
                 ('task_0108_user_0007_scene_0003_cfg_0005', 'cam_037522062165'),
                 ('task_0108_user_0007_scene_0003_cfg_0005', 'cam_104122061850'),
                 ('task_0108_user_0007_scene_0003_cfg_0005', 'cam_104122063678'),
                 ('task_0108_user_0007_scene_0003_cfg_0005', 'cam_104422070044'),
                 ('task_0108_user_0007_scene_0003_cfg_0005', 'cam_105422061350'),
                 ('task_0108_user_0007_scene_0004_cfg_0005', 'cam_036422060215'),
                 ('task_0108_user_0007_scene_0004_cfg_0005', 'cam_037522062165'),
                 ('task_0108_user_0007_scene_0004_cfg_0005', 'cam_104122061850'),
                 ('task_0108_user_0007_scene_0004_cfg_0005', 'cam_104122063678'),
                 ('task_0108_user_0007_scene_0004_cfg_0005', 'cam_105422061350'),
                 ('task_0108_user_0007_scene_0005_cfg_0005', 'cam_036422060215'),
                 ('task_0108_user_0007_scene_0005_cfg_0005', 'cam_037522062165'),
                 ('task_0108_user_0007_scene_0005_cfg_0005', 'cam_104122061850'),
                 ('task_0108_user_0007_scene_0005_cfg_0005', 'cam_104122063678'),
                 ('task_0108_user_0007_scene_0005_cfg_0005', 'cam_104422070044'),
                 ('task_0108_user_0007_scene_0005_cfg_0005', 'cam_105422061350'),
                 ('task_0108_user_0007_scene_0006_cfg_0005', 'cam_036422060215'),
                 ('task_0108_user_0007_scene_0006_cfg_0005', 'cam_037522062165'),
                 ('task_0108_user_0007_scene_0006_cfg_0005', 'cam_104122061850'),
                 ('task_0108_user_0007_scene_0006_cfg_0005', 'cam_104122063678'),
                 ('task_0108_user_0007_scene_0006_cfg_0005', 'cam_104422070044'),
                 ('task_0108_user_0007_scene_0006_cfg_0005', 'cam_105422061350'),
                 ('task_0108_user_0007_scene_0007_cfg_0005', 'cam_036422060215'),
                 ('task_0108_user_0007_scene_0007_cfg_0005', 'cam_037522062165'),
                 ('task_0108_user_0007_scene_0007_cfg_0005', 'cam_104122061850'),
                 ('task_0108_user_0007_scene_0007_cfg_0005', 'cam_104122063678'),
                 ('task_0108_user_0007_scene_0007_cfg_0005', 'cam_105422061350'),
                 ('task_0108_user_0007_scene_0008_cfg_0005', 'cam_036422060215'),
                 ('task_0108_user_0007_scene_0008_cfg_0005', 'cam_037522062165'),
                 ('task_0108_user_0007_scene_0008_cfg_0005', 'cam_104122061850'),
                 ('task_0108_user_0007_scene_0008_cfg_0005', 'cam_104122063678'),
                 ('task_0108_user_0007_scene_0008_cfg_0005', 'cam_104422070044'),
                 ('task_0108_user_0007_scene_0008_cfg_0005', 'cam_105422061350'),
                 ('task_0108_user_0007_scene_0009_cfg_0005', 'cam_036422060215'),
                 ('task_0108_user_0007_scene_0009_cfg_0005', 'cam_037522062165'),
                 ('task_0108_user_0007_scene_0009_cfg_0005', 'cam_104122061850'),
                 ('task_0108_user_0007_scene_0009_cfg_0005', 'cam_104122063678'),
                 ('task_0108_user_0007_scene_0009_cfg_0005', 'cam_104422070044'),
                 ('task_0108_user_0007_scene_0009_cfg_0005', 'cam_105422061350'),
                 ('task_0108_user_0007_scene_0010_cfg_0005', 'cam_036422060215'),
                 ('task_0108_user_0007_scene_0010_cfg_0005', 'cam_037522062165'),
                 ('task_0108_user_0007_scene_0010_cfg_0005', 'cam_104122061850'),
                 ('task_0108_user_0007_scene_0010_cfg_0005', 'cam_104122063678'),
                 ('task_0108_user_0007_scene_0010_cfg_0005', 'cam_104422070044'),
                 ('task_0108_user_0007_scene_0010_cfg_0005', 'cam_105422061350'),
                 ('task_0109_user_0007_scene_0001_cfg_0005', 'cam_036422060215'),
                 ('task_0109_user_0007_scene_0001_cfg_0005', 'cam_037522062165'),
                 ('task_0109_user_0007_scene_0001_cfg_0005', 'cam_104122061850'),
                 ('task_0109_user_0007_scene_0001_cfg_0005', 'cam_104122063678'),
                 ('task_0109_user_0007_scene_0001_cfg_0005', 'cam_104422070044'),
                 ('task_0109_user_0007_scene_0001_cfg_0005', 'cam_105422061350'),
                 ('task_0109_user_0007_scene_0002_cfg_0005', 'cam_036422060215'),
                 ('task_0109_user_0007_scene_0002_cfg_0005', 'cam_037522062165'),
                 ('task_0109_user_0007_scene_0002_cfg_0005', 'cam_104122061850'),
                 ('task_0109_user_0007_scene_0002_cfg_0005', 'cam_104122063678'),
                 ('task_0109_user_0007_scene_0002_cfg_0005', 'cam_104422070044'),
                 ('task_0109_user_0007_scene_0002_cfg_0005', 'cam_105422061350'),
                 ('task_0109_user_0007_scene_0003_cfg_0005', 'cam_036422060215'),
                 ('task_0109_user_0007_scene_0003_cfg_0005', 'cam_037522062165'),
                 ('task_0109_user_0007_scene_0003_cfg_0005', 'cam_104122061850'),
                 ('task_0109_user_0007_scene_0003_cfg_0005', 'cam_104122063678'),
                 ('task_0109_user_0007_scene_0003_cfg_0005', 'cam_105422061350'),
                 ('task_0109_user_0007_scene_0004_cfg_0005', 'cam_036422060215'),
                 ('task_0109_user_0007_scene_0004_cfg_0005', 'cam_037522062165'),
                 ('task_0109_user_0007_scene_0004_cfg_0005', 'cam_104122061850'),
                 ('task_0109_user_0007_scene_0004_cfg_0005', 'cam_104122063678'),
                 ('task_0109_user_0007_scene_0004_cfg_0005', 'cam_104422070044'),
                 ('task_0109_user_0007_scene_0004_cfg_0005', 'cam_105422061350'),
                 ('task_0109_user_0007_scene_0005_cfg_0005', 'cam_036422060215'),
                 ('task_0109_user_0007_scene_0005_cfg_0005', 'cam_037522062165'),
                 ('task_0109_user_0007_scene_0005_cfg_0005', 'cam_104122061850'),
                 ('task_0109_user_0007_scene_0005_cfg_0005', 'cam_104122063678'),
                 ('task_0109_user_0007_scene_0005_cfg_0005', 'cam_104422070044'),
                 ('task_0109_user_0007_scene_0005_cfg_0005', 'cam_105422061350'),
                 ('task_0109_user_0007_scene_0006_cfg_0005', 'cam_036422060215'),
                 ('task_0109_user_0007_scene_0006_cfg_0005', 'cam_037522062165'),
                 ('task_0109_user_0007_scene_0006_cfg_0005', 'cam_104122061850'),
                 ('task_0109_user_0007_scene_0006_cfg_0005', 'cam_104122063678'),
                 ('task_0109_user_0007_scene_0006_cfg_0005', 'cam_105422061350'),
                 ('task_0109_user_0007_scene_0007_cfg_0005', 'cam_036422060215'),
                 ('task_0109_user_0007_scene_0007_cfg_0005', 'cam_037522062165'),
                 ('task_0109_user_0007_scene_0007_cfg_0005', 'cam_104122061850'),
                 ('task_0109_user_0007_scene_0007_cfg_0005', 'cam_104122063678'),
                 ('task_0109_user_0007_scene_0007_cfg_0005', 'cam_104422070044'),
                 ('task_0109_user_0007_scene_0007_cfg_0005', 'cam_105422061350'),
                 ('task_0109_user_0007_scene_0008_cfg_0005', 'cam_036422060215'),
                 ('task_0109_user_0007_scene_0008_cfg_0005', 'cam_037522062165'),
                 ('task_0109_user_0007_scene_0008_cfg_0005', 'cam_104122061850'),
                 ('task_0109_user_0007_scene_0008_cfg_0005', 'cam_104122063678'),
                 ('task_0109_user_0007_scene_0008_cfg_0005', 'cam_104422070044'),
                 ('task_0109_user_0007_scene_0008_cfg_0005', 'cam_105422061350'),
                 ('task_0109_user_0007_scene_0009_cfg_0005', 'cam_036422060215'),
                 ('task_0109_user_0007_scene_0009_cfg_0005', 'cam_037522062165'),
                 ('task_0109_user_0007_scene_0009_cfg_0005', 'cam_104122061850'),
                 ('task_0109_user_0007_scene_0009_cfg_0005', 'cam_104122063678'),
                 ('task_0109_user_0007_scene_0009_cfg_0005', 'cam_104422070044'),
                 ('task_0109_user_0007_scene_0009_cfg_0005', 'cam_105422061350'),
                 ('task_0109_user_0007_scene_0010_cfg_0005', 'cam_036422060215'),
                 ('task_0109_user_0007_scene_0010_cfg_0005', 'cam_037522062165'),
                 ('task_0109_user_0007_scene_0010_cfg_0005', 'cam_104122061850'),
                 ('task_0109_user_0007_scene_0010_cfg_0005', 'cam_104122063678'),
                 ('task_0109_user_0007_scene_0010_cfg_0005', 'cam_104422070044'),
                 ('task_0109_user_0007_scene_0010_cfg_0005', 'cam_105422061350'),
                 ('task_0110_user_0007_scene_0001_cfg_0005', 'cam_036422060215'),
                 ('task_0110_user_0007_scene_0001_cfg_0005', 'cam_037522062165'),
                 ('task_0110_user_0007_scene_0001_cfg_0005', 'cam_104122061850'),
                 ('task_0110_user_0007_scene_0001_cfg_0005', 'cam_104122063678'),
                 ('task_0110_user_0007_scene_0001_cfg_0005', 'cam_104422070044'),
                 ('task_0110_user_0007_scene_0001_cfg_0005', 'cam_105422061350'),
                 ('task_0110_user_0007_scene_0002_cfg_0005', 'cam_036422060215'),
                 ('task_0110_user_0007_scene_0002_cfg_0005', 'cam_037522062165'),
                 ('task_0110_user_0007_scene_0002_cfg_0005', 'cam_104122061850'),
                 ('task_0110_user_0007_scene_0002_cfg_0005', 'cam_104122063678'),
                 ('task_0110_user_0007_scene_0002_cfg_0005', 'cam_104422070044'),
                 ('task_0110_user_0007_scene_0002_cfg_0005', 'cam_105422061350'),
                 ('task_0110_user_0007_scene_0003_cfg_0005', 'cam_036422060215'),
                 ('task_0110_user_0007_scene_0003_cfg_0005', 'cam_037522062165'),
                 ('task_0110_user_0007_scene_0003_cfg_0005', 'cam_104122061850'),
                 ('task_0110_user_0007_scene_0003_cfg_0005', 'cam_104122063678'),
                 ('task_0110_user_0007_scene_0003_cfg_0005', 'cam_104422070044'),
                 ('task_0110_user_0007_scene_0003_cfg_0005', 'cam_105422061350'),
                 ('task_0110_user_0007_scene_0004_cfg_0005', 'cam_036422060215'),
                 ('task_0110_user_0007_scene_0004_cfg_0005', 'cam_037522062165'),
                 ('task_0110_user_0007_scene_0004_cfg_0005', 'cam_104122061850'),
                 ('task_0110_user_0007_scene_0004_cfg_0005', 'cam_104122063678'),
                 ('task_0110_user_0007_scene_0004_cfg_0005', 'cam_104422070044'),
                 ('task_0110_user_0007_scene_0004_cfg_0005', 'cam_105422061350'),
                 ('task_0110_user_0007_scene_0005_cfg_0005', 'cam_036422060215'),
                 ('task_0110_user_0007_scene_0005_cfg_0005', 'cam_037522062165'),
                 ('task_0110_user_0007_scene_0005_cfg_0005', 'cam_104122061850'),
                 ('task_0110_user_0007_scene_0005_cfg_0005', 'cam_104122063678'),
                 ('task_0110_user_0007_scene_0005_cfg_0005', 'cam_104422070044'),
                 ('task_0110_user_0007_scene_0005_cfg_0005', 'cam_105422061350'),
                 ('task_0110_user_0007_scene_0006_cfg_0005', 'cam_036422060215'),
                 ('task_0110_user_0007_scene_0006_cfg_0005', 'cam_037522062165'),
                 ('task_0110_user_0007_scene_0006_cfg_0005', 'cam_104122061850'),
                 ('task_0110_user_0007_scene_0006_cfg_0005', 'cam_104122063678'),
                 ('task_0110_user_0007_scene_0006_cfg_0005', 'cam_104422070044'),
                 ('task_0110_user_0007_scene_0006_cfg_0005', 'cam_105422061350'),
                 ('task_0110_user_0007_scene_0007_cfg_0005', 'cam_036422060215'),
                 ('task_0110_user_0007_scene_0007_cfg_0005', 'cam_037522062165'),
                 ('task_0110_user_0007_scene_0007_cfg_0005', 'cam_104122061850'),
                 ('task_0110_user_0007_scene_0007_cfg_0005', 'cam_104122063678'),
                 ('task_0110_user_0007_scene_0007_cfg_0005', 'cam_104422070044'),
                 ('task_0110_user_0007_scene_0007_cfg_0005', 'cam_105422061350'),
                 ('task_0110_user_0007_scene_0008_cfg_0005', 'cam_036422060215'),
                 ('task_0110_user_0007_scene_0008_cfg_0005', 'cam_037522062165'),
                 ('task_0110_user_0007_scene_0008_cfg_0005', 'cam_104122061850'),
                 ('task_0110_user_0007_scene_0008_cfg_0005', 'cam_104122063678'),
                 ('task_0110_user_0007_scene_0008_cfg_0005', 'cam_104422070044'),
                 ('task_0110_user_0007_scene_0008_cfg_0005', 'cam_105422061350'),
                 ('task_0110_user_0007_scene_0009_cfg_0005', 'cam_036422060215'),
                 ('task_0110_user_0007_scene_0009_cfg_0005', 'cam_037522062165'),
                 ('task_0110_user_0007_scene_0009_cfg_0005', 'cam_104122061850'),
                 ('task_0110_user_0007_scene_0009_cfg_0005', 'cam_104122063678'),
                 ('task_0110_user_0007_scene_0009_cfg_0005', 'cam_104422070044'),
                 ('task_0110_user_0007_scene_0009_cfg_0005', 'cam_105422061350'),
                 ('task_0110_user_0007_scene_0010_cfg_0005', 'cam_036422060215'),
                 ('task_0110_user_0007_scene_0010_cfg_0005', 'cam_037522062165'),
                 ('task_0110_user_0007_scene_0010_cfg_0005', 'cam_104122061850'),
                 ('task_0110_user_0007_scene_0010_cfg_0005', 'cam_104122063678'),
                 ('task_0110_user_0007_scene_0010_cfg_0005', 'cam_104422070044'),
                 ('task_0110_user_0007_scene_0010_cfg_0005', 'cam_105422061350'),
                 ('task_0111_user_0007_scene_0001_cfg_0005', 'cam_036422060215'),
                 ('task_0111_user_0007_scene_0001_cfg_0005', 'cam_037522062165'),
                 ('task_0111_user_0007_scene_0001_cfg_0005', 'cam_104122061850'),
                 ('task_0111_user_0007_scene_0001_cfg_0005', 'cam_104122063678'),
                 ('task_0111_user_0007_scene_0001_cfg_0005', 'cam_104422070044'),
                 ('task_0111_user_0007_scene_0001_cfg_0005', 'cam_105422061350'),
                 ('task_0111_user_0007_scene_0002_cfg_0005', 'cam_036422060215'),
                 ('task_0111_user_0007_scene_0002_cfg_0005', 'cam_037522062165'),
                 ('task_0111_user_0007_scene_0002_cfg_0005', 'cam_104122061850'),
                 ('task_0111_user_0007_scene_0002_cfg_0005', 'cam_104122063678'),
                 ('task_0111_user_0007_scene_0002_cfg_0005', 'cam_105422061350'),
                 ('task_0111_user_0007_scene_0003_cfg_0005', 'cam_036422060215'),
                 ('task_0111_user_0007_scene_0003_cfg_0005', 'cam_037522062165'),
                 ('task_0111_user_0007_scene_0003_cfg_0005', 'cam_104122061850'),
                 ('task_0111_user_0007_scene_0003_cfg_0005', 'cam_104122063678'),
                 ('task_0111_user_0007_scene_0003_cfg_0005', 'cam_104422070044'),
                 ('task_0111_user_0007_scene_0003_cfg_0005', 'cam_105422061350'),
                 ('task_0111_user_0007_scene_0004_cfg_0005', 'cam_036422060215'),
                 ('task_0111_user_0007_scene_0004_cfg_0005', 'cam_037522062165'),
                 ('task_0111_user_0007_scene_0004_cfg_0005', 'cam_104122061850'),
                 ('task_0111_user_0007_scene_0004_cfg_0005', 'cam_104122063678'),
                 ('task_0111_user_0007_scene_0004_cfg_0005', 'cam_104422070044'),
                 ('task_0111_user_0007_scene_0004_cfg_0005', 'cam_105422061350'),
                 ('task_0111_user_0007_scene_0005_cfg_0005', 'cam_036422060215'),
                 ('task_0111_user_0007_scene_0005_cfg_0005', 'cam_037522062165'),
                 ('task_0111_user_0007_scene_0005_cfg_0005', 'cam_104122061850'),
                 ('task_0111_user_0007_scene_0005_cfg_0005', 'cam_104122063678'),
                 ('task_0111_user_0007_scene_0005_cfg_0005', 'cam_105422061350'),
                 ('task_0111_user_0007_scene_0006_cfg_0005', 'cam_036422060215'),
                 ('task_0111_user_0007_scene_0006_cfg_0005', 'cam_037522062165'),
                 ('task_0111_user_0007_scene_0006_cfg_0005', 'cam_104122061850'),
                 ('task_0111_user_0007_scene_0006_cfg_0005', 'cam_104122063678'),
                 ('task_0111_user_0007_scene_0006_cfg_0005', 'cam_104422070044'),
                 ('task_0111_user_0007_scene_0006_cfg_0005', 'cam_105422061350'),
                 ('task_0111_user_0007_scene_0007_cfg_0005', 'cam_036422060215'),
                 ('task_0111_user_0007_scene_0007_cfg_0005', 'cam_037522062165'),
                 ('task_0111_user_0007_scene_0007_cfg_0005', 'cam_104122061850'),
                 ('task_0111_user_0007_scene_0007_cfg_0005', 'cam_104122063678'),
                 ('task_0111_user_0007_scene_0007_cfg_0005', 'cam_104422070044'),
                 ('task_0111_user_0007_scene_0007_cfg_0005', 'cam_105422061350'),
                 ('task_0111_user_0007_scene_0008_cfg_0005', 'cam_036422060215'),
                 ('task_0111_user_0007_scene_0008_cfg_0005', 'cam_037522062165'),
                 ('task_0111_user_0007_scene_0008_cfg_0005', 'cam_104122061850'),
                 ('task_0111_user_0007_scene_0008_cfg_0005', 'cam_104122063678'),
                 ('task_0111_user_0007_scene_0008_cfg_0005', 'cam_104422070044'),
                 ('task_0111_user_0007_scene_0008_cfg_0005', 'cam_105422061350'),
                 ('task_0021_user_0014_scene_0009_cfg_0006', 'cam_104122064161'),
                 ('task_0077_user_0014_scene_0010_cfg_0006', 'cam_104122061018'),
                 ('task_0105_user_0014_scene_0006_cfg_0006', 'cam_104122064161'),
                 ('task_0205_user_0007_scene_0001_cfg_0006', 'cam_104122060811')]
IGNORED_TASKS = ['task_0040', 'task_0093', 'task_0095', 'task_0112', 'task_0116', 'task_0129', 'task_0130', 'task_0131', 'task_0132', 'task_0213', 'task_0214', 'task_0215', 'task_0216', 'task_0217', 'task_0218', 'task_0220', 'task_0221', 'task_0222', 'task_0223', 'task_0225', 'task_0226', 'task_0329']
PICK_PLACE_TASKS = ['task_0008', 'task_0009', 'task_0010', 'task_0011', 'task_0012', 'task_0013', 'task_0014', 'task_0017', 'task_0028', 'task_0029', 'task_0031', 'task_0035', 'task_0037', 'task_0038', 'task_0040', 'task_0044', 'task_0045', 'task_0046', 'task_0047', 'task_0050', 'task_0051', 'task_0052', 'task_0054', 'task_0056', 'task_0061', 'task_0062', 'task_0064', 'task_0072', 'task_0073', 'task_0076', 'task_0077', 'task_0088', 'task_0093', 'task_0095', 'task_0096', 'task_0105', 'task_0106', 'task_0107', 'task_0122', 'task_0123', 'task_0222']
# PICK_PLACE_TASKS = ['task_0013']
FINETUNE_TASKS = ['task_0012', 'task_0013', 'task_0014']
# IGNORED_TASKS = []
# ABANDONED_TASKS = ['task_0220', 'task_0221', 'task_0223']


class RH100TDataset(Dataset):
    def __init__(self, root, task_config_list, split='train', task_descriptions=None, num_input=1, horizon=1+20, timestep=-1, filter_frames=False, filter_thresh=None, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], voxel_size=0.005, from_cache=False, cache_path='dataset/valid_data_dict.json', intrinsics_path='dataset/camera_intrinsics.npy', augmentation=False, frame_sample_step=1, centralize_gripper=False, top_down_view=False, rot_6d=False):
        assert split in ['train', 'val', 'all']

        self.root = root
        self.split = split
        self.task_descriptions = task_descriptions
        self.num_input = num_input
        self.horizon = horizon
        self.filter_frames = filter_frames
        self.filter_thresh = filter_thresh if filter_thresh is not None else\
                            {'translation': 0.005, 'rotation': np.pi/12, 'gripper_width': 0.005}
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.voxel_size = voxel_size
        self.augmentation = augmentation
        self.centralize_gripper = centralize_gripper
        self.top_down_view = top_down_view
        self.rot_6d = rot_6d
        
        self.input_task_ids = []
        self.input_cam_ids = []
        self.input_task_configs = []
        self.target_frame_ids = []
        self.padding_mask_list = []

        self.camera_intrinsics = np.load(intrinsics_path, allow_pickle=True)[()]

        if not from_cache:
            self.task_ids, self.cam_ids, self.task_configs = load_all_tasks(root, task_config_list, split, top_down_view=top_down_view)
            num_tasks = len(self.task_ids)
            print('#tasks:', num_tasks)

            valid_data_dict = {cfg:{} for cfg in self.task_configs}
            for i in tqdm(range(num_tasks), desc='loading data samples...'):
                task_id, cam_id, task_config = self.task_ids[i], self.cam_ids[i], self.task_configs[i]
                if (task_id, cam_id) in PROBLEM_TASKS: continue
                # print(task_id, cam_id)
                meta_path = os.path.join(self.root, task_config, task_id, 'metadata.json')
                metadata = json.load(open(meta_path))
                try:
                    frame_ids = self._get_frames(task_id, cam_id, task_config, metadata['finish_time'])
                    if task_id not in valid_data_dict[task_config]:
                        valid_data_dict[task_config][task_id] = {cam_id: frame_ids}
                    else:
                        valid_data_dict[task_config][task_id][cam_id] = frame_ids
                except:
                    print("('%s', '%s')" % (task_id, cam_id))
                    continue
                target_frame_ids, padding_mask_list = self._get_input_output_frame_id_lists(frame_ids, num_input=num_input, horizon=horizon, timestep=timestep, frame_sample_step=frame_sample_step)
                self.target_frame_ids += target_frame_ids
                self.padding_mask_list += padding_mask_list
                self.input_task_ids += [task_id] * len(target_frame_ids)
                self.input_cam_ids += [cam_id] * len(target_frame_ids)
                self.input_task_configs += [task_config] * len(target_frame_ids)
            if cache_path is not None:
                with open(cache_path, 'w') as f:
                    json.dump(valid_data_dict, f, indent=4)
        
        else:
            with open(cache_path, 'r') as f:
                valid_data_dict = json.load(f)
            self.task_ids, self.cam_ids, self.task_configs = load_all_tasks(root, task_config_list, split, cache_dict=valid_data_dict, top_down_view=top_down_view)
            num_tasks = len(self.task_ids)
            print('#tasks:', num_tasks)

            for i in tqdm(range(num_tasks), desc='loading data samples...'):
                task_id, cam_id, task_config = self.task_ids[i], self.cam_ids[i], self.task_configs[i]
                if (task_id, cam_id) in PROBLEM_TASKS: continue
                # print(task_id, cam_id)
                meta_path = os.path.join(self.root, task_config, task_id, 'metadata.json')
                metadata = json.load(open(meta_path))
                frame_ids = valid_data_dict[task_config][task_id][cam_id]
                frame_ids = [x for x in frame_ids if x <= metadata['finish_time']]
                # frame_ids = frame_ids[:(len(frame_ids)+2)//3] # TODO: test first 33% data
                target_frame_ids, padding_mask_list = self._get_input_output_frame_id_lists(frame_ids, num_input=num_input, horizon=horizon, timestep=timestep, frame_sample_step=frame_sample_step)
                self.target_frame_ids += target_frame_ids
                self.padding_mask_list += padding_mask_list
                self.input_task_ids += [task_id] * len(target_frame_ids)
                self.input_cam_ids += [cam_id] * len(target_frame_ids)
                self.input_task_configs += [task_config] * len(target_frame_ids)

        # prepare language embeddings
        print('Loading CLIP model...')
        self.task_descriptions = self._load_language_embeddings(self.task_descriptions)

    def _load_language_embeddings(self, task_descriptions):
        clip_model, preprocess = clip.load('RN50', device='cpu') # CLIP-ResNet50
        for task_name in tqdm(task_descriptions, desc="Embedding language conditions..."):
            tokens = clip.tokenize([task_descriptions[task_name]['task_description_english']])
            # encode text
            embed = clip_model.token_embedding(tokens).type(clip_model.dtype)
            embed = embed + clip_model.positional_embedding.type(clip_model.dtype)
            embed = embed.permute(1, 0, 2)  # NLD -> LND
            embed = clip_model.transformer(embed)
            embed = embed.permute(1, 0, 2)  # LND -> NLD
            embed = clip_model.ln_final(embed).type(clip_model.dtype)

            token_mask = torch.full([tokens.size(-1)], True)
            token_mask[:tokens.argmax(dim=-1)+1] = False
            task_descriptions[task_name]['embedding'] = embed[0].detach().numpy()
            task_descriptions[task_name]['token_mask'] = token_mask.numpy()

        del clip_model
        return task_descriptions

    def __len__(self):
        return len(self.target_frame_ids)

    def _get_frames(self, task_id, cam_id, task_config, finish_time=None):
        color_dir = os.path.join(self.root, task_config, task_id, cam_id, 'color')
        frame_ids = sorted(os.listdir(color_dir))
        frame_ids = [int(x.split('.')[0]) for x in frame_ids]

        if self.filter_frames:
            tcps, gripper_widths = self._get_scene_gripper_poses(task_id, cam_id, task_config)
            kept_frame_indices = self._filter_scene_frames(tcps, gripper_widths)
            frame_ids = [frame_ids[x] for x in kept_frame_indices]

        if finish_time is not None:
            frame_ids = [x for x in frame_ids if x <= finish_time]
        
        return frame_ids

    def _get_scene_gripper_poses(self, task_id, cam_id, task_config):
        # load tcps
        tcp_path = os.path.join(self.root, task_config, task_id, 'transformed', 'tcp.npy')
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
                    cur_target_frame_ids += [frame_id_list[-1]] * (horizon - len(cur_target_frame_ids))
                    padding_mask[len(cur_target_frame_ids):] = 1
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

    def _randomly_rotate_points(self, cloud_list, tcp_array):
        angle_x = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
        angle_y = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
        angle_z = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
        c, s = np.cos(angle_x), np.sin(angle_x)
        rot_mat_x = np.array([[1, 0, 0],
                              [0, c,-s],
                              [0, s, c]])
        c, s = np.cos(angle_y), np.sin(angle_y)
        rot_mat_y = np.array([[c, 0, s],
                              [0, 1, 0],
                              [-s,0, c]])
        c, s = np.cos(angle_z), np.sin(angle_z)
        rot_mat_z = np.array([[c,-s, 0],
                              [s, c, 0],
                              [0, 0, 1]])
        random_rot_mat = np.matmul(rot_mat_z, np.matmul(rot_mat_y, rot_mat_x)).astype(np.float32)

        trans_array = tcp_array[:,0:3]
        quat_array = tcp_array[:,3:7]
        
        # quat to rot
        rot_mat = []
        for i in range(quat_array.shape[0]):
            rot_mat.append(quat2mat(quat_array[i]))
        rot_mat = np.array(rot_mat)

        # transformation
        for i, cloud in enumerate(cloud_list):
            points = cloud[:,:3]
            points = np.matmul(random_rot_mat, points.T).T
            cloud[:,:3] = points
            cloud_list[i] = cloud
        trans_array = np.matmul(random_rot_mat, trans_array.T).T
        rot_mat = np.matmul(random_rot_mat[np.newaxis], rot_mat)
        tcp_array[:,0:3] = trans_array
        for i in range(tcp_array.shape[0]):
            tcp_array[i,3:7] = mat2quat(rot_mat[i])

        return cloud_list, tcp_array

    def _randomly_translate_points(self, cloud_list, tcp_array):
        offset_x = np.random.random() * 0.4 - 0.2 # -0.2 ~ 0.2
        offset_y = np.random.random() * 0.4 - 0.2 # -0.2 ~ 0.2
        offset_z = np.random.random() * 0.4 - 0.2 # -0.2 ~ 0.2
        
        tcp_array[:,0:3] += [offset_x, offset_y, offset_z]
        for i, cloud in enumerate(cloud_list):
            points = cloud[:,:3]
            points += [offset_x, offset_y, offset_z]
            cloud[:,:3] = points
            cloud_list[i] = cloud
        return cloud_list, tcp_array

    def _augment_points(self, cloud_list, tcp_array):
        # rotation around xyz
        cloud_list, tcp_array = self._randomly_rotate_points(cloud_list, tcp_array)
        # translation along xyz
        if not self.centralize_gripper:
            cloud_list, tcp_array = self._randomly_translate_points(cloud_list, tcp_array)
        return cloud_list, tcp_array
    
    def _centralize_points_and_tcp_by_gripper(self, center, cloud_list, tcp_array):
        for i, cloud in enumerate(cloud_list):
            points = cloud[:,:3]
            points -= center
            cloud[:,:3] = points
            cloud_list[i] = cloud
        tcp_array[:,:3] -= center
        return cloud_list, tcp_array
    
    def _get_action_label(self, trans, rpy, gripper_width, max_gripper_width=0.11):
        trans_min, trans_max = np.array([-0.64, -0.64, 0]), np.array([0.64, 0.64, 1.28]) # TODO: need adjustification
        rpy_min, rpy_max = -np.pi, np.pi

        trans_label = (trans - trans_min) / (trans_max - trans_min)
        trans_label = np.clip(trans_label, 0, 1)
        trans_label = (trans_label * 255).astype(np.int32)

        rpy_label = (rpy - rpy_min) / (rpy_max - rpy_min)
        rpy_label = (rpy_label * 255).astype(np.int32)
        rpy_label = np.clip(rpy_label, 0, 255)

        gripper_label = gripper_width / max_gripper_width
        gripper_label = (gripper_label * 255).astype(np.int32)
        gripper_label = np.clip(gripper_label, 0, 255)

        action_label = np.ones(7, dtype=np.int64)
        action_label[0:3] = trans_label
        action_label[3:6] = rpy_label
        action_label[6] = gripper_label

        return action_label

    def _clip_tcp(self, tcp_list):
        ''' tcp_list: [T, 8]'''
        if self.centralize_gripper:
            tcp_list[:,0] = np.clip(tcp_list[:,0], -0.3, 0.3)
            tcp_list[:,1] = np.clip(tcp_list[:,1], -0.3, 0.3)
            tcp_list[:,2] = np.clip(tcp_list[:,2], -0.3, 0.3)
        else:
            tcp_list[:,0] = np.clip(tcp_list[:,0], -0.64, 0.64)
            tcp_list[:,1] = np.clip(tcp_list[:,1], -0.64, 0.64)
            tcp_list[:,2] = np.clip(tcp_list[:,2], 0, 1.28)
        tcp_list[:,7] = np.clip(tcp_list[:,7], 0, 0.11)
        return tcp_list

    def _normalize_tcp(self, tcp_list):
        ''' tcp_list: [T, 8]'''
        if self.centralize_gripper:
            trans_min, trans_max = np.array([-0.15, -0.15, -0.15]), np.array([0.15, 0.15, 0.15])
        elif self.top_down_view:
            trans_min, trans_max = np.array([-0.35, -0.35, 0]), np.array([0.35, 0.35, 0.7])
        else:
            trans_min, trans_max = np.array([-0.64, -0.64, 0]), np.array([0.64, 0.64, 1.28])
        max_gripper_width = 0.11 # meter
        tcp_list[:,:3] = (tcp_list[:,:3] - trans_min) / (trans_max - trans_min) * 2 - 1
        tcp_list[:,7] = tcp_list[:,7] / max_gripper_width * 2 - 1
        return tcp_list

    def __getitem__(self, index):
        task_id = self.input_task_ids[index]
        # input_frame_ids = self.input_frame_id_lists[index]
        target_frame_ids = self.target_frame_ids[index]
        padding_mask = self.padding_mask_list[index]
        cam_id = self.input_cam_ids[index]
        task_config = self.input_task_configs[index]
        # tcp_list = self.tcp_dicts[task_id][cam_id[4:]]
        # gripper_list = self.gripper_dicts[task_id][cam_id[4:]]

        # load input rgbs
        input_cloud_list = []
        for input_frame_id in target_frame_ids[:self.num_input]:
            color_path = os.path.join(self.root, task_config, task_id, cam_id, 'color', '%d.jpg'%input_frame_id)
            depth_path = os.path.join(self.root, task_config, task_id, cam_id, 'depth_copy', '%d.png'%input_frame_id)
            colors = np.array(Image.open(color_path), dtype=np.float32) / 255.0
            depths = np.array(Image.open(depth_path))
            # imagenet normalization
            colors = (colors - self.image_mean) / self.image_std
            # create point cloud
            depth_scale = 1000. if 'f' not in cam_id else 4000.
            points = create_point_cloud_from_depth_image(depths, self.camera_intrinsics[cam_id[4:]], depth_scale)
            x_mask = ((points[:,:,0] > -1.5) & (points[:,:,0] < 1.5))
            y_mask = ((points[:,:,1] > -1.5) & (points[:,:,1] < 1.5))
            z_mask = ((points[:,:,2] > 0.01) & (points[:,:,2] < 2))
            point_mask = (x_mask & y_mask & z_mask)
            points = points[point_mask]
            colors = colors[point_mask]
            cloud = np.concatenate([points, colors], axis=-1)
            input_cloud_list.append(cloud.astype(np.float32))

        gripper_path = os.path.join(self.root, task_config, task_id, 'transformed', 'gripper.npy')
        tcp_path = os.path.join(self.root, task_config, task_id, 'transformed', 'tcp.npy')

        # load input and target gripper pose
        tcp_list = np.load(tcp_path, allow_pickle=True)[()][cam_id[4:]]
        target_frame_tcp_list = []
        i, p = 0, 0
        while i < len(tcp_list):
            while p < self.horizon and tcp_list[i]['timestamp'] == target_frame_ids[p]:
                target_frame_tcp_list.append(tcp_list[i]['tcp'].astype(np.float32))
                p += 1
            if p == self.horizon:
                break
            i += 1
        assert p == self.horizon, 'p:%d, input:%d' % (p, self.horizon)

        target_frame_tcp_list = np.array(target_frame_tcp_list, dtype=np.float32)

        # get gripper label
        gripper_list = np.load(gripper_path, allow_pickle=True)[()][cam_id[4:]]
        target_gripper_width_list = []
        for i,fid in enumerate(target_frame_ids):
            if i < self.num_input:
                gripper_command = gripper_list[fid]['gripper_info']
            else:
                gripper_command = gripper_list[fid]['gripper_command']
            gripper_width = gripper_command[0] / 1000. # transform mm into m
            target_gripper_width_list.append(gripper_width)

        if self.centralize_gripper:
            center = target_frame_tcp_list[self.num_input-1][:3].copy()
            # remove tcp outlier
            center[0] = max(-0.64, min(0.64, center[0]))
            center[1] = max(-0.64, min(0.64, center[1]))
            center[2] = max(0, min(1.28, center[2]))
            input_cloud_list, target_frame_tcp_list = self._centralize_points_and_tcp_by_gripper(center, input_cloud_list, target_frame_tcp_list)
        if self.augmentation:
            input_cloud_list, target_frame_tcp_list = self._augment_points(input_cloud_list, target_frame_tcp_list)

        target_gripper_width_list = np.array(target_gripper_width_list, dtype=np.float32)[:,np.newaxis]
        target_frame_tcp_list = np.concatenate([target_frame_tcp_list, target_gripper_width_list], axis=-1)


        # get normalized tcp
        target_frame_tcp_list = np.array(target_frame_tcp_list, dtype=np.float32)
        target_frame_tcp_list = self._clip_tcp(target_frame_tcp_list)
        target_frame_tcp_normalized = self._normalize_tcp(target_frame_tcp_list.copy())
        # transform quaternion to 6d rotation
        if self.rot_6d:
            target_frame_rotation_6d = batch_quaternion_to_rotation6d(target_frame_tcp_normalized[:,3:7])
            target_frame_tcp_normalized = np.concatenate([target_frame_tcp_normalized[:,0:3], target_frame_rotation_6d, target_frame_tcp_normalized[:,-1:]], axis=-1)
        # # remove rotation
        # target_frame_tcp_normalized = np.concatenate([target_frame_tcp_normalized[:,0:3],target_frame_tcp_normalized[:,-1:]], axis=-1)

        # make voxel input
        input_coords_list = []
        input_feats_list = []
        for cloud in input_cloud_list:
            # Upd Note. Make coords contiguous.
            coords = np.ascontiguousarray(cloud[:,:3] / self.voxel_size, dtype=np.int32)
            # Upd Note. API change.
            _, idxs = ME.utils.sparse_quantize(coords, return_index=True)
            input_coords_list.append(coords[idxs])
            input_feats_list.append(cloud[idxs].astype(np.float32))

        # get instruction
        if self.task_descriptions is not None:
            instruction = self.task_descriptions[task_id[:9]]['task_description_english']
            language_embed = self.task_descriptions[task_id[:9]]['embedding']
            language_mask = self.task_descriptions[task_id[:9]]['token_mask']
        else:
            instruction = 'hello!'

        # split data
        input_frame_tcp_list = target_frame_tcp_list[:self.num_input]
        target_frame_tcp_list = target_frame_tcp_list[self.num_input:]
        input_frame_tcp_normalized = target_frame_tcp_normalized[:self.num_input]
        target_frame_tcp_normalized = target_frame_tcp_normalized[self.num_input:]
        padding_mask = padding_mask[self.num_input:]

        # convert to torch
        # input_cloud_list = [torch.from_numpy(x) for x in input_cloud_list]
        input_frame_tcp_normalized = torch.from_numpy(input_frame_tcp_normalized)
        target_frame_tcp_normalized = torch.from_numpy(target_frame_tcp_normalized)
        padding_mask = torch.from_numpy(padding_mask)
        language_embed = torch.from_numpy(language_embed)
        language_mask = torch.from_numpy(language_mask)

        if self.num_input == 1:
            # input_cloud_list = input_cloud_list[0]
            input_frame_tcp_list = input_frame_tcp_list[0]
            input_frame_tcp_normalized = input_frame_tcp_normalized[0]

        ret_dict = {#'input_cloud_list': input_cloud_list,
                    'instruction': instruction,
                    'language_embed': language_embed,
                    'language_mask': language_mask,
                    'input_coords_list': input_coords_list,
                    'input_feats_list': input_feats_list,
                    'input_frame_tcp_list': input_frame_tcp_list,
                    'input_frame_tcp_normalized': input_frame_tcp_normalized,
                    'target_frame_tcp_list': target_frame_tcp_list,
                    'target_frame_tcp_normalized': target_frame_tcp_normalized,
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
        # ret_dict = {key:collate_fn([d[key] for d in batch]) for key in batch[0] if key != 'instruction'}
        # ret_dict['instruction'] = [d['instruction'] for d in batch]
        coords_batch = ret_dict['input_coords_list']
        feats_batch = ret_dict['input_feats_list']
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
        ret_dict['input_coords_list'] = coords_batch
        ret_dict['input_feats_list'] = feats_batch
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

def load_all_tasks(task_root, task_configs, split='train', cache_dict=None, inference_mode=False, top_down_view=False):
    assert split in ['train', 'val', 'all']
    task_ids = []
    cam_ids = []
    config_ids = []
    in_hand_cam_ids = {'RH100T_cfg1': ['cam_043322070878'],
                       'RH100T_cfg2': ['cam_104422070042'],
                       'RH100T_cfg3': ['cam_045322071843'],
                       'RH100T_cfg4': ['cam_045322071843'],
                       'RH100T_cfg5': ['cam_104422070042', 'cam_135122079702'],
                       'RH100T_cfg6': ['cam_135122070361', 'cam_135122075425'],
                       'RH100T_cfg7': ['cam_135122070361', 'cam_135122075425']}
    top_down_cam_ids = {'RH100T_cfg1': ['cam_750612070851', 'cam_039422060546', 'cam_750612070853'],
                        'RH100T_cfg2': ['cam_f0461559', 'cam_037522062165', 'cam_104122061850'],
                        'RH100T_cfg3': ['cam_038522062288', 'cam_104122062295'],
                        'RH100T_cfg4': ['cam_038522062288', 'cam_104122062295'],
                        'RH100T_cfg5': ['cam_037522062165'],
                        'RH100T_cfg6': ['cam_104122061330'],
                        'RH100T_cfg7': ['cam_104122061330']}
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

    def _get_scene_meta(scene_dir):
        meta_path = os.path.join(scene_dir, 'metadata.json')
        metadata = json.load(open(meta_path))
        return metadata

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
                scene_dir = os.path.join(task_root, task_config, task_id)
                metadata = _get_scene_meta(scene_dir)
                if 'rating' not in metadata or metadata['rating'] <= 1:
                    continue
                cur_cam_ids = sorted(cache_dict[task_config][task_id])
                for cam_id in cur_cam_ids:
                    if top_down_view and cam_id not in top_down_cam_ids[task_config]:
                        continue
                    ## comment the following filtering because this has been done during generating cache_dict
                    # if cam_id in in_hand_cam_ids[task_config]:
                    #     print(scene_dir, cam_id)
                    #     continue
                    # if 'bad_calib_view' in metadata and cam_id[4:] in metadata['bad_calib_view']:
                    #     print(scene_dir, cam_id)
                    #     continue
                    # if not _validate_scene(scene_dir, cam_id):
                    #     print(scene_dir, cam_id)
                    #     continue
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
            metadata = _get_scene_meta(scene_dir)
            if 'rating' not in metadata or metadata['rating'] <= 1:
                continue
            cur_cam_ids = _search_cam_ids(scene_dir)
            for cam_id in cur_cam_ids:
                if top_down_view and cam_id not in top_down_cam_ids[task_config]:
                    continue
                if cam_id in in_hand_cam_ids[task_config]:
                    # print(scene_dir, cam_id)
                    continue
                if 'bad_calib_view' in metadata and cam_id[4:] in metadata['bad_calib_view']:
                    # print(scene_dir, cam_id)
                    continue
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

def parse_action_preds(action_preds, max_gripper_width=0.11, tcp_in_center=False, center=None, top_down_view=False):
    ''' logits: numpy.ndarray, [B,T,8]
    '''
    if tcp_in_center:
        trans_min, trans_max = np.array([-0.15, -0.15, -0.15]), np.array([0.15, 0.15, 0.15]) # TODO: need adjustification
    elif top_down_view:
        trans_min, trans_max = np.array([-0.35, -0.35, 0]), np.array([0.35, 0.35, 0.7]) # TODO: need adjustification
    else:
        trans_min, trans_max = np.array([-0.64, -0.64, 0]), np.array([0.64, 0.64, 1.28]) # TODO: need adjustification

    trans_preds = action_preds[...,0:3]
    trans_preds = (trans_preds + 1) / 2.0
    trans_preds = trans_preds * (trans_max - trans_min) + trans_min
    if tcp_in_center and center is not None:
        trans_preds = trans_preds + center

    quat_preds = action_preds[...,3:7]
    quat_preds /=  np.linalg.norm(quat_preds, axis=2, keepdims=True) + 1e-6

    gripper_width_preds = action_preds[...,7:8]
    gripper_width_preds = (gripper_width_preds + 1) / 2.0
    gripper_width_preds = gripper_width_preds * max_gripper_width

    action_preds = np.concatenate([trans_preds, quat_preds, gripper_width_preds], axis=-1)

    return action_preds

def compute_action_error(action_preds, action_labels):
    trans_error = np.linalg.norm(action_preds[...,:3]-action_labels[...,:3], axis=-1).mean()
    width_error = np.abs(action_preds[...,7]-action_labels[...,7]).mean()
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
    task_config_list = ['RH100T_cfg1','RH100T_cfg2','RH100T_cfg3','RH100T_cfg4','RH100T_cfg5','RH100T_cfg6','RH100T_cfg7']
    # task_config_list = ['RH100T_cfg2','RH100T_cfg3','RH100T_cfg4','RH100T_cfg5','RH100T_cfg6','RH100T_cfg7']
    task_descriptions = json.load(open('task_list/task_description.json'))
    dataset = RH100TDataset(dataset_root, task_config_list, 'all', task_descriptions, num_input=2, horizon=16, timestep=-1, filter_frames=True, from_cache=True, augmentation=False, top_down_view=True)
    # for i in tqdm(range(0,len(dataset),1000)):
    #     batch = dataset[i]
    xmin, xmax, ymin, ymax, zmin, zmax = 1, -1, 1, -1, 2, 0
    cnt = 0
    for i in tqdm(range(0,len(dataset),1000)):
        batch = dataset[i]
        tcp_list = batch['target_frame_tcp_list']
        _xmin, _xmax = tcp_list[:,0].min(), tcp_list[:,0].max()
        _ymin, _ymax = tcp_list[:,1].min(), tcp_list[:,1].max()
        _zmin, _zmax = tcp_list[:,2].min(), tcp_list[:,2].max()
        if _xmin < -0.63 or _xmax > 0.63: cnt+=1; continue
        if _ymin < -0.63 or _ymax > 0.63: cnt+=1; continue
        # if _zmin < 0.01 or _zmax > 1.27: cnt+=1; continue
        xmin = min(xmin, _xmin)
        xmax = max(xmax, _xmax)
        ymin = min(ymin, _ymin)
        ymax = max(ymax, _ymax)
        zmin = min(zmin, _zmin)
        zmax = max(zmax, _zmax)
    print('xmin:', xmin)
    print('xmax:', xmax)
    print('ymin:', ymin)
    print('ymax:', ymax)
    print('zmin:', zmin)
    print('zmax:', zmax)
    print('error count:', cnt)