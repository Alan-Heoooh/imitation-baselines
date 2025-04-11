import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from .diffusion import DiffusionUNetImagePolicy


class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        # model, optimizer = build_ACT_model_and_optimizer(args_override)
        # self.model = model # CVAE decoder
        # self.optimizer = optimizer
        action_dim = args_override['state_dim']
        n_action_steps = args_override['num_queries']
        n_obs_steps = 1
        horizon = n_obs_steps + n_action_steps - 1
        obs_input_dim = 3
        obs_feature_dim = 512 * 2
        self.model = DiffusionUNetImagePolicy(args_override, action_dim, horizon, n_action_steps, n_obs_steps, obs_input_dim, obs_feature_dim)
        self.model.cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args_override['lr'], betas=[0.95,0.999], weight_decay=1e-6)
        print("Model params: %e" % sum(p.numel() for p in self.model.parameters()))

    def __call__(self, obs_top, obs_wrist, actions=None):
        if actions is not None: # training time
            l2 = self.model.compute_loss(obs_top, obs_wrist, actions)
            loss_dict = dict()
            loss_dict['loss'] = l2
            return loss_dict
        else: # inference time
            result = self.model.predict_action(obs_top, obs_wrist)
            return result['action']

    def configure_optimizers(self):
        return self.optimizer

