from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# from diffusion_policy.model.common.normalizer import LinearNormalizer
from .diffusion_modules.diffusion.conditional_unet1d import ConditionalUnet1D
from .diffusion_modules.diffusion.mask_generator import LowdimMaskGenerator
from .vision import get_resnet
# from .diffusion_modules.common.pytorch_util import dict_apply

class DiffusionUNetImagePolicy(nn.Module):
    def __init__(self,
            args_override,
            action_dim,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            obs_input_dim,
            obs_feature_dim,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=128,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # create obs encoder and get feature dim
        self.obs_encoder_top = get_resnet()
        # self.obs_encoder_wrist = get_resnet()

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        # create noise scheduler
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon"
        )

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_top) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # assert 'past_action' not in obs_dict # not implemented yet
        # # load input
        # coords = batch['input_coords_list'] # (B*To*N, 3)
        # feats = batch['input_feats_list'] # (B*To*N, 6)
        # sinput = ME.SparseTensor(feats, coords)
        # text_embeddings = batch['text_embedding'] # (B, 1024)
        # gripper_states = batch['target_frame_tcp_normalized'][:,:self.n_obs_steps] # (B, To, Da)

        B = obs_top.shape[0]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = obs_top.device
        dtype = obs_top.dtype

        obs_features_top = self.obs_encoder_top(obs_top)
        # obs_features_wrist = self.obs_encoder_wrist(obs_wrist)
        # obs_features = torch.cat([obs_features_top, obs_features_wrist], dim=1)
        obs_features = obs_features_top
        assert obs_features.shape[0] == B * To
        
        # handle different ways of passing observation
        # condition through global feature
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # reshape back to B, Do
            global_cond = obs_features.reshape(B, -1)
            # # film conditioning
            # global_cond = self.film_condition(global_cond, text_embeddings)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # # condition through impainting
            # this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            # nobs_features = self.obs_encoder(this_nobs)
            # # reshape back to B, T, Do
            # nobs_features = nobs_features.reshape(B, To, -1)
            # cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            # cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            # cond_data[:,:To,Da:] = nobs_features
            # cond_mask[:,:To,Da:] = True
            assert False

        # run sampling
        # cond_data = cond_data.view(B, T*Da)
        # cond_mask = cond_mask.view(B, T*Da)
        sample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        # sample = sample.view(B, T, Da)
        
        action_pred = sample[...,:Da]

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    # def set_normalizer(self, normalizer: LinearNormalizer):
    #     self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, obs_top, actions):
        ''' qpos: [action_dim,]
            sinput: [B*To*N, 6]
            language_embed: [B, 77, 1024]
            language_mask: [B, 77]
            actions: [B, T, action_dim]
            is_pad: [B, T]
        '''
        # # load input
        # coords_batch = batch['input_coords_list'] # (B*To*N, 3)
        # feats_batch = batch['input_feats_list'] # (B*To*N, 6)
        # sinput = ME.SparseTensor(feats_batch, coords_batch)
        # nactions = batch['target_frame_tcp_normalized'] # (B, T, action_dim)
        # text_embeddings = batch['text_embedding'] # (B, 1024)

        batch_size = actions.shape[0]
        horizon = actions.shape[1]
        action_dim = actions.shape[2]

        # # add guassian noises to text_embeddings during training (BC-Z)
        # gaussian_noises = torch.normal(mean=0.0, std=0.02, size=text_embeddings.size(), dtype=text_embeddings.dtype)
        # text_embeddings = text_embeddings + gaussian_noises.to(text_embeddings.device)
        # # text_embeddings = text_embeddings.unsqueeze(1).repeat(1,self.n_obs_steps,1).view(batch_size*self.n_obs_steps, -1)

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = actions
        cond_data = trajectory

        # reshape B, T, ... to B*T
        # obs = obs.reshape(-1, *obs.shape[2:]) # (B*T, 3, H, W)
        # # obs_features = self.obs_encoder.extract_features(obs, text_embeddings, point_embeddings) # (B*T, C, h, w)
        # obs_features = self.obs_encoder.extract_features(obs, text_embeddings, points) # (B*T, C, h, w)
        # obs_features = self.obs_feature_pool(obs_features).flatten(start_dim=1) # (B*T, C)
        # soutput = self.obs_encoder(sinput)
        # sfeat = self.obs_feature_pool(soutput)
        # obs_features = sfeat.F

        # print("obs_top shape:", obs_top.shape)
        obs_features_top = self.obs_encoder_top(obs_top)
        # obs_features = torch.cat([obs_features_top, obs_features_wrist], dim=1)
        obs_features = obs_features_top
        assert obs_features.shape[0] == batch_size * self.n_obs_steps
            
        if self.obs_as_global_cond:
            # reshape back to B, Do
            global_cond = obs_features.reshape(batch_size, -1) # (B, T*C)
            # obs_features = obs_features.reshape(batch_size, -1) # (B, T*C)
            # # film conditioning
            # global_cond = self.film_condition(global_cond, text_embeddings)
        else:
            # # reshape B, T, ... to B*T
            # this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            # nobs_features = self.obs_encoder(this_nobs)
            # # reshape back to B, T, Do
            # nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            # cond_data = torch.cat([nactions, nobs_features], dim=-1)
            # trajectory = cond_data.detach()
            assert False

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)
        # noisy_trajectory = noisy_trajectory.view(batch_size, horizon*action_dim)
        # pred = self.model(obs_features, noisy_trajectory, timesteps.unsqueeze(-1).float())
        # pred = pred.view(batch_size, horizon, action_dim)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
