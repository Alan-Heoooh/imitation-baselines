# ps -ef | grep 'finetune.py' | awk '{for(i=1;i<=NF;i++)print "kill -9 " $2;}' | sudo sh

# torchrun --master_addr 192.168.3.50 --master_port 12355 --nproc_per_node 10 --nnodes 1 --node_rank 0 finetune.py --ckpt_dir logs_6025/log_20240209_usepretrain5_task520_50sample_rot6d --policy_class Diffusion --batch_size 24 --seed 233 --num_epoch 1000 --save_epoch 50 --lr 3e-4 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --dim_feedforward 2048 --task_name task_0013 --resume_ckpt logs_6025/log_20240202_pretrain_actdiffusion-256-512_enc4dec4_minkres14_dim2048_rot6d/policy_epoch_5_seed_233.ckpt

# CUDA_VISIBLE_DEVICES=2 torchrun --master_addr 192.168.3.50 --master_port 12356 --nproc_per_node 1 --nnodes 1 --node_rank 0 finetune.py --ckpt_dir logs_6025/log_20240216_task513_50sample_rot6d --policy_class ACT --batch_size 64 --seed 233 --num_epoch 1000 --save_epoch 50 --lr 5e-5 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --dim_feedforward 3200 --task_name task_0013

# CUDA_VISIBLE_DEVICES=4 torchrun --master_addr 192.168.3.50 --master_port 12357 --nproc_per_node 1 --nnodes 1 --node_rank 0 finetune.py --ckpt_dir logs_6025/log_20240216_task515_50sample_rot6d --policy_class ACT --batch_size 64 --seed 233 --num_epoch 1000 --save_epoch 50 --lr 5e-5 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --dim_feedforward 3200 --task_name task_0013

# CUDA_VISIBLE_DEVICES=6 torchrun --master_addr 192.168.3.50 --master_port 12358 --nproc_per_node 1 --nnodes 1 --node_rank 0 finetune.py --ckpt_dir logs_6025/log_20240216_task516_50sample_rot6d --policy_class ACT --batch_size 64 --seed 233 --num_epoch 1000 --save_epoch 50 --lr 5e-5 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --dim_feedforward 3200 --task_name task_0013

# CUDA_VISIBLE_DEVICES=0 torchrun --master_addr 192.168.3.50 --master_port 12355 --nproc_per_node 1 --nnodes 1 --node_rank 0 finetune.py --ckpt_dir logs_6025/log_20240218_task517_50sample_rot6d --policy_class ACT --batch_size 64 --seed 233 --num_epoch 1000 --save_epoch 50 --lr 5e-5 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --dim_feedforward 3200 --task_name task_0013

# CUDA_VISIBLE_DEVICES=0 torchrun --master_addr 192.168.3.50 --master_port 12355 --nproc_per_node 1 --nnodes 1 --node_rank 0 finetune.py --ckpt_dir logs_6025/log_20240218_task517_50sample_rot6d --policy_class ACT --batch_size 64 --seed 233 --num_epoch 1000 --save_epoch 50 --lr 5e-5 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --dim_feedforward 3200 --task_name task_0013

# CUDA_VISIBLE_DEVICES=8 torchrun --master_addr 192.168.3.50 --master_port 12359 --nproc_per_node 1 --nnodes 1 --node_rank 0 finetune.py --ckpt_dir logs_6025/log_20240218_task518_50sample_rot6d --policy_class ACT --batch_size 64 --seed 233 --num_epoch 1000 --save_epoch 50 --lr 5e-5 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --dim_feedforward 3200 --task_name task_0013

# CUDA_VISIBLE_DEVICES=4 torchrun --master_addr 192.168.3.50 --master_port 12357 --nproc_per_node 1 --nnodes 1 --node_rank 0 finetune.py --ckpt_dir logs_6025/log_20240218_task519_50sample_rot6d --policy_class ACT --batch_size 64 --seed 233 --num_epoch 1000 --save_epoch 50 --lr 5e-5 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --dim_feedforward 3200 --task_name task_0013

# CUDA_VISIBLE_DEVICES=0,1 torchrun --master_addr 192.168.3.23 --master_port 12355 --nproc_per_node 2 --nnodes 1 --node_rank 0 finetune.py --ckpt_dir logs_6024/log_20240226_task521_50sample_rot6d --policy_class ACT --batch_size 64 --seed 233 --num_epoch 1000 --save_epoch 50 --lr 5e-5 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --dim_feedforward 3200 --task_name task_0013

# CUDA_VISIBLE_DEVICES=8,9 torchrun --master_addr 192.168.3.50 --master_port 12355 --nproc_per_node 2 --nnodes 1 --node_rank 0 finetune.py --ckpt_dir logs/test/20240913_cup_mix_2 --policy_class ACT --batch_size 64 --seed 233 --num_epoch 1000 --save_epoch 50 --lr 5e-5 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --dim_feedforward 3200 --task_name task_0013 --dataset_root /data/zihao_6024/data/wipe

python3 finetune.py --ckpt_dir logs/test/pich_and_place --policy_class ACT --batch_size 64 --seed 233 --num_epoch 1000 --save_epoch 50 --lr 5e-5 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --dim_feedforward 3200 --task_name task_0013 --dataset_root /zihao-fast-vol/vr_data