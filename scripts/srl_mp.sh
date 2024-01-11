#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# 定义种子值数组
seeds=(0 10 20 30 40 50)

# 循环遍历每个种子值
for seed in "${seeds[@]}"
do
    # 第一个Python程序
    python ../train/rllib_trainer.py \
            --map-name cramped_room_tomato \
            --clip_param 0.1543 \
            --gamma 0.9777 \
            --grad_clip 0.2884 \
            --kl_coeff 0.2408 \
            --lmbda 0.6 \
            --lr 2.69e-4 \
            --num_training_iters 500 \
            --reward_shaping_horizon 4500000 \
            --use_phi False \
            --vf_loss_coeff 0.0069 \
            --seed $seed

    # 第二个Python程序
    python ../train/rllib_trainer.py \
            --map-name forced_coordination_tomato \
            --clip_param 0.0608 \
            --gamma 0.9738 \
            --grad_clip 0.3022 \
            --kl_coeff 0.2527 \
            --lmbda 0.8 \
            --lr 2.5e-4 \
            --num_training_iters 1000 \
            --reward_shaping_horizon 2500000 \
            --use_phi True \
            --vf_loss_coeff 0.009 \
            --seed $seed

    # 第三个Python程序
    python ../train/rllib_trainer.py \
            --map-name soup_coordination \
            --clip_param 0.1245 \
            --gamma 0.966 \
            --grad_clip 0.2469 \
            --kl_coeff 0.2355 \
            --lmbda 0.5 \
            --lr 2.07e-4 \
            --num_training_iters 1000 \
            --reward_shaping_horizon 5000000 \
            --use_phi True \
            --vf_loss_coeff 0.0158 \
            --seed $seed
done
