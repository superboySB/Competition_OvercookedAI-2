#!/bin/bash
# 设置变量
episode_length=400
num_env_steps=10000000
n_rollout_threads=128
over_layouts=("cramped_room_tomato" "forced_coordination_tomato" "soup_coordination")

# for 循环遍历 over_layouts 数组
for layout in "${over_layouts[@]}"
do
    python ../train/trainer.py --num_env_steps $num_env_steps \
                               --pop_size 1 \
                               --episode_length $episode_length \
                               --env_length $episode_length \
                               --env_name overcooked \
                               --seed 1 \
                               --restored 0 \
                               --n_rollout_threads $n_rollout_threads \
                               --ppo_epoch 10 \
                               --cuda \
                               --layer_N 2 \
                               --hidden_size 64 \
                               --lr 1e-2 \
                               --critic_lr 1e-2 \
                               --over_layout $layout \
                               --run_dir sp
done