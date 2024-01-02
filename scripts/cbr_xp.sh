#!/bin/bash

# 设置变量
num_env_steps=2000000  # 比赛：2000000, 测试：100000
episode_length=400
pop_size=8
cuda_flag=""
n_rollout_threads=256
over_layouts=("cramped_room_tomato" "forced_coordination_tomato" "soup_coordination")

# 外部循环遍历 over_layouts 数组
for layout in "${over_layouts[@]}"
do
    # 判断是否使用 CUDA
    # if [ "$layout" == "soup_coordination" ]; then
    #     cuda_flag=""
    #     n_rollout_threads=128
    # else
    #     cuda_flag="--cuda"
    #     n_rollout_threads=32
    # fi

    python ../train/best_response_trainer.py --num_env_steps $num_env_steps \
                                    --episode_length $episode_length \
                                    --env_length $episode_length \
                                    --use_linear_lr_decay \
                                    --entropy_coef 1e-3 \
                                    --env_name overcooked \
                                    --seed 1 \
                                    --restored 0 \
                                    --n_rollout_threads $n_rollout_threads \
                                    --ppo_epoch 100 \
                                    $cuda_flag \
                                    --layer_N 2 \
                                    --hidden_size 64 \
                                    --lr 1e-2 \
                                    --critic_lr 1e-2 \
                                    --over_layout $layout \
                                    --run_dir xp \
                                    --pop_size $pop_size \
                                    --xp_weight 0.25 \
                                    --mp_weight 0.0
done
