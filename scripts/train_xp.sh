#!/bin/bash

for p in 8
do
    python ../train/serial_trainer.py --num_env_steps 1000000 --episode_length 200 --env_length 200 --use_linear_lr_decay --entropy_coef 0.0 --env_name overcooked --seed 1 --restored 0 --n_rollout_threads 50 --ppo_epoch 10 --cuda --layer_N 2 --hidden_size 64 --lr 1e-2 --critic_lr 1e-2 --over_layout $1 --run_dir xp --pop_size $p --xp_weight 0.25 --mp_weight 0.0

done
