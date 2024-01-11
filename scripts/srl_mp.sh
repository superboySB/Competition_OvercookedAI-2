#!/bin/bash

# 设置变量
export CUDA_VISIBLE_DEVICES=0

over_layouts=("cramped_room_tomato" "forced_coordination_tomato" "soup_coordination")

# 内部循环遍历 over_layouts 数组
for layout in "${over_layouts[@]}"
do
    python ../train/rllib_trainer.py \
        --map-name $layout \
        --stop-iters 5
done
