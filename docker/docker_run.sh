#!/bin/bash

get_available_gpu() {
    local mem_threshold=500
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | 
    awk -v threshold="$mem_threshold" -F', ' '$2 < threshold { print $1; exit }'
} # comment out "exit" if wanted to see all available gpus
GPU_ID=$(get_available_gpu)
time=$(date +%Y%m%d_%H%M%S)
username=$(id -u -n)
container_name="${username}_zeroshot-hierarchical_${time}"
docker run --gpus "device=$GPU_ID" \
        -it --rm \
        --shm-size=50G \
	-u $(id -u $username):$(id -g $username) \
        --name "$container_name" \
        -v /mnt/workspace2024/xing/zeroshot_hierarchical:/home/${username}/mnt/workspace/zeroshot_hierarchical \
        -v /mnt/workspace2024/xing/dataset/zeroshot_hierarchical:/datasets \
        -v ~/:/home/${username} \
        repo-luna.ist.osaka-u.ac.jp:5000/${username}/zeroshot-hierarchical:latest bash
