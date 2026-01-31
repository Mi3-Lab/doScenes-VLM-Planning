#!/bin/bash

# Loop through all scenes
for i in $(seq 61 1109); do
    echo "========= Running scene $i ========="
    printf "%d\n\n" "$i" | python main.py --model-path llava-v1.6-mistral-7b --method openemma --dataroot /data/nuscenes
done
