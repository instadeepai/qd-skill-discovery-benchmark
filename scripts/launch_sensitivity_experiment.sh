#!/bin/bash
python qdbenchmark/training/train_dads_reward.py -m --config-name ant_omni \
    seed=0,1,2,3,4 \
    iso_sigma=0.001,0.01,0.1 \
    line_sigma=0.01,0.1,0.5 \


python qdbenchmark/training/train_map_elites.py -m --config-name ant_omni \
    seed=0,1,2,3,4 \
    diversity_reward_scale=0.1,1.0,10 \
    alpha_init=0.1,0.5,1.0


