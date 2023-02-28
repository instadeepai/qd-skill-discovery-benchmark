#!/bin/bash
declare -a qd_algorithms=(
        "train_map_elites.py"
        "train_pga_me.py"
        #"train_aurora"
        #"train_pga_aurora"
)

declare -a sd_algorithms=(
        "train_dads_reward.py"
        "train_diayn_reward.py"
        "train_dads_smerl.py"
        "train_diayn_smerl.py"

)

declare -a environments=(
        "ant_uni"
        "halfcheetah_uni"
        "walker2d_uni"
        "antmaze"

)


for env in ${environments[@]}; do
    for alg in ${algorithms[@]}; do
        python qdbenchmark/training/$alg -m --config-name $env seed=0\
                time_limit=30\
                buffer_size=100_000
    done
done

