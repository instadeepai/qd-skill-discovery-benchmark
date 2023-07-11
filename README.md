# ICLR 2023 - Neuroevolution is a competitive alternative to reinforcement learning for skill discovery

[![PyPI Python Version](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Jax 0.3.10](https://img.shields.io/badge/jax-0.3.10-informational?logo=data:image/png;base64,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)](https://jax.readthedocs.io/en/latest/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)




<img align="right" src="https://github.com/instadeepai/qd-skill-discovery-benchmark/blob/main/images/pga_me_halfcheetah.gif" alt="alternate text" width="400" height="400">


This repository contains the code for the paper "Neuroevolution is a Competitive Alternative to Reinforcement Learning for Skill Discovery.", [(Chalumeau, Boige et al., 2023)](https://openreview.net/forum?id=6BHlZgyPOZY) :computer: :zap:.

## First-time setup

### Install Docker
This code requires docker to run. To install docker please follow the online instructions 
[here](https://docs.docker.com/engine/install/ubuntu/). To enable the code to run on GPU, please
install [Nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (as well as
the latest nvidia driver available for your GPU).

### Build and run a docker image
Once docker and docker Nvidia are installed, you can simply build the docker image with the following command:
```bash
make build
```
and, once the image is built, start the container with:
```bash
make dev_container
```
Inside the container, you can run the `nvidia-smi` command to verify that your GPU is found.


## Run experiments from the paper

### Training
#### Syntax
To train an algorithm on an environment use the following command:
```python3
make train script_name=[SCRIPT_NAME] env_name=[ENV_NAME]
```
This will load the default config in `qdbenchmark/training/config` and launch the training. The list of available scripts and configs are present in `qdbenchmark/training/`.


#### Example
For instance to train **MAP-ELITES** on the environment **Halfcheetah Uni** run:

```python
make train script_name=train_map_elites.py env_name=halfcheetah_uni
```

### Adaptation 
#### Syntax
To perform adaptations tasks, the user must provide a path to the policy/repertoire resulting from the training of an agent as well as the path to the config used to train this agent. Three commands are available, one for each of the three adaptation-task families (Gravity multiplier, Actuator update or Default position change).


For the QD algorithms:
```python3
make adaptation_gravity_qd repertoire_path=[PATH_TO_REPERTOIRE] run_config_path=[PATH_TO_CONFIG] env_name=[ENV_NAME] algorithm_name=[ALGORITHM_NAME]
```
```python3
make adaptation_actuator_qd repertoire_path=[PATH_TO_REPERTOIRE] run_config_path=[PATH_TO_CONFIG] env_name=[ENV_NAME] algorithm_name=[ALGORITHM_NAME]
```
```python3
make adaptation_position_qd repertoire_path=[PATH_TO_REPERTOIRE] run_config_path=[PATH_TO_CONFIG] env_name=[ENV_NAME] algorithm_name=[ALGORITHM_NAME]
```

and for the Skill Discovery algorithms:
```python3
make adaptation_gravity_sd policy_path=[PATH_TO_POLICY] run_config_path=[PATH_TO_CONFIG] env_name=[ENV_NAME] algorithm_name=[ALGORITHM_NAME]
```
```python3
make adaptation_actuator_sd policy_path=[PATH_TO_POLICY] run_config_path=[PATH_TO_CONFIG] env_name=[ENV_NAME] algorithm_name=[ALGORITHM_NAME]
```
```python3
make adaptation_position_sd policy_path=[PATH_TO_POLICY] run_config_path=[PATH_TO_CONFIG] env_name=[ENV_NAME] algorithm_name=[ALGORITHM_NAME]
```

#### Example

Sample configs and checkpoints are provided to give an example, for instance the user can run the following command:
```python3
make adaptation_gravity_sd policy_path=sample/sample_policy/dads-reward-ant-uni-policy-0.npy run_config_path=sample/sample_config/dads_reward_ant_uni.yaml env_name=ant_uni algorithm_name=DADS+REWARD
```

### Hierarchical
#### Syntax
To perform the Halfcheetah-Hurdle hierarchical task, the user should also provide a policy/repertoire path. The syntax is the following:

For QD algorithms:
```python
make hierarchical_qd repertoire_path=[PATH_TO_REPERTOIRE] algorithm_name=[ALGORITHM_NAME]
```

For Skill Discovery algorithms:
```python
make hierarchical_sd policy_path=[PATH_TO_POLICY] algorithm_name=[ALGORITHM_NAME]
```

#### Example
Sample checkpoints are provided to give an example, for instance the user can run the following command:
```python3
make hierarchical_qd repertoire_path=sample/sample_repertoire/map_elites_halfcheetah_uni/ algorithm_name=MAP-ELITES
```




## Contributors

<a href="https://github.com/felixchalumeau" title="Felix Chalumean"><img src="https://github.com/felixchalumeau.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/egiob" title="Raphaël Boige"><img src="https://github.com/egiob.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/limbryan" title="Bryan Lim"><img src="https://github.com/limbryan.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/valentinmace" title="Valentin Macé"><img src="https://github.com/valentinmace.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/maxiallard" title="Maxime Allard"><img src="https://github.com/maxiallard.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/flajolet" title="Arthur Flajolet"><img src="https://github.com/flajolet.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/Aneoshun" title="Antoine Cully"><img src="https://github.com/Aneoshun.png" height="auto" width="50" style="border-radius:50%"></a>
<a href="https://github.com/ranzenTom" title="Thomas Pierrot"><img src="https://github.com/ranzenTom.png" height="auto" width="50" style="border-radius:50%"></a>

## Citing this work

If you use the code or data in this package, please cite:

```bibtex
@inproceedings{
chalumeau2023neuroevolution,
title={Neuroevolution is a Competitive Alternative to Reinforcement Learning for Skill Discovery},
author={Felix Chalumeau and Raphael Boige and Bryan Lim and Valentin Mac{\'e} and Maxime Allard and Arthur Flajolet and Antoine Cully and Thomas PIERROT},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=6BHlZgyPOZY}
}
