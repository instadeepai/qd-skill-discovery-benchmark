SHELL := /bin/bash

# variables
WORK_DIR = $(PWD)
DOCKER_RUN_FLAGS = --rm --gpus all
DOCKER_IMAGE_NAME = instadeep/qdbenchmark:$(USER)
USER_ID = $$(id -u)
GROUP_ID = $$(id -g)




.PHONY: build
build:
	sudo docker build -t $(DOCKER_IMAGE_NAME) -f dev.gpu.Dockerfile . --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID)

.PHONY: dev_container
dev_container: build
	sudo docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) /bin/bash


.PHONY: train
train: build
	sudo docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) /bin/bash \
				-c "python qdbenchmark/training/$(script_name) --config-name $(env_name)"


.PHONY: adaptation_gravity_qd
adaptation_gravity_qd: build
	sudo docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) /bin/bash \
				-c "python qdbenchmark/adaptation/qd_adaptation.py -m\
					    env_name=$(env_name) \
						algorithm_name=$(algorithm_name) \
						run_config_path=/app/$(run_config_path) \
						repertoire_path=/app/$(repertoire_path) \
						adaptation_name=gravity_multiplier \
						adaptation_idx=$$(seq -s , 19)"
.PHONY: adaptation_gravity_sd
adaptation_gravity_sd: build
	sudo docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) /bin/bash \
				-c "python qdbenchmark/adaptation/sd_adaptation.py -m\
					    env_name=$(env_name) \
						algorithm_name=$(algorithm_name) \
						run_config_path=/app/$(run_config_path) \
						policy_path=/app/$(policy_path) \
						adaptation_name=gravity_multiplier \
						adaptation_idx=$$(seq -s , 19)"

.PHONY: adaptation_actuator_qd
adaptation_actuator_qd: build
	sudo docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) /bin/bash \
				-c "python qdbenchmark/adaptation/qd_adaptation.py -m\
					    env_name=$(env_name) \
						algorithm_name=$(algorithm_name) \
						run_config_path=/app/$(run_config_path) \
						repertoire_path=/app/$(repertoire_path) \
						adaptation_name=actuator_update \
						adaptation_idx=$$(seq -s , 19)"

.PHONY: adaptation_actuator_sd
adaptation_actuator_sd: build
	sudo docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) /bin/bash \
				-c "python qdbenchmark/adaptation/sd_adaptation.py -m\
					    env_name=$(env_name) \
						algorithm_name=$(algorithm_name) \
						run_config_path=/app/$(run_config_path) \
						policy_path=/app/$(policy_path) \
						adaptation_name=actuator_update \
						adaptation_idx=$$(seq -s , 19)"

.PHONY: adaptation_position_qd
adaptation_position_qd: build
	sudo docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) /bin/bash \
				-c "python qdbenchmark/adaptation/qd_adaptation.py -m\
					    env_name=$(env_name) \
						algorithm_name=$(algorithm_name) \
						run_config_path=/app/$(run_config_path) \
						repertoire_path=/app/$(repertoire_path) \
						adaptation_name=default_target_position \
						adaptation_idx=$$(seq -s , 9) "

.PHONY: adaptation_position_sd
adaptation_position_sd: build
	sudo docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) /bin/bash \
				-c "python qdbenchmark/adaptation/sd_adaptation.py -m\
					    env_name=$(env_name) \
						algorithm_name=$(algorithm_name) \
						run_config_path=/app/$(run_config_path) \
						policy_path=/app/$(policy_path) \
						adaptation_name=default_target_position \
						adaptation_idx=$$(seq -s , 9)"

.PHONY: hierarchical_sd
hierarchical_sd: build
	sudo docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) /bin/bash \
				-c "python qdbenchmark/hierarchical/train_hierarchical_sd.py -m\
						algorithm_name=$(algorithm_name) \
						policy_path=/app/$(policy_path) "	


.PHONY: hierarchical_sd
hierarchical_qd: build
	sudo docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) /bin/bash \
				-c "python qdbenchmark/hierarchical/train_hierarchical_qd.py -m\
						algorithm_name=$(algorithm_name) \
						repertoire_path=/app/$(repertoire_path) "
