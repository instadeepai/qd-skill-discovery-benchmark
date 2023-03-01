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