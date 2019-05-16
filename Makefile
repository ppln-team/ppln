APP_NAME=amirassov/ppln
CONTAINER_NAME=ppln

# HELP
.PHONY: help

help: ## This help.
	@awk 'BEGIN (FS = ":.*?## ") /^[a-zA-Z_-]+:.*?## / (printf "\033[36m%-30s\033[0m %s\n", $$1, $$2)' $(MAKEFILE_LIST)

build:  ## Build the container
	nvidia-docker build -t $(APP_NAME) .

run-omen: ## Run container in omen
	nvidia-docker run \
		-itd \
		--ipc=host \
		--name=$(CONTAINER_NAME) \
		-v /home/videoanalytics/data:/data \
		-v $(shell pwd):/ppln $(APP_NAME) bash

run-dl2: ## Run container in omen
	nvidia-docker run \
		-itd \
		--ipc=host \
		--name=$(CONTAINER_NAME) \
		-v /mnt/hdd1/amirassov/dumps:/dumps \
		-v $(shell pwd):/ppln $(APP_NAME) bash

exec: ## Run a bash in a running container
	nvidia-docker exec -it $(CONTAINER_NAME) bash

stop: ## Stop and remove a running container
	docker stop $(CONTAINER_NAME); docker rm $(CONTAINER_NAME)
