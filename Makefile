APP_NAME=amirassov/ppln
CONTAINER_NAME=ppln

# HELP
.PHONY: help

help: ## This help.
	@awk 'BEGIN (FS = ":.*?## ") /^[a-zA-Z_-]+:.*?## / (printf "\033[36m%-30s\033[0m %s\n", $$1, $$2)' $(MAKEFILE_LIST)

build:  ## Build the container
	nvidia-docker build -t $(APP_NAME) .

run: ## Run container in omen
	nvidia-docker run \
		-itd \
		--ipc=host \
		--name=$(CONTAINER_NAME) \
		-v /home/videoanalytics/data:/data \
		-v $(shell pwd):/ppln $(APP_NAME) bash

exec: ## Run a bash in a running container
	nvidia-docker exec -it $(CONTAINER_NAME) bash

stop: ## Stop and remove a running container
	docker stop $(CONTAINER_NAME); docker rm $(CONTAINER_NAME)

clean:
	[ -e ppln.egg-info ] && rm -r ppln.egg-info ||:
	[ -e build ] && rm -r build ||:
	[ -e .eggs ] && rm -r .eggs ||:
	[ -e dist ] && rm -r dist ||:
	[ -e .pytest_cache ] && rm -r .pytest_cache ||:
	python setup.py clean

test:
	python setup.py test

install:
	python setup.py install

format:
	unify --in-place --recursive .
	yapf --in-place --recursive .
	isort --recursive .