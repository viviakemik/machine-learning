all: check-docker
docker: build_docker run_docker
podman: build_podman run_podman



# Docker Sequence
check-docker:
	@which docker > /dev/null 2>&1 && $(MAKE) docker || $(MAKE) check-podman

build_docker:
	docker build -f ./dockerfiles/tf.Dockerfile -t tf_ml_in_finance .

run_docker:
	docker run -it --rm --group-add keep-groups \
		-v $(shell pwd):/src \
		tf_ml_in_finance \
		bash


# Podman Sequence
check-podman:
	@which podman > /dev/null 2>&1 && $(MAKE) podman || $(MAKE) no-container

build_podman:
	podman build -f ./dockerfiles/tf.Dockerfile -t tf_ml_in_finance .

run_podman:
	podman run -it --rm --group-add keep-groups \
		-v $(shell pwd):/src \
		tf_ml_in_finance \
		bash



# Error Sequence
no-container:
	@echo "Error: Neither Docker nor Podman is installed on this system."
