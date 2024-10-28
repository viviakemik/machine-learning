all: check-docker
docker: build_docker run_docker
podman: build_podman run_podman






##############################################
#  _____                              ____        _ _     _
# |_   _|                            |  _ \      (_) |   | |
#   | |  _ __ ___   __ _  __ _  ___  | |_) |_   _ _| | __| |
#   | | | '_ ` _ \ / _` |/ _` |/ _ \ |  _ <| | | | | |/ _` |
#  _| |_| | | | | | (_| | (_| |  __/ | |_) | |_| | | | (_| |
# |_____|_| |_| |_|\__,_|\__, |\___| |____/ \__,_|_|_|\__,_|
#                         __/ |
#                        |___/
##############################################

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






#########################################################################
#       _                   _               _____
#      | |                 | |             / ____|
#      | |_   _ _ __  _   _| |_ ___ _ __  | (___   ___ _ ____   _____ _ __
#  _   | | | | | '_ \| | | | __/ _ \ '__|  \___ \ / _ \ '__\ \ / / _ \ '__|
# | |__| | |_| | |_) | |_| | ||  __/ |     ____) |  __/ |   \ V /  __/ |
#  \____/ \__,_| .__/ \__, |\__\___|_|    |_____/ \___|_|    \_/ \___|_|
#              | |     __/ |
#              |_|    |___/
##########################################################################




jupyter: 
	@which podman > /dev/null 2>&1 && $(MAKE) jupyter_podman || $(MAKE) jupyter_alt
jupyter_alt: 
	@which docker > /dev/null 2>&1 && $(MAKE) jupyter_docker || $(MAKE) no-container



jupyter_docker:
	@read -p "Type the Jupyter Port:" PORT; \
        read -p "Type the Jupyter Password:" PASSWD; \
        docker run -it --rm --group-add keep-groups \
        -v $(shell pwd):/src \
        -p $${PORT}:$${PORT} \
        tf_env \
        jupyter lab --port=$${PORT} --NotebookApp.token=$${PASSWD} --allow-root

jupyter_podman:
	@read -p "Type the Jupyter Port:" PORT; \
        read -p "Type the Jupyter Password:" PASSWD; \
        podman run -it --rm --group-add keep-groups \
        -v $(shell pwd):/src \
        -p $${PORT}:$${PORT} \
        tf_env \
        jupyter lab --port=$${PORT} --NotebookApp.token=$${PASSWD} --allow-root









# Error Sequence
no-container:
	@echo "Error: Neither Docker nor Podman is installed on this system."
