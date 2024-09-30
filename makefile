all: build run

build:
	docker build -f ./dockerfiles/tf.Dockerfile -t tf_ml_in_finance .

run:
	docker run -it --rm --group-add keep-groups \
		-v $(shell pwd):/src \
		tf_ml_in_finance \
		bash
