SHELL = /bin/sh

model_image:
	python3 ./src/generate_model_image.py

prep_data:
	python3 ./src/prepare_data.py

train:
	python3 ./src/train.py