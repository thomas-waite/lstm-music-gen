SHELL = /bin/sh

model_image:
	python3 generate_model_image.py

prep_data:
	python3 prepare_data.py

train:
	python3 train.py