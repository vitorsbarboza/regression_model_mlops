# Makefile para pipeline de MLOps

install:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

preprocess:
	. .venv/bin/activate && python src/data_preprocessing.py

train:
	. .venv/bin/activate && python src/train.py

predict:
	. .venv/bin/activate && python src/predict.py

mlflow-ui:
	. .venv/bin/activate && mlflow ui
