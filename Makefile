# Makefile
.PHONY: setup data test run

setup:
	pip install -e .
	pip install -r requirements.txt

data:
	python scripts/01_generate_data.py --config configs/baseline.yaml

test:
	pytest tests/

run:
	python scripts/02_train_probes.py