.PHONY: setup data baseline test clean

# 1. Setup Environment
setup:
	pip install -r requirements.txt
	pip install -e .

# 2. Download Multi-Dataset (500 samples per type)
data:
	python scripts/00_data_setup.py --samples 500

# 2b. Download Small Dataset for Testing (50 samples per type)
data-small:
	python scripts/00_data_setup.py --samples 50

# 3. Run Baseline Evaluation on All 3 Datasets
baseline:
	python scripts/01_run_baseline.py

# 4. Run Tests
test:
	pytest tests/

# 5. Clean up junk
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info