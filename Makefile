.PHONY: setup data baseline test clean

# 1. Setup Environment
setup:
	pip install -r requirements.txt
	pip install -e .

# 2. Get the Data
data:
	python src/data/download_anthropic.py

# 3. Run the Baseline Experiment
baseline:
	python scripts/01_check_baseline.py

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