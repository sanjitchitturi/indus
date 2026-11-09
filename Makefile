.PHONY: help install install-dev lint format test test-cov clean train evaluate infer baseline sweep docker-build docker-run mlflow-ui generate-data

PYTHON := python3
SRC := src/indus
TESTS := tests

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	$(PYTHON) -m pip install -e ".[train]"

install-dev: ## Install with development dependencies
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Run linting checks
	ruff check $(SRC) $(TESTS) scripts/
	black --check $(SRC) $(TESTS) scripts/

format: ## Auto-format code
	ruff check --fix $(SRC) $(TESTS) scripts/
	black $(SRC) $(TESTS) scripts/

test: ## Run tests
	pytest $(TESTS) -v

test-cov: ## Run tests with coverage
	pytest $(TESTS) -v --cov=$(SRC) --cov-report=term-missing --cov-report=html

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov/
	rm -rf outputs/ multirun/
	find . -type d -name __pycache__ -exec rm -rf {} +

train: ## Run fine-tuning with default config
	$(PYTHON) scripts/train.py

evaluate: ## Run evaluation with default config
	$(PYTHON) scripts/evaluate.py

infer: ## Run inference demo
	$(PYTHON) scripts/infer.py

baseline: ## Run baseline evaluation (untuned model)
	$(PYTHON) scripts/run_baseline.py

sweep: ## Run experiment sweep across configurations
	$(PYTHON) scripts/run_sweep.py

generate-data: ## Generate synthetic sample dataset
	$(PYTHON) scripts/generate_sample_data.py

docker-build: ## Build Docker image
	docker build -t indus:latest .

docker-run: ## Run training inside Docker
	docker compose up training

mlflow-ui: ## Launch MLflow tracking UI
	mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns


demo: ## Run the CPU baseline and generate a reviewable HTML report
	$(PYTHON) -m indus demo
