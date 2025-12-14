# NYC Taxi MLOps - Makefile
# ==========================
# Easy commands for the MLOps pipeline

.PHONY: help install setup data train serve test clean docker-build docker-up docker-down

# Default target
help:
	@echo "NYC Taxi MLOps - Available Commands"
	@echo "===================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install Python dependencies"
	@echo "  make setup        - Create necessary directories"
	@echo ""
	@echo "Data:"
	@echo "  make data-download  - Download NYC Taxi data"
	@echo "  make data-preprocess - Preprocess data"
	@echo "  make data-all       - Download and preprocess"
	@echo ""
	@echo "Training:"
	@echo "  make train        - Train default model (Random Forest)"
	@echo "  make train-all    - Train and compare all models"
	@echo "  make tune         - Hyperparameter tuning with Optuna"
	@echo ""
	@echo "Serving:"
	@echo "  make serve        - Start FastAPI server"
	@echo "  make mlflow       - Start MLFlow UI"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build all Docker images"
	@echo "  make docker-up    - Start all services"
	@echo "  make docker-down  - Stop all services"
	@echo "  make docker-logs  - View logs"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linter"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove cache and temp files"

# ==========================================
# SETUP
# ==========================================

install:
	pip install -r requirements.txt

setup:
	mkdir -p data/raw data/processed models reports notebooks mlruns

# ==========================================
# DATA
# ==========================================

data-download:
	python cli/main.py data download --year 2024 --months 1,2,3

data-preprocess:
	python cli/main.py data preprocess

data-sample:
	python cli/main.py data preprocess --sample 100000

data-all: data-download data-preprocess

# ==========================================
# TRAINING
# ==========================================

train:
	python cli/main.py train run --model random_forest

train-all:
	python cli/main.py train compare

train-xgboost:
	python cli/main.py train run --model xgboost

train-lightgbm:
	python cli/main.py train run --model lightgbm

tune:
	python -c "from src.models.tuning import tune_model; tune_model('random_forest')"

# ==========================================
# SERVING
# ==========================================

serve:
	python cli/main.py serve start --port 8000

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

# ==========================================
# MONITORING
# ==========================================

monitor:
	python cli/main.py monitor drift \
		--reference data/processed/train.parquet \
		--current data/processed/test.parquet

# ==========================================
# DOCKER
# ==========================================

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-mlflow:
	docker-compose up -d mlflow
	@echo "MLFlow UI: http://localhost:5000"

docker-api:
	docker-compose up -d api
	@echo "API Swagger: http://localhost:8000/docs"

docker-jupyter:
	docker-compose up -d jupyter
	@echo "Jupyter: http://localhost:8888"

docker-train:
	docker-compose run --rm training python cli/main.py train run --model random_forest

docker-train-all:
	docker-compose run --rm training python cli/main.py train compare

# ==========================================
# TESTING & QUALITY
# ==========================================

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ cli/ --max-line-length=100
	black src/ cli/ --check
	isort src/ cli/ --check-only

format:
	black src/ cli/
	isort src/ cli/

type-check:
	mypy src/ cli/

# ==========================================
# CLEANUP
# ==========================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage

clean-data:
	rm -rf data/raw/*
	rm -rf data/processed/*

clean-all: clean clean-data
	rm -rf models/*
	rm -rf reports/*
	rm -rf mlruns/*
