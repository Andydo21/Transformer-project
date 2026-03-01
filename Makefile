# Makefile for transformer-project
# Tự động hóa các tác vụ thường dùng

.PHONY: help install setup clean train predict test lint format

help:  ## Hiển thị help
	@echo "Available commands:"
	@echo "  make install    - Cài đặt dependencies"
	@echo "  make setup      - Setup project (install + tạo folders)"
	@echo "  make clean      - Xóa cache và temporary files"
	@echo "  make train      - Chạy training"
	@echo "  make predict    - Chạy prediction"
	@echo "  make test       - Chạy tests"
	@echo "  make lint       - Check code quality"
	@echo "  make format     - Format code"

install:  ## Cài đặt dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

setup: install  ## Setup project
	@echo "Creating directories..."
	@if not exist "outputs" mkdir outputs
	@if not exist "logs" mkdir logs
	@if not exist "data\raw" mkdir data\raw
	@if not exist "data\processed" mkdir data\processed
	@echo "Setup completed!"

clean:  ## Xóa cache và temp files
	@echo "Cleaning cache..."
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	@for /r . %%f in (*.pyc) do @if exist "%%f" del /q "%%f"
	@for /r . %%f in (*.pyo) do @if exist "%%f" del /q "%%f"
	@for /r . %%f in (*.pyd) do @if exist "%%f" del /q "%%f"
	@if exist ".pytest_cache" rd /s /q ".pytest_cache"
	@if exist ".coverage" del /q ".coverage"
	@if exist "htmlcov" rd /s /q "htmlcov"
	@echo "Cleaned!"

train:  ## Chạy training với sample data
	python scripts/quick_train.py

predict:  ## Chạy prediction
	python scripts/quick_predict.py

test:  ## Chạy all tests
	pytest tests/ -v

test-cov:  ## Chạy tests với coverage
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

lint:  ## Check code quality
	@echo "Running flake8..."
	@python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || echo "flake8 not installed"

format:  ## Format code với black
	@echo "Formatting with black..."
	@python -m black . --line-length 100 || echo "black not installed"

prepare-data:  ## Chuẩn bị sample data
	python main.py prepare-data --input-file data/sample_data.json

train-full:  ## Full training với config
	python main.py train --config configs/config.yaml

notebook:  ## Mở Jupyter notebook
	jupyter notebook notebooks/
