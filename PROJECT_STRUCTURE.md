# Transformer Project - Complete Structure

## ✅ Project Overview

Full-featured PhoBERT-based transformer project for Vietnamese contract processing following industry best practices.

---

## 📁 Directory Structure

```
transformer-project/
├── configs/                    # Configuration files
│   └── config.yaml            # Main config (model, training, data)
│
├── data/                       # Data handling
│   ├── __init__.py
│   ├── dataset.py             # Dataset classes (ContractDataset, ContractQADataset)
│   ├── sample_data.json       # Sample data for testing
│   ├── raw/                   # Raw data (empty, placeholder)
│   └── processed/             # Processed data (empty, placeholder)
│
├── models/                     # Model definitions
│   ├── __init__.py
│   └── model.py               # ContractClassifier, ContractNER, ContractQA
│
├── training/                   # Training pipeline
│   ├── __init__.py
│   ├── train.py               # Training functions (HF Trainer & custom loop)
│   ├── eval.py                # Evaluation functions
│   └── loss.py                # Custom loss functions (Focal, LabelSmoothing, etc.)
│
├── inference/                  # Inference and prediction
│   ├── __init__.py
│   └── predict.py             # ContractPredictor class
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── tokenizer.py           # PhoBERT tokenizer utilities
│   ├── metrics.py             # Metric computation
│   ├── logger.py              # Logging utilities
│   └── config.py              # Config loader with env override
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── README.md              # Testing guide
│   ├── conftest.py            # Test fixtures
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_tokenizer.py
│   ├── test_metrics.py
│   └── test_training.py
│
├── scripts/                    # Utility scripts
│   ├── __init__.py
│   ├── README.md              # Scripts documentation
│   ├── quick_train.py         # Quick training
│   ├── quick_predict.py       # Quick prediction
│   ├── export_onnx.py         # Export to ONNX
│   ├── benchmark.py           # Performance benchmarking
│   ├── visualize.py           # Metrics visualization
│   └── serve_api.py           # FastAPI server
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_training_tutorial.ipynb
│   └── 02_data_exploration.ipynb
│
├── outputs/                    # Training outputs
│   └── .gitkeep
│
├── logs/                       # Training logs
│   └── .gitkeep
│
├── .github/                    # GitHub workflows
│   └── workflows/
│       └── ci.yml             # CI/CD pipeline
│
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
├── setup.ps1                   # Windows setup script
├── setup.sh                    # Linux/Mac setup script
├── Makefile                    # Build automation
├── Dockerfile                  # Docker image
├── docker-compose.yml          # Docker compose for services
├── .gitignore                  # Git ignore patterns
├── .env.example                # Environment variables example
├── .env.template               # Environment template
├── .pre-commit-config.yaml     # Pre-commit hooks
├── setup.cfg                   # Tool configurations (pytest, black, etc.)
├── pyproject.toml              # Python project metadata
├── README.md                   # Main documentation
├── CHANGELOG.md                # Version history
├── CONTRIBUTING.md             # Contribution guidelines
└── LICENSE                     # MIT License
```

---

## 🎯 Core Features

### 1. **Model Support**
- ✅ PhoBERT base model (vinai/phobert-base)
- ✅ Three task types: Classification, NER, QA
- ✅ Custom model architectures
- ✅ Model save/load utilities

### 2. **Training Pipeline**
- ✅ HuggingFace Trainer integration
- ✅ Custom training loop
- ✅ Multiple loss functions (Focal, LabelSmoothing, Dice)
- ✅ Early stopping & checkpointing
- ✅ TensorBoard & Weights & Biases logging
- ✅ Mixed precision training (FP16)

### 3. **Data Processing**
- ✅ Dataset classes for all task types
- ✅ PhoBERT tokenization
- ✅ Data splitting utilities
- ✅ Sample data included

### 4. **Evaluation**
- ✅ Task-specific evaluators
- ✅ Comprehensive metrics (Accuracy, F1, Precision, Recall)
- ✅ Classification reports
- ✅ Confusion matrices

### 5. **Inference**
- ✅ ContractPredictor class
- ✅ Single & batch prediction
- ✅ QA support
- ✅ Probability outputs

### 6. **Deployment**
- ✅ FastAPI REST API server
- ✅ Docker support
- ✅ Docker Compose with monitoring
- ✅ ONNX export capability
- ✅ Health check endpoints

### 7. **Development Tools**
- ✅ CLI interface (main.py)
- ✅ Makefile for automation
- ✅ Setup scripts (Windows/Linux)
- ✅ Jupyter notebooks for tutorials
- ✅ Benchmarking tools
- ✅ Metrics visualization

### 8. **Code Quality**
- ✅ Unit tests with pytest
- ✅ Pre-commit hooks (black, isort, flake8, mypy)
- ✅ GitHub Actions CI/CD
- ✅ Code coverage reporting
- ✅ Type hints
- ✅ Comprehensive documentation

### 9. **Configuration**
- ✅ YAML-based configuration
- ✅ Environment variable override
- ✅ .env file support
- ✅ Flexible and extensible

---

## 🚀 Quick Start

### Setup
```bash
# Windows
.\setup.ps1

# Linux/Mac
chmod +x setup.sh
./setup.sh
```

### Training
```bash
# Quick training
python scripts/quick_train.py

# Full training with config
python main.py train --config configs/config.yaml

# Using Makefile
make train
```

### Inference
```bash
# Quick predict
python scripts/quick_predict.py

# CLI predict
python main.py predict --checkpoint outputs/best_model.pt --text "HỢP ĐỒNG MUA BÁN"
```

### API Server
```bash
# Start server
python scripts/serve_api.py

# With Docker
docker-compose up

# Test endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "HỢP ĐỒNG", "return_probabilities": true}'
```

---

## 📊 Advanced Features

### Benchmarking
```bash
python scripts/benchmark.py --checkpoint outputs/best_model.pt --num-runs 100
```

### ONNX Export
```bash
python scripts/export_onnx.py --checkpoint outputs/best_model.pt --output model.onnx
```

### Visualization
```bash
python scripts/visualize.py --metrics-file outputs/metrics.json
```

### Testing
```bash
# Run all tests
make test

# With coverage
make test-cov
```

---

## 🔧 Technology Stack

### Core Libraries
- **PyTorch** >= 2.0.0
- **Transformers** >= 4.30.0
- **scikit-learn** >= 1.3.0

### API & Deployment
- **FastAPI** >= 0.100.0
- **Uvicorn** >= 0.23.0
- **Docker** & Docker Compose

### Development Tools
- **pytest** - Testing
- **black** - Code formatting
- **flake8** - Linting
- **mypy** - Type checking
- **pre-commit** - Git hooks

### Monitoring
- **TensorBoard** - Training visualization
- **Weights & Biases** - Experiment tracking (optional)

---

## 📖 Documentation

- **README.md** - Main project documentation
- **CHANGELOG.md** - Version history
- **CONTRIBUTING.md** - Contribution guidelines
- **tests/README.md** - Testing guide
- **scripts/README.md** - Scripts documentation
- **Notebooks** - Interactive tutorials

---

## ✨ Best Practices Followed

1. ✅ **Modular Architecture** - Clear separation of concerns
2. ✅ **Type Hints** - For better code quality
3. ✅ **Comprehensive Testing** - Unit tests for all modules
4. ✅ **CI/CD Pipeline** - Automated testing and deployment
5. ✅ **Docker Support** - Containerized deployment
6. ✅ **API First** - REST API for easy integration
7. ✅ **Configuration Management** - YAML + env variables
8. ✅ **Logging & Monitoring** - Comprehensive logging
9. ✅ **Documentation** - README, docstrings, notebooks
10. ✅ **Code Quality Tools** - Linting, formatting, type checking

---

## 🎓 Learning Resources

### Notebooks
1. **01_training_tutorial.ipynb** - Step-by-step training guide
2. **02_data_exploration.ipynb** - Data analysis and visualization

### Example Scripts
- **quick_train.py** - Minimal training example
- **quick_predict.py** - Inference example
- **serve_api.py** - API server example

---

## 🔄 Integration with Parent Project

This transformer-project can be:
1. **Standalone** - Run independently
2. **Integrated** - Import from parent Django project
3. **API Service** - Run as microservice for parent project

Example integration:
```python
# From parent project
import sys
sys.path.append('transformer-project')

from inference.predict import ContractPredictor

predictor = ContractPredictor('transformer-project/outputs/best_model.pt')
result = predictor.predict("HỢP ĐỒNG MUA BÁN")
```

---

## 📝 Summary

**transformer-project** is a production-ready, enterprise-grade PhoBERT implementation following all transformer project best practices:

✅ Complete ML pipeline (data → training → evaluation → deployment)
✅ Modern Python development practices
✅ Comprehensive testing & CI/CD
✅ Docker deployment ready
✅ REST API for easy integration
✅ Extensive documentation
✅ Interactive tutorials

**Ready for:**
- ✅ Development
- ✅ Training
- ✅ Evaluation
- ✅ Deployment
- ✅ Production use
