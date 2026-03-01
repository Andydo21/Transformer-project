# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-02-12

### Added
- Initial release of PhoBERT Contract Processing Project
- Support for 3 task types: Classification, NER, QA
- PhoBERT model integration (vinai/phobert-base)
- Complete training pipeline with HuggingFace Trainer
- Custom training loop implementation
- Multiple loss functions: Cross Entropy, Focal Loss, Label Smoothing
- Comprehensive evaluation metrics
- CLI interface via main.py
- Sample data for testing
- Setup scripts (Windows PowerShell & Linux/Mac Bash)
- Unit tests for core modules
- Documentation in README.md
- Vietnamese language support

### Features
- **Data Processing**
  - Contract dataset loader
  - QA dataset loader  
  - NER dataset with label alignment
  - Data split utilities (train/val/test)

- **Models**
  - ContractClassifier: Contract type classification
  - ContractNER: Named entity recognition
  - ContractQA: Question answering

- **Training**
  - HuggingFace Trainer integration
  - Custom training loop
  - Multiple optimizers and schedulers
  - Gradient clipping
  - Model checkpointing
  - TensorBoard logging

- **Inference**
  - Single text prediction
  - Batch prediction
  - Interactive mode
  - File-based prediction

- **Utilities**
  - PhoBERT tokenizer helpers
  - Comprehensive metrics computation
  - Flexible logging system
  - Configuration management via YAML

### Dependencies
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- scikit-learn >= 1.3.0
- PyYAML >= 6.0
- And more (see requirements.txt)

## Future Plans

### [1.1.0] - Planned
- [ ] Add support for more PhoBERT variants
- [ ] Implement data augmentation techniques
- [ ] Add ensemble methods
- [ ] Improve NER evaluation with seqeval
- [ ] Add Gradio web interface
- [ ] Docker containerization
- [ ] Model compression and quantization

### [1.2.0] - Planned
- [ ] Multi-task learning
- [ ] Few-shot learning support
- [ ] Active learning pipeline
- [ ] API server with FastAPI
- [ ] Model versioning with MLflow
- [ ] Advanced visualization tools
