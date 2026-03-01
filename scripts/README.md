# Scripts Directory

Collection of utility scripts for transformer-project.

## 🚀 Quick Start - Training All Models

### Train Everything (Recommended)
```bash
# Train all 4 models: Classification, NER, QA, Summarization
python scripts/train_all.py --tasks all --fp16

# Train specific models only
python scripts/train_all.py --tasks classification ner --fp16
```

### Individual Model Training

**1. Classification** (PhoBERT - 8 legal types)
```bash
python scripts/train_classification.py --data-dir data/processed/classification --fp16
```

**2. NER** (PhoBERT - Named Entity Recognition)
```bash
python scripts/train_ner.py --data-dir data/processed/ner --fp16
```

**3. QA** (PhoBERT - Question Answering)
```bash
python scripts/train_qa.py --data-dir data/processed/qa --fp16
```

**4. Summarization** (ViT5)
```bash
python scripts/train_vit5_summarizer.py \
    --train-file data/processed/summarization/train.json \
    --val-file data/processed/summarization/val.json --fp16
```

## Available Scripts

### Training & Fine-tuning (✅ NEW)
- **`train_all.py`** ⭐ - Train all 4 models in one command
- **`train_classification.py`** ⭐ - PhoBERT classification training
- **`train_ner.py`** ⭐ - PhoBERT NER training
- **`train_qa.py`** ⭐ - PhoBERT QA training
- **`train_vit5_summarizer.py`** - ViT5 summarization training
- **`finetune_phobert.py`** - General PhoBERT fine-tuning
- **`advanced_finetune.py`** - Advanced training techniques

### Data Processing
- **`organize_crawldata.py`** ⭐ - Reorganize data for training
- **`generate_summary_data.py`** - Generate synthetic summaries
- **`generate_data.py`** - Generate synthetic contract samples

### Inference & Prediction
- **`predict_summary.py`** - Summarization inference
- **`quick_predict.py`** - Quick prediction script
- **`quick_train.py`** - Quick training script

### Model Management
- **`export_onnx.py`** - Export PyTorch model to ONNX format
- **`visualize.py`** - Visualize training metrics and plots
- **`benchmark.py`** - Performance benchmarking tool

### Deployment
- **`serve_api.py`** - FastAPI server for model inference

## Usage Examples

### Data Organization
```bash
# Reorganize crawldata from domain-based to task-based
python scripts/organize_crawldata.py
```

### Training with Custom Settings
# Basic training
python scripts/quick_train.py

# Full fine-tuning with all features
python scripts/finetune_phobert.py \
  --train-file data/train_samples.json \
  --val-file data/val_samples.json \
  --epochs 10 \
  --batch-size 8 \
  --augmentation \
  --use-peft
python scripts/quick_train.py
```

### Export to ONNX
```bash
python scripts/export_onnx.py --checkpoint outputs/best_model.pt --output model.onnx
```

### Generate Sample Data
```bash
# Generate 100 samples
python scripts/generate_data.py --num-samples 100

# Generate and split into train/val/test
python scripts/generate_data.py --num-samples 500 --split
```

### Benchmark Model
```bash
python scripts/benchmark.py --checkpoint outputs/best_model.pt --num-runs 100
```

### Visualize Metrics
```bash
python scripts/visualize.py --metrics-file outputs/metrics.json --output-dir outputs/plots
```

### Start API Server
```bash
python scripts/serve_api.py
# or
uvicorn scripts.serve_api:app --reload --port 8000
```

## API Server Endpoints

Once `serve_api.py` is running:

- **GET** `/` - API information
- **GET** `/health` - Health check
- **POST** `/predict` - Single text prediction
- **POST** `/batch-predict` - Batch prediction
- **POST** `/qa` - Question answering (if model supports)

### Example API Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "HỢP ĐỒNG MUA BÁN số 001/2026", "return_probabilities": true}'
```

## Docker Deployment

```bash
# Build image
docker build -t transformer-contract .

# Run container
docker run -p 8000:8000 -v $(pwd)/outputs:/app/outputs transformer-contract

# Or use docker-compose
docker-compose up
```
