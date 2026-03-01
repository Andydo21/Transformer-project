# Fine-tuning Guide

## PhoBERT Fine-tuning for Vietnamese Contract Classification

Complete guide để fine-tune PhoBERT cho Vietnamese contract tasks.

---

## 🎯 Overview

Transformer-project cung cấp comprehensive fine-tuning pipeline:

✅ **Full fine-tuning** - Train toàn bộ model  
✅ **PEFT/LoRA** - Efficient fine-tuning với ít parameters hơn  
✅ **Data augmentation** - Tăng cường data tự động  
✅ **Data validation** - Validate và fix data issues  
✅ **Early stopping** - Tránh overfitting  
✅ **Best model selection** - Tự động save best checkpoint  

---

## 🚀 Quick Start

### Basic Fine-tuning

```bash
python scripts/finetune_phobert.py \
  --train-file data/train_samples.json \
  --val-file data/val_samples.json \
  --output-dir outputs/finetuned \
  --epochs 10 \
  --batch-size 8
```

### With All Features

```bash
python scripts/finetune_phobert.py \
  --train-file data/train_samples.json \
  --val-file data/val_samples.json \
  --test-file data/test_samples.json \
  --output-dir outputs/phobert_finetuned \
  --model-name vinai/phobert-base \
  --num-labels 3 \
  --epochs 20 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --max-length 512 \
  --use-peft \
  --augmentation
```

---

## 📋 Arguments

### Required
- `--train-file` - Training data JSON file
- `--val-file` - Validation data JSON file (optional, will split from train if not provided)

### Model
- `--model-name` - PhoBERT model (default: `vinai/phobert-base`)
- `--num-labels` - Number of labels (default: 3)

### Training
- `--epochs` - Number of epochs (default: 10)
- `--batch-size` - Batch size (default: 8)
- `--learning-rate` - Learning rate (default: 2e-5)
- `--max-length` - Max sequence length (default: 512)

### Advanced
- `--use-peft` - Use LoRA (PEFT) for efficient training
- `--augmentation` - Apply data augmentation
- `--no-validation` - Skip data validation
- `--output-dir` - Output directory (default: `outputs/finetuned_phobert`)
- `--test-file` - Test data for final evaluation

---

## 🔧 Fine-tuning Methods

### 1. Full Fine-tuning

Train toàn bộ PhoBERT parameters:

```bash
python scripts/finetune_phobert.py \
  --train-file data/train.json \
  --epochs 10
```

**Pros**: Tối ưu performance  
**Cons**: Cần nhiều VRAM, train chậm hơn  

### 2. PEFT/LoRA Fine-tuning

Chỉ train một phần parameters với LoRA:

```bash
python scripts/finetune_phobert.py \
  --train-file data/train.json \
  --use-peft \
  --epochs 10
```

**Pros**: 
- Nhanh hơn 2-3x
- Ít VRAM hơn ~40%
- Trainable params chỉ ~1-5% total
- Performance gần tương đương full fine-tuning

**Cons**: Cần cài `peft` package

```bash
pip install peft
```

---

## 📊 Data Augmentation

Auto augment data để tăng training samples:

```bash
python scripts/finetune_phobert.py \
  --train-file data/train.json \
  --augmentation
```

**Augmentation methods:**
- Synonym replacement (dùng từ đồng nghĩa)
- Random deletion (xóa từ ngẫu nhiên)
- Random swap (hoán đổi vị trí từ)
- Random insertion (chèn từ đồng nghĩa)

**Effect**: Tăng 2-3x số samples, giảm overfitting

---

## ✅ Data Validation

Script tự động validate data trước training:

- Check missing fields
- Check empty texts
- Check text length (min/max)
- Check class imbalance
- Detect duplicates
- Auto-fix common issues

Disable validation:
```bash
python scripts/finetune_phobert.py \
  --train-file data/train.json \
  --no-validation
```

---

## 📈 Training Process

### 1. Data Loading & Validation
```
Loading training data from data/train.json...
Loaded 500 samples
Validating data...
✓ All checks passed!
```

### 2. Data Augmentation (if enabled)
```
Applying data augmentation...
Augmented: 500 → 1500 samples
```

### 3. Model Loading
```
Loading model: vinai/phobert-base...
Loading tokenizer...
Creating datasets...
Train dataset: 1500 samples
Val dataset: 200 samples
```

### 4. Training
```
Epoch 1/10
Train loss: 0.8523
Eval loss: 0.7123
Eval F1: 0.7845

Epoch 2/10
Train loss: 0.6234
Eval loss: 0.5987
Eval F1: 0.8234
...
```

### 5. Evaluation
```
Validation Results:
  eval_loss: 0.4321
  eval_accuracy: 0.8923
  eval_precision: 0.8756
  eval_recall: 0.8845
  eval_f1: 0.8800
```

### 6. Model Saving
```
Saving model to outputs/finetuned_phobert...
✓ Model saved successfully!
✓ Metrics saved to outputs/finetuned_phobert/metrics.json
```

---

## 🎓 Best Practices

### Data Preparation
1. **Validate data first**:
   ```python
   from data import DataValidator, validate_json_file
   validate_json_file('data/train.json')
   ```

2. **Check label distribution**: Đảm bảo balanced dataset

3. **Generate more data if needed**:
   ```bash
   python scripts/generate_data.py --num-samples 1000 --split
   ```

### Hyperparameters

**Small dataset (< 500 samples):**
- Epochs: 15-20
- Batch size: 4-8
- Learning rate: 3e-5
- Use augmentation: ✅
- Use PEFT: ✅ (tránh overfit)

**Medium dataset (500-5000 samples):**
- Epochs: 10-15
- Batch size: 8-16
- Learning rate: 2e-5
- Use augmentation: Optional
- Use PEFT: ✅ (faster)

**Large dataset (> 5000 samples):**
- Epochs: 5-10
- Batch size: 16-32
- Learning rate: 1e-5 to 2e-5
- Use augmentation: ❌
- Use PEFT: Optional

### GPU Memory

**8GB GPU:**
- Batch size: 4-8
- Max length: 256-384
- Use PEFT: ✅
- FP16: ✅

**16GB+ GPU:**
- Batch size: 16-32
- Max length: 512
- Full fine-tuning: ✅
- FP16: Optional

---

## 📊 Monitoring

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir outputs/finetuned_phobert/logs

# Open in browser
http://localhost:6006
```

### Weights & Biases (optional)

1. Install: `pip install wandb`
2. Login: `wandb login`
3. Script tự động log lên W&B

---

## 🧪 Testing

After training, test on held-out test set:

```bash
python scripts/finetune_phobert.py \
  --train-file data/train.json \
  --val-file data/val.json \
  --test-file data/test.json
```

Test results will be saved to `outputs/finetuned_phobert/metrics.json`.

---

## 💾 Output Structure

```
outputs/finetuned_phobert/
├── best_model.pt            # Model trong format của project
├── pytorch_model.bin        # HuggingFace model
├── config.json              # Model config
├── tokenizer_config.json    # Tokenizer config
├── vocab.txt                # Vocabulary
├── metrics.json             # Train/Val/Test metrics
├── logs/                    # TensorBoard logs
└── checkpoint-*/            # Training checkpoints
```

---

## 🔄 Resume Training

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Resume from checkpoint
trainer.train(resume_from_checkpoint='outputs/finetuned_phobert/checkpoint-500')
```

---

## 🚀 Using Fine-tuned Model

```python
from inference import ContractPredictor

# Load fine-tuned model
predictor = ContractPredictor('outputs/finetuned_phobert/best_model.pt')

# Predict
result = predictor.predict("HỢP ĐỒNG MUA BÁN hàng hóa")
print(result)
# {'label': 0, 'confidence': 0.9523}
```

---

## ❓ Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 4`
- Reduce max length: `--max-length 256`
- Use PEFT: `--use-peft`
- Enable gradient checkpointing (in code)

### Underfitting
- Increase epochs
- Increase learning rate
- Use less augmentation
- Check if data is too easy/clean

### Overfitting
- Add augmentation: `--augmentation`
- Use PEFT: `--use-peft`
- Add dropout
- Use early stopping (automatic)
- Get more data

### Poor Performance
- Check data quality with validation
- Balance classes
- Increase training data
- Try different learning rates
- Use data augmentation

---

**Fine-tuning pipeline is production-ready! 🎉**
