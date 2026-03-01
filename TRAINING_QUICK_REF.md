# 🎯 QUICK REFERENCE - Best Training Techniques

## ⚡ TL;DR - Maximum Accuracy Setup

```bash
# 1. Generate data (if needed)
python scripts/generate_data.py --num-samples 1000 --split

# 2. Train với ALL best techniques
python scripts/advanced_finetune.py \
  --train-file data/train_samples.json \
  --val-file data/val_samples.json \
  --test-file data/test_samples.json \
  --epochs 25 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --label-smoothing \
  --mixup \
  --rdrop

# 3. Monitor
tensorboard --logdir outputs/advanced_training/logs
```

**Expected gain: +15-30% accuracy!**

---

## 📊 Techniques Ranking (by Impact)

### 🥇 Tier 1 - MUST USE (Essential)
| Technique | Gain | Cost | Difficulty |
|-----------|------|------|------------|
| **Warmup + Cosine LR** | +3-5% | Free | Easy |
| **Label Smoothing** | +2-3% | Free | Easy |
| **Layer-wise LR Decay** | +1-2% | Free | Easy |
| **Gradient Clipping** | Stability | Free | Easy |
| **FP16 Mixed Precision** | 2x Speed | Free | Easy |

### 🥈 Tier 2 - HIGHLY RECOMMENDED
| Technique | Gain | Cost | Difficulty |
|-----------|------|------|------------|
| **Data Augmentation** | +3-5% | +100% time | Easy |
| **EMA** | +1-3% | +5% time | Easy |
| **SWA** | +1-2% | +10% time | Medium |
| **Gradient Accumulation** | +1-2% | Free | Easy |
| **Early Stopping** | Prevent overfit | Free | Easy |

### 🥉 Tier 3 - OPTIONAL (Advanced)
| Technique | Gain | Cost | Difficulty |
|-----------|------|------|------------|
| **Focal Loss** | +2-4%* | Free | Easy |
| **Mixup** | +1-2% | +10% time | Medium |
| **R-Drop** | +1-2% | +50% time | Medium |
| **Multi-Sample Dropout** | +0.5-1% | +20% time | Hard |
| **TTA** | +0.5-1.5% | +400% infer | Easy |

*Only for imbalanced datasets

---

## 🚀 Quick Commands

### **Beginner - Basic Best Practices**
```bash
python scripts/finetune_phobert.py \
  --train-file data/train.json \
  --epochs 15 \
  --batch-size 8 \
  --use-peft \
  --augmentation
```
**Time**: ~30 min (1000 samples)  
**Expected**: 75-85% accuracy

### **Intermediate - Recommended Setup**
```bash
python scripts/advanced_finetune.py \
  --train-file data/train.json \
  --val-file data/val.json \
  --epochs 20 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --label-smoothing \
  --fp16
```
**Time**: ~1 hour (1000 samples)  
**Expected**: 85-90% accuracy

### **Advanced - Maximum Accuracy**
```bash
python scripts/advanced_finetune.py \
  --config configs/advanced_config.yaml
```
**Time**: ~2 hours (1000 samples)  
**Expected**: 90-95% accuracy

---

## 📋 Configuration Cheatsheet

### **Small Dataset (< 500 samples)**
```yaml
epochs: 30
batch_size: 8
learning_rate: 3e-5
warmup_ratio: 0.1
label_smoothing: 0.1
use_augmentation: true  # x3-5
use_ema: true
use_swa: true
early_stopping_patience: 3
```

### **Medium Dataset (500-5000 samples)**
```yaml
epochs: 20
batch_size: 16
learning_rate: 2e-5
warmup_ratio: 0.1
label_smoothing: 0.1
use_augmentation: true  # x2
use_ema: true
use_swa: true
early_stopping_patience: 5
```

### **Large Dataset (> 5000 samples)**
```yaml
epochs: 10
batch_size: 32
learning_rate: 2e-5
warmup_ratio: 0.1
label_smoothing: 0.1
use_augmentation: false
use_ema: true
use_swa: false
early_stopping_patience: 3
```

### **Imbalanced Dataset**
```yaml
loss_type: "focal"  # Use Focal Loss
focal_alpha: 0.25
focal_gamma: 2.0
use_augmentation: true  # Especially for minority classes
use_mixup: true
```

---

## 🔧 Hyperparameter Quick Guide

### Learning Rate
```python
Small dataset:  3e-5
Medium dataset: 2e-5
Large dataset:  1e-5 to 2e-5
With PEFT:      1e-4 to 5e-4
```

### Batch Size (effective)
```python
Always aim for: 32
  
8GB GPU:  batch=8,  accum=4
16GB GPU: batch=16, accum=2
24GB GPU: batch=32, accum=1
```

### Epochs
```python
Small:  25-30 epochs
Medium: 15-20 epochs
Large:  5-10 epochs

+ Add 5-10 for early stopping buffer
```

### Other Important Params
```python
warmup_ratio: 0.1 (10% of training)
max_grad_norm: 1.0
weight_decay: 0.01
label_smoothing: 0.1
ema_decay: 0.999
layerwise_lr_decay: 0.95
dropout: 0.1
```

---

## 🎓 Usage by Scenario

### **Scenario 1: Limited Data (< 100 samples)**
```yaml
✅ MUST:
  - Generate synthetic data (500+ samples)
  - Heavy augmentation (x5)
  - Small learning rate (3e-5)
  - Many epochs (30+)
  - PEFT/LoRA
  - Label smoothing
  - EMA + SWA
```

### **Scenario 2: Imbalanced Classes**
```yaml
✅ MUST:
  - Focal Loss
  - Augment minority classes
  - Stratified splits
  - Class weights
  - Mixup
```

### **Scenario 3: Time Constrained**
```yaml
✅ Priority:
  1. Label Smoothing (free)
  2. FP16 (2x faster)
  3. Gradient Accumulation
  4. Skip augmentation
  5. Fewer epochs (10)
```

### **Scenario 4: Maximum Performance**
```yaml
✅ Use ALL:
  - Label Smoothing
  - Warmup + Cosine
  - Layer-wise LR Decay
  - Heavy Augmentation
  - EMA
  - SWA
  - Mixup
  - R-Drop
  - Multi-Sample Dropout
  - TTA at inference
  - Ensemble 5 models
```

---

## 📈 Expected Results

### Baseline (No Techniques)
```
Accuracy: 70-75%
F1 Score: 0.68-0.73
Training: 20 min
```

### With Tier 1 Only
```
Accuracy: 80-85%
F1 Score: 0.78-0.83
Training: 20 min
Gain: +10-12%
```

### With Tier 1 + 2
```
Accuracy: 85-90%
F1 Score: 0.83-0.88
Training: 40 min
Gain: +15-18%
```

### With ALL Techniques
```
Accuracy: 90-95%
F1 Score: 0.88-0.93
Training: 120 min
Gain: +20-25%
```

---

## ⚠️ Common Mistakes to Avoid

```yaml
❌ Don't:
  - Use very high LR (> 5e-5) without warmup
  - Train too long without early stopping
  - Use small batch size (< 8) without gradient accumulation
  - Forget to validate data quality
  - Use data augmentation with large datasets
  - Apply all techniques blindly
  - Ignore train/val gap (overfitting)

✅ Do:
  - Always use warmup + cosine scheduler
  - Monitor validation metrics closely
  - Use gradient clipping
  - Save best model by F1, not loss
  - Set random seed for reproducibility
  - Validate on EMA/SWA model
  - Use TensorBoard
```

---

## 🎯 Action Plan

### **Week 1 - Foundation**
```bash
Day 1-2: Generate/prepare data (1000+ samples)
Day 3-4: Basic training with Tier 1 techniques
Day 5:   Evaluate, analyze errors
Day 6-7: Improve data quality, add augmentation
```

### **Week 2 - Optimization**
```bash
Day 1-2: Add Tier 2 techniques (EMA, SWA)
Day 3-4: Hyperparameter tuning
Day 5:   Try Tier 3 techniques
Day 6-7: Ensemble multiple models, TTA
```

### **Expected Timeline**
```
Hours 0-2:   Setup + data generation
Hours 2-4:   Basic training (Tier 1)
Hours 4-8:   Advanced training (Tier 1+2)
Hours 8-12:  Maximum training (All tiers)
Hours 12-16: Ensemble + TTA
```

---

## 📚 Files Reference

```
📁 transformer-project/
├── training/
│   └── advanced_techniques.py    # All techniques implemented
├── scripts/
│   ├── finetune_phobert.py       # Basic fine-tuning
│   └── advanced_finetune.py      # Advanced fine-tuning
├── configs/
│   ├── config.yaml               # Basic config
│   └── advanced_config.yaml      # Advanced config
└── docs/
    ├── FINETUNING.md             # Basic guide
    ├── ADVANCED_TRAINING.md      # Complete guide
    └── TRAINING_QUICK_REF.md     # This file
```

---

## 💡 Pro Tips

1. **Start Simple**: Begin with Tier 1, add techniques gradually
2. **Monitor Closely**: Watch train/val gap for overfitting
3. **Data First**: Good data > fancy techniques
4. **Be Patient**: Some techniques need more epochs to show effect
5. **Reproduce**: Always set seed for reproducibility
6. **Save Everything**: Save config, metrics, logs with model
7. **Test Incrementally**: Add one technique at a time to measure impact
8. **Use EMA for Val**: Always validate on EMA weights
9. **SWA for Final**: Use SWA model for final test
10. **Ensemble Works**: If possible, ensemble 3-5 models

---

## ✅ Pre-Training Checklist

```yaml
Data:
  ☐ 500+ training samples (or augment to reach)
  ☐ Data validated (no errors)
  ☐ Classes balanced (or use Focal Loss)
  ☐ Proper train/val/test split

Environment:
  ☐ GPU available (recommended)
  ☐ CUDA installed (for FP16)
  ☐ All dependencies installed

Configuration:
  ☐ Config file prepared
  ☐ Hyperparameters set
  ☐ Output directory configured
  ☐ TensorBoard ready

Code:
  ☐ Model loads correctly
  ☐ Data loader works
  ☐ Single batch forward pass OK
  ☐ Loss computes correctly
```

---

## 🚀 START NOW!

```bash
# Quick start with best practices
python scripts/advanced_finetune.py \
  --train-file data/train_samples.json \
  --val-file data/val_samples.json \
  --epochs 20 \
  --batch-size 16 \
  --label-smoothing

# Monitor in real-time
tensorboard --logdir outputs/advanced_training/logs
```

**GO FOR 90%+ ACCURACY! 🎯**
