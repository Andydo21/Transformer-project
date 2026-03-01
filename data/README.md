# Data Processing Guide

## Overview

Transformer-project hỗ trợ đầy đủ xử lý data cho Vietnamese contract classification.

---

## 📦 Data Components

### 1. **Dataset Classes** (`data/dataset.py`)
- `ContractDataset` - Classification & NER tasks
- `ContractQADataset` - Question Answering tasks
- `load_dataset()` - Load JSON data
- `prepare_data_splits()` - Split train/val/test

### 2. **Data Augmentation** (`data/augmentation.py`)
- `VietnameseTextAugmenter` - Text augmentation cho tiếng Việt
- Synonym replacement
- Random deletion
- Random swap
- Random insertion
- `augment_dataset()` - Augment toàn bộ dataset

### 3. **Data Validation** (`data/validation.py`)
- `DataValidator` - Validate data quality
- Check missing fields
- Check text length
- Check class imbalance
- Detect duplicates
- Auto-fix common issues

### 4. **Data Generation** (`scripts/generate_data.py`)
- Generate synthetic contract samples
- Tạo training data cho testing

---

## 🚀 Usage Examples

### Load Data
```python
from data import load_dataset

# Load JSON file
data = load_dataset('data/sample_data.json')
print(f"Loaded {len(data)} samples")
```

### Validate Data
```python
from data import DataValidator

validator = DataValidator()
result = validator.validate_dataset(data)

if not result['valid']:
    # Auto-fix issues
    cleaned_data, actions = validator.fix_common_issues(data)
    print(f"Fixed: {len(cleaned_data)} samples")
```

### Augment Data
```python
from data import augment_dataset, VietnameseTextAugmenter

# Method 1: Augment entire dataset
augmented_data = augment_dataset(data, num_aug_per_sample=2)
print(f"Original: {len(data)}, Augmented: {len(augmented_data)}")

# Method 2: Augment single text
augmenter = VietnameseTextAugmenter()
aug_texts = augmenter.augment(
    "Hợp đồng mua bán hàng hóa",
    methods=['synonym', 'swap'],
    num_aug=3
)
```

### Generate Sample Data
```bash
# Generate 100 samples
python scripts/generate_data.py --num-samples 100 --output data/samples.json

# Generate and split into train/val/test
python scripts/generate_data.py --num-samples 500 --split
```

### Create Datasets
```python
from data import ContractDataset
from utils import load_tokenizer

tokenizer = load_tokenizer('vinai/phobert-base')

dataset = ContractDataset(
    data=train_data,
    tokenizer=tokenizer,
    task_type='classification',
    max_length=512
)
```

---

## 📋 Data Format

### Classification
```json
[
  {
    "text": "HỢP ĐỒNG MUA BÁN...",
    "label": 0
  },
  {
    "text": "HỢP ĐỒNG THUÊ NHÀ...",
    "label": 1
  }
]
```

### NER
```json
[
  {
    "text": "Công ty ABC ký hợp đồng...",
    "entities": [
      {"text": "Công ty ABC", "start": 0, "end": 11, "label": "ORGANIZATION"}
    ]
  }
]
```

### QA
```json
[
  {
    "context": "Hợp đồng mua bán...",
    "question": "Bên A là ai?",
    "answer": "Công ty ABC"
  }
]
```

---

## 🔧 Data Pipelines

### Complete Training Pipeline

```python
from data import (
    load_dataset,
    DataValidator,
    augment_dataset,
    prepare_data_splits,
    ContractDataset
)
from utils import load_tokenizer

# 1. Load raw data
raw_data = load_dataset('data/raw_data.json')

# 2. Validate
validator = DataValidator()
result = validator.validate_dataset(raw_data)

if not result['valid']:
    raw_data, _ = validator.fix_common_issues(raw_data)

# 3. Augment
augmented_data = augment_dataset(raw_data, num_aug_per_sample=2)

# 4. Split
train_data, val_data, test_data = prepare_data_splits(
    augmented_data,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# 5. Create datasets
tokenizer = load_tokenizer('vinai/phobert-base')

train_dataset = ContractDataset(
    data=train_data,
    tokenizer=tokenizer,
    task_type='classification',
    max_length=512
)
```

---

## 📊 Data Statistics

### Check Distribution
```python
from collections import Counter

labels = [item['label'] for item in data]
distribution = Counter(labels)

for label, count in distribution.items():
    print(f"Label {label}: {count} ({count/len(data)*100:.1f}%)")
```

### Text Length Analysis
```python
lengths = [len(item['text']) for item in data]
print(f"Min: {min(lengths)}")
print(f"Max: {max(lengths)}")
print(f"Mean: {sum(lengths)/len(lengths):.0f}")
```

---

## ⚡ Performance Tips

### 1. **Augmentation Best Practices**
- Use augmentation for small datasets (< 1000 samples)
- Recommended: 2-3 augmented versions per sample
- Methods: synonym + swap works best for Vietnamese

### 2. **Validation**
- Always validate before training
- Fix common issues automatically
- Remove duplicates to avoid overfitting

### 3. **Data Splits**
- Standard: 70% train, 15% val, 15% test
- For small datasets: 80% train, 20% val, no test
- Use stratified splitting for imbalanced data

---

## 🎯 Sample Data

Project includes:
- `data/sample_data.json` - 3 hand-written samples
- `scripts/generate_data.py` - Generate synthetic samples
- Support for loading from multiple sources

Generate more:
```bash
python scripts/generate_data.py --num-samples 1000 --split
```

---

## 📚 Additional Resources

- **Augmentation Guide**: See `data/augmentation.py` docstrings
- **Validation Rules**: See `data/validation.py` for all checks
- **Dataset Implementation**: See `data/dataset.py` for details
- **Interactive Tutorial**: `notebooks/02_data_exploration.ipynb`

---

## ✅ Checklist

Before training, ensure:
- [ ] Data loaded successfully
- [ ] Data validated (no critical issues)
- [ ] Appropriate augmentation applied
- [ ] Data split into train/val/test
- [ ] Dataset classes created
- [ ] Class distribution checked
- [ ] Text lengths within limits

---

**Data processing pipeline is complete and ready for production use!**
