# PhoBERT + ViT5 Contract Processing Project

Dự án xử lý hợp đồng pháp lý tiếng Việt sử dụng:
- **PhoBERT** (Vietnamese BERT) - Classification, NER, QA
- **ViT5** (Vietnamese T5) - Summarization

## 📁 Cấu trúc Project

```
transformer-project/
│
├── configs/
│   └── config.yaml         # Cấu hình model, training, data
│
├── data/
│   ├── raw/                # Dữ liệu gốc
│   ├── processed/          # Dữ liệu đã tokenize (train/val/test)
│   └── dataset.py          # Dataset classes
│
├── models/
│   ├── __init__.py
│   ├── model.py            # Model definitions (Classification, NER, QA)
│   └── vit5_summarizer.py  # ViT5 Summarization
│
├── training/
│   ├── train.py            # Training script
│   ├── eval.py             # Evaluation utilities
│   └── loss.py             # Loss functions
│
├── inference/
│   └── predict.py          # Inference/prediction
│
├── utils/
│   ├── tokenizer.py        # Tokenizer utilities
│   ├── metrics.py          # Metrics computation
│   └── logger.py           # Logging utilities
│
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🚀 Cài đặt

### 1. Tạo môi trường ảo

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# hoặc
source venv/bin/activate  # Linux/Mac
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Tải PhoBERT model (tự động khi chạy lần đầu)

Model sẽ được tự động tải từ HuggingFace khi chạy lần đầu.

## 📊 Chuẩn bị dữ liệu

### Format dữ liệu

**Phân loại hợp đồng:**
```json
[
  {
    "text": "Nội dung hợp đồng...",
    "label": 0
  }
]
```

**Question Answering:**
```json
[
  {
    "contract_text": "Nội dung hợp đồng đầy đủ...",
    "questions": [
      {
        "question": "Bên A là ai?",
        "answer": "Công ty TNHH ABC",
        "answer_start": 115
      }
    ]
  }
]
```

### Chia dữ liệu train/val/test

```bash
python main.py prepare-data \
    --input-file data/raw/contracts.json \
    --output-dir data/processed \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

## 🎯 Training

### Sử dụng HuggingFace Trainer (Khuyên dùng)

```bash
python main.py train --config configs/config.yaml
```

### Sử dụng Custom Training Loop

```bash
python main.py train --config configs/config.yaml --use-custom-loop
```

### Cấu hình Training

Chỉnh sửa `configs/config.yaml`:

```yaml
model:
  name: "vinai/phobert-base"
  task_type: "classification"  # classification, ner, qa
  num_labels: 3

training:
  num_train_epochs: 5
  per_device_train_batch_size: 8
  learning_rate: 2.0e-5
```

## 🔮 Inference/Prediction

### Interactive Mode

```bash
python main.py predict --checkpoint outputs/best_model
```

### Từ file

```bash
python main.py predict \
    --checkpoint outputs/best_model \
    --input-file data/test.json \
    --output-file predictions.json
```

### Sử dụng trong code

```python
from inference.predict import ContractPredictor

# Load predictor
predictor = ContractPredictor(
    checkpoint_path='outputs/best_model',
    config_path='configs/config.yaml'
)

# Predict
text = "Nội dung hợp đồng..."
result = predictor.predict(text, return_probs=True)
print(result)
```

## 📈 Các Task được hỗ trợ

### 1. Phân loại hợp đồng (Classification)
- Phân loại các loại hợp đồng: mua bán, thuê, dịch vụ, v.v.
- Model: `ContractClassifier`

### 2. Trích xuất thông tin (NER)
- Trích xuất các thực thể: Bên A, Bên B, giá trị, thời hạn
- Model: `ContractNER`

### 3. Hỏi đáp (Question Answering)
- Trả lời câu hỏi dựa trên nội dung hợp đồng
- Model: `ContractQA`

## 🛠 Các Loss Functions

- **Cross Entropy**: Standard loss cho classification
- **Focal Loss**: Xử lý class imbalance
- **Label Smoothing**: Prevent overconfidence
- **Weighted CE**: Class-weighted cross entropy

Chọn loss function trong `configs/config.yaml`:

```yaml
loss:
  type: "focal"  # cross_entropy, focal, label_smoothing
  focal_alpha: 0.25
  focal_gamma: 2.0
```

## 📊 Metrics

- **Classification**: Accuracy, Precision, Recall, F1
- **NER**: Token-level accuracy, Precision, Recall, F1
- **QA**: Exact Match, F1 score

## 🔧 Tùy chỉnh Model

### Thêm model mới

Tạo class mới trong `models/model.py`:

```python
class CustomModel(nn.Module):
    def __init__(self, model_name, ...):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        # Custom layers...
    
    def forward(self, input_ids, attention_mask, ...):
        # Custom forward pass
        pass
```

## 📝 Logs & Checkpoints

- **Logs**: `logs/training.log`
- **Checkpoints**: `outputs/checkpoint-*/`
- **Best Model**: `outputs/best_model/`
- **TensorBoard**: `runs/`

Xem logs với TensorBoard:
```bash
tensorboard --logdir=runs
```

## 🎓 Ví dụ sử dụng

### Training model phân loại

```bash
# 1. Chuẩn bị dữ liệu
python main.py prepare-data --input-file data/raw/contracts.json

# 2. Training
python main.py train

# 3. Prediction
python main.py predict --checkpoint outputs/best_model
```

### Question Answering

```python
from inference.predict import ContractPredictor

predictor = ContractPredictor('outputs/best_model')

result = predictor.predict_qa(
    question="Bên A là ai?",
    context="Hợp đồng giữa Bên A: Công ty ABC..."
)
print(f"Câu trả lời: {result['answer']}")
```

## ✨ Features

- ✅ PhoBERT-based Vietnamese contract processing
- ✅ Three task types: Classification, NER, Question Answering
- ✅ Complete training pipeline with HuggingFace Trainer
- ✅ **Advanced training techniques for maximum accuracy (90%+)**
- ✅ Custom loss functions (Focal, Label Smoothing, Dice)
- ✅ **EMA, SWA, Mixup, R-Drop, Multi-Sample Dropout, and more**
- ✅ Data augmentation for Vietnamese text
- ✅ Data validation and quality checking
- ✅ PEFT/LoRA for efficient fine-tuning (40% less memory, 2-3x faster)
- ✅ Inference API with FastAPI
- ✅ Docker deployment support
- ✅ Comprehensive documentation and examples

## 📚 Documentation

- 📘 [Project Structure](PROJECT_STRUCTURE.md) - Complete project overview
- 📗 [Fine-tuning Guide](FINETUNING.md) - Basic fine-tuning guide  
- 📕 [Advanced Training](ADVANCED_TRAINING.md) - **15+ techniques for 90%+ accuracy**
- 📙 [Quick Reference](TRAINING_QUICK_REF.md) - **TL;DR training tips and cheatsheet**
- 📓 [Data Guide](data/README.md) - Data processing and augmentation
- 📔 [Scripts Guide](scripts/README.md) - Available scripts and usage
- 📖 [Data & Fine-tuning Summary](DATA_FINETUNE_SUMMARY.md) - Complete summary

## 🎯 Quick Start for Maximum Accuracy

```bash
# 1. Generate training data
python scripts/generate_data.py --num-samples 1000 --split

# 2. Train with advanced techniques
python scripts/advanced_finetune.py \
  --train-file data/train_samples.json \
  --val-file data/val_samples.json \
  --epochs 25 \
  --batch-size 16 \
  --label-smoothing \
  --mixup

# 3. Monitor training
tensorboard --logdir outputs/advanced_training/logs
```

**Expected: 90%+ accuracy with advanced techniques!**

See [TRAINING_QUICK_REF.md](TRAINING_QUICK_REF.md) for complete guide.

## 🤝 Contributing

Đóng góp cho project:
1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📄 License

MIT License

## 📧 Contact

- Email: your-email@example.com
- GitHub: https://github.com/yourusername

## 🙏 Acknowledgments

- [PhoBERT](https://github.com/VinAIResearch/PhoBERT) - Vietnamese BERT model
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- Contract data từ project gốc

## 🔗 References

- PhoBERT Paper: [PhoBERT: Pre-trained language models for Vietnamese](https://arxiv.org/abs/2003.00744)
- BERT Paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

---

**Lưu ý**: Đây là project template. Cần chuẩn bị dữ liệu training phù hợp với domain cụ thể của bạn.
