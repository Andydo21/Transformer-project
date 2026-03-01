# 📝 ViT5 Summarization - Quick Guide

## 🚀 Quick Start

### 1. Generate Training Data
```bash
python scripts/generate_summary_data.py \
  --num-samples 500 \
  --output-dir data/summarization \
  --split
```

### 2. Train ViT5 Model
```bash
python scripts/train_vit5_summarizer.py \
  --train-file data/summarization/train.json \
  --val-file data/summarization/val.json \
  --epochs 10 \
  --batch-size 8 \
  --output-dir outputs/vit5_summarizer
```

### 3. Generate Summaries
```bash
# Single text
python scripts/predict_summary.py \
  --model outputs/vit5_summarizer/final_model \
  --text "Văn bản hợp đồng dài..."

# From file
python scripts/predict_summary.py \
  --model outputs/vit5_summarizer/final_model \
  --input-file data/test.txt \
  --output-file summaries.txt
```

## 📊 Expected Results

```yaml
Training Time: ~2 hours (500 samples, 10 epochs)
Inference Speed: ~2s per document

Metrics (on Vietnamese contracts):
  ROUGE-1: 0.50-0.55
  ROUGE-2: 0.35-0.42
  ROUGE-L: 0.45-0.52
  Compression: 10-15% of original
```

## 💻 Code Example

```python
from models.vit5_summarizer import ViT5ContractSummarizer

# Load model
summarizer = ViT5ContractSummarizer(
    model_name="VietAI/vit5-base"  # or your trained model path
)

# Generate summary
text = """
HỢP ĐỒNG CUNG CẤP DỊCH VỤ
Bên A: Công ty ABC
Bên B: Công ty XYZ
Giá trị: 500 triệu đồng
...
"""

summary = summarizer.generate_summary(
    text,
    max_length=256,
    num_beams=4
)

print(summary)
# Output: "Hợp đồng cung cấp dịch vụ giữa Công ty ABC và XYZ, 
#          giá trị 500 triệu đồng..."
```

## ⚙️ Configuration

```python
# Quality vs Speed tradeoff
summarizer.generate_summary(
    text,
    num_beams=4,           # Higher = better quality, slower
    length_penalty=2.0,     # >1 = longer summaries
    no_repeat_ngram_size=3  # Prevent repetition
)
```

## 📈 Training Tips

1. **Data Quality**: Ensure summaries are concise and informative
2. **Batch Size**: Use 8 for 16GB GPU, 4 for 8GB GPU
3. **Epochs**: 10-15 epochs usually sufficient
4. **Learning Rate**: 3e-5 works well for ViT5
5. **Validation**: Monitor ROUGE scores during training

## 🎯 Use Cases

- **Quick Contract Review**: Generate executive summaries
- **Highlighting Key Info**: Extract parties, value, duration
- **Report Generation**: Create concise reports from contracts
- **Search & Discovery**: Summarize for better indexing

---

**Model:** VietAI/vit5-base  
**Architecture:** T5 (Text-to-Text Transformer)  
**Parameters:** 250M  
**Quality:** ⭐⭐⭐⭐⭐ Excellent
