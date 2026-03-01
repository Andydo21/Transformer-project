"""
Example notebook for PhoBERT Contract Processing

This notebook demonstrates:
1. Loading data
2. Training a model
3. Making predictions
4. Evaluating results
"""

# # PhoBERT Contract Processing - Tutorial
# 
# Hướng dẫn sử dụng PhoBERT để xử lý hợp đồng pháp lý tiếng Việt

# ## 1. Setup

# ```python
# # Install dependencies
# !pip install -q torch transformers scikit-learn pyyaml tqdm

# # Import libraries
import sys
sys.path.append('..')

from models.model import load_model, ContractClassifier
from data.dataset import ContractDataset, load_dataset
from inference.predict import ContractPredictor
from utils.tokenizer import load_tokenizer
from utils.metrics import compute_multiclass_metrics
import torch
import yaml
# ```

# ## 2. Load Configuration

# ```python
with open('../configs/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("Configuration:")
print(f"Model: {config['model']['name']}")
print(f"Task: {config['model']['task_type']}")
print(f"Num labels: {config['model']['num_labels']}")
# ```

# ## 3. Prepare Data

# ```python
# Sample contract texts
sample_texts = [
    "HỢP ĐỒNG MUA BÁN\nBên A: Công ty ABC\nBên B: Công ty XYZ\nGiá trị: 100 triệu",
    "HỢP ĐỒNG THUÊ NHÀ\nBên cho thuê: Ông A\nGiá thuê: 10 triệu/tháng", 
    "HỢP ĐỒNG DỊCH VỤ\nCung cấp dịch vụ bảo trì IT\nThời hạn: 12 tháng"
]

labels = [0, 1, 2]  # 0: Mua bán, 1: Thuê, 2: Dịch vụ

print(f"Number of samples: {len(sample_texts)}")
# ```

# ## 4. Load Tokenizer and Model

# ```python
# Load tokenizer
tokenizer = load_tokenizer('vinai/phobert-base')

# Load model (or initialize new one)
model = load_model(
    model_name='vinai/phobert-base',
    task_type='classification',
    num_labels=3,
    device='cpu'
)

print(f"Model loaded: {type(model)}")
# ```

# ## 5. Tokenization Example

# ```python
# Tokenize a sample text
text = sample_texts[0]
encoding = tokenizer(
    text,
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

print(f"Input IDs shape: {encoding['input_ids'].shape}")
print(f"Attention mask shape: {encoding['attention_mask'].shape}")
print(f"\nFirst 20 tokens:")
print(tokenizer.decode(encoding['input_ids'][0][:20]))
# ```

# ## 6. Model Inference

# ```python
# Make prediction
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=encoding['input_ids'],
        attention_mask=encoding['attention_mask']
    )
    
    logits = outputs['logits']
    probs = torch.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)
    
print(f"Predicted label: {pred.item()}")
print(f"Probabilities: {probs[0].tolist()}")
# ```

# ## 7. Batch Prediction

# ```python
# Predict for all samples
predictions = []

for text in sample_texts:
    encoding = tokenizer(
        text,
        add_special_keys=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
        pred = torch.argmax(outputs['logits'], dim=-1)
        predictions.append(pred.item())

print("Predictions:", predictions)
print("True labels:", labels)
# ```

# ## 8. Evaluation

# ```python
import numpy as np

metrics = compute_multiclass_metrics(
    np.array(labels),
    np.array(predictions),
    average='weighted'
)

print("\nEvaluation Metrics:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
# ```

# ## 9. Using ContractPredictor

# ```python
# If you have a trained checkpoint
checkpoint_path = '../outputs/best_model'

# predictor = ContractPredictor(
#     checkpoint_path=checkpoint_path,
#     config_path='../configs/config.yaml'
# )

# result = predictor.predict(sample_texts[0], return_probs=True)
# print(result)
# ```

# ## 10. Training (Optional)

# ```python
# To train the model, use:
# !python ../main.py train --config ../configs/config.yaml
# ```

# ## Next Steps
# 
# 1. Prepare your own contract data
# 2. Adjust configuration in config.yaml
# 3. Train the model with your data
# 4. Evaluate and iterate
# 
# For more details, see README.md
