"""
PhoBERT Classification Training Script
Train PhoBERT for legal document classification (8 legal types)
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import ContractDataset
from utils.logger import setup_logger
from utils.metrics import compute_classification_metrics


# Label mapping
LABEL_MAP = {
    "luat_hinh_su": 0,
    "luat_thuong_mai": 1,
    "luat_hanh_chinh": 2,
    "luat_giao_thong": 3,
    "luat_doanh_nghiep": 4,
    "luat_dat_dai": 5,
    "bo_luat_lao_dong": 6,
    "luat_bat_dong_san": 7
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def load_data(file_path: str):
    """Load and prepare data for classification."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} samples from {file_path}")
    return data


def preprocess_function(examples, tokenizer, max_length=512):
    """Tokenize texts and convert labels."""
    texts = [ex['text'] for ex in examples]
    labels = [LABEL_MAP.get(ex['label'], 0) for ex in examples]
    
    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    encodings['labels'] = torch.tensor(labels)
    return encodings


class ClassificationDataset(torch.utils.data.Dataset):
    """Dataset for classification task."""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = LABEL_MAP.get(item['label'], 0)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train(args):
    """Main training function."""
    
    # Setup logger
    logger = setup_logger('classification_training', args.output_dir)
    logger.info("=" * 60)
    logger.info("PHOBERT CLASSIFICATION TRAINING")
    logger.info("=" * 60)
    
    # Define paths
    data_dir = Path(args.data_dir)
    train_file = data_dir / "train.json"
    val_file = data_dir / "val.json"
    test_file = data_dir / "test.json"
    
    logger.info(f"\n📁 Data paths:")
    logger.info(f"  - Train: {train_file}")
    logger.info(f"  - Val: {val_file}")
    logger.info(f"  - Test: {test_file}")
    
    # Load data
    logger.info(f"\n📊 Loading data...")
    train_data = load_data(train_file)
    val_data = load_data(val_file)
    test_data = load_data(test_file)
    
    # Count labels
    from collections import Counter
    train_labels = Counter([d['label'] for d in train_data])
    logger.info(f"\n📈 Label distribution (train):")
    for label, count in sorted(train_labels.items()):
        logger.info(f"  - {label}: {count} samples")
    
    # Load tokenizer
    logger.info(f"\n🔤 Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    logger.info(f"\n📦 Creating datasets...")
    train_dataset = ClassificationDataset(train_data, tokenizer, args.max_length)
    val_dataset = ClassificationDataset(val_data, tokenizer, args.max_length)
    test_dataset = ClassificationDataset(test_data, tokenizer, args.max_length)
    
    logger.info(f"  - Train: {len(train_dataset)} samples")
    logger.info(f"  - Val: {len(val_dataset)} samples")
    logger.info(f"  - Test: {len(test_dataset)} samples")
    
    # Load model
    logger.info(f"\n🤖 Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_MAP),
        ignore_mismatched_sizes=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        report_to=["tensorboard"],
        seed=42,
    )
    
    logger.info(f"\n⚙️ Training configuration:")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info(f"  - Max length: {args.max_length}")
    logger.info(f"  - FP16: {args.fp16}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    logger.info(f"\n🚀 Starting training...")
    train_result = trainer.train()
    
    # Save model
    logger.info(f"\n💾 Saving model to {args.output_dir}/final_model")
    trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    # Save training metrics
    train_metrics = train_result.metrics
    with open(f"{args.output_dir}/train_metrics.json", 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    logger.info(f"\n📊 Training metrics:")
    for key, value in train_metrics.items():
        logger.info(f"  - {key}: {value:.4f}")
    
    # Evaluate on validation set
    logger.info(f"\n📊 Evaluating on validation set...")
    val_metrics = trainer.evaluate()
    
    with open(f"{args.output_dir}/val_metrics.json", 'w') as f:
        json.dump(val_metrics, f, indent=2)
    
    logger.info(f"\n📊 Validation metrics:")
    for key, value in val_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  - {key}: {value:.4f}")
    
    # Evaluate on test set
    logger.info(f"\n📊 Evaluating on test set...")
    test_metrics = trainer.evaluate(test_dataset)
    
    with open(f"{args.output_dir}/test_metrics.json", 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info(f"\n📊 Test metrics:")
    for key, value in test_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  - {key}: {value:.4f}")
    
    # Detailed classification report
    logger.info(f"\n📋 Generating classification report...")
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Convert to label names
    pred_label_names = [ID_TO_LABEL[i] for i in pred_labels]
    true_label_names = [ID_TO_LABEL[i] for i in true_labels]
    
    report = classification_report(
        true_label_names, 
        pred_label_names, 
        digits=4
    )
    
    logger.info(f"\n📋 Classification Report:\n{report}")
    
    # Save report
    with open(f"{args.output_dir}/classification_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"\n✅ Training complete!")
    logger.info(f"📁 Model saved to: {args.output_dir}/final_model")
    logger.info(f"📊 Metrics saved to: {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Train PhoBERT for classification')
    
    parser.add_argument('--data-dir', type=str, 
                       default='data/processed/classification',
                       help='Directory containing train/val/test.json files')
    parser.add_argument('--model-name', type=str,
                       default='vinai/phobert-base',
                       help='Pretrained model name')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/classification',
                       help='Output directory for model and logs')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                       help='Warmup ratio')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Max sequence length')
    parser.add_argument('--logging-steps', type=int, default=50,
                       help='Logging frequency')
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train
    train(args)


if __name__ == '__main__':
    main()
