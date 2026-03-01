"""
PhoBERT NER Training Script
Train PhoBERT for Named Entity Recognition on legal documents
Entities: LAW, ARTICLE, ORGANIZATION, PERSON, DATE, LOCATION
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
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger


# NER label mapping
NER_LABELS = [
    "O",          # Outside
    "B-LAW",      # Begin Law
    "I-LAW",      # Inside Law
    "B-ARTICLE",  # Begin Article
    "I-ARTICLE",  # Inside Article
    "B-ORG",      # Begin Organization
    "I-ORG",      # Inside Organization
    "B-PER",      # Begin Person
    "I-PER",      # Inside Person
    "B-DATE",     # Begin Date
    "I-DATE",     # Inside Date
    "B-LOC",      # Begin Location
    "I-LOC",      # Inside Location
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(NER_LABELS)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(NER_LABELS)}


def load_data(file_path: str):
    """Load NER data from JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} samples from {file_path}")
    return data


def align_labels_with_tokens(labels, word_ids):
    """Align labels with tokenized words."""
    new_labels = []
    current_word = None
    
    for word_id in word_ids:
        if word_id is None:
            # Special token
            new_labels.append(-100)
        elif word_id != current_word:
            # Start of new word
            current_word = word_id
            label = labels[word_id] if word_id < len(labels) else 0
            new_labels.append(label)
        else:
            # Continuation of word (subword)
            label = labels[word_id] if word_id < len(labels) else 0
            # If it's a B- tag, change to I- tag
            if label > 0 and ID_TO_LABEL[label].startswith('B-'):
                i_label = 'I-' + ID_TO_LABEL[label][2:]
                new_labels.append(LABEL_TO_ID[i_label])
            else:
                new_labels.append(label)
    
    return new_labels


class NERDataset(torch.utils.data.Dataset):
    """Dataset for NER task."""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        entities = item.get('entities', [])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Create labels (all O initially)
        labels = [0] * len(text)  # Character-level labels
        
        # Fill in entity labels
        for entity in entities:
            start = entity.get('start', 0)
            end = entity.get('end', start)
            label_type = entity.get('label', 'O')
            
            if label_type != 'O':
                # Mark first character as B-
                if start < len(labels):
                    labels[start] = LABEL_TO_ID.get(f"B-{label_type}", 0)
                
                # Mark remaining characters as I-
                for i in range(start + 1, min(end, len(labels))):
                    labels[i] = LABEL_TO_ID.get(f"I-{label_type}", 0)
        
        # Align labels with tokens
        offset_mapping = encoding['offset_mapping'][0]
        word_ids = encoding.word_ids(0)
        
        # Convert character-level labels to token-level
        token_labels = []
        for idx, (start, end) in enumerate(offset_mapping):
            if start == end == 0:
                # Special token
                token_labels.append(-100)
            else:
                # Use label from first character of token
                token_labels.append(labels[start] if start < len(labels) else 0)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(token_labels, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Compute NER metrics using seqeval."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Convert to label names
    true_labels = []
    pred_labels = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_labels = []
        
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:  # Ignore padding
                true_seq.append(ID_TO_LABEL[label])
                pred_seq_labels.append(ID_TO_LABEL[pred])
        
        if true_seq:  # Only add non-empty sequences
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_labels)
    
    # Compute metrics
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def train(args):
    """Main training function."""
    
    # Setup logger
    logger = setup_logger('ner_training', args.output_dir)
    logger.info("=" * 60)
    logger.info("PHOBERT NER TRAINING")
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
    
    # Load tokenizer
    logger.info(f"\n🔤 Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    logger.info(f"\n📦 Creating datasets...")
    train_dataset = NERDataset(train_data, tokenizer, args.max_length)
    val_dataset = NERDataset(val_data, tokenizer, args.max_length)
    test_dataset = NERDataset(test_data, tokenizer, args.max_length)
    
    logger.info(f"  - Train: {len(train_dataset)} samples")
    logger.info(f"  - Val: {len(val_dataset)} samples")
    logger.info(f"  - Test: {len(test_dataset)} samples")
    
    logger.info(f"\n🏷️  NER labels ({len(NER_LABELS)}):")
    for label in NER_LABELS[:5]:
        logger.info(f"  - {label}")
    logger.info(f"  ... and {len(NER_LABELS) - 5} more")
    
    # Load model
    logger.info(f"\n🤖 Loading model: {args.model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(NER_LABELS),
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
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    logger.info(f"\n🚀 Starting training...")
    train_result = trainer.train()
    
    # Save model
    logger.info(f"\n💾 Saving model to {args.output_dir}/final_model")
    trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    # Save label mapping
    label_map_file = f"{args.output_dir}/final_model/label_map.json"
    with open(label_map_file, 'w', encoding='utf-8') as f:
        json.dump({
            'labels': NER_LABELS,
            'label_to_id': LABEL_TO_ID,
            'id_to_label': ID_TO_LABEL
        }, f, indent=2, ensure_ascii=False)
    
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
    logger.info(f"\n📋 Generating NER classification report...")
    predictions = trainer.predict(test_dataset)
    pred_labels_ids = np.argmax(predictions.predictions, axis=2)
    true_labels_ids = predictions.label_ids
    
    # Convert to label names
    true_labels = []
    pred_labels = []
    
    for pred_seq, label_seq in zip(pred_labels_ids, true_labels_ids):
        true_seq = []
        pred_seq_labels = []
        
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                true_seq.append(ID_TO_LABEL[label])
                pred_seq_labels.append(ID_TO_LABEL[pred])
        
        if true_seq:
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_labels)
    
    report = classification_report(true_labels, pred_labels, digits=4)
    
    logger.info(f"\n📋 NER Classification Report:\n{report}")
    
    # Save report
    with open(f"{args.output_dir}/ner_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"\n✅ Training complete!")
    logger.info(f"📁 Model saved to: {args.output_dir}/final_model")
    logger.info(f"📊 Metrics saved to: {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Train PhoBERT for NER')
    
    parser.add_argument('--data-dir', type=str,
                       default='data/processed/ner',
                       help='Directory containing train/val/test.json files')
    parser.add_argument('--model-name', type=str,
                       default='vinai/phobert-base',
                       help='Pretrained model name')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/ner',
                       help='Output directory for model and logs')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-5,
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
