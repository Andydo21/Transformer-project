"""
PhoBERT QA Training Script
Train PhoBERT for Question Answering on legal documents
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
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from sklearn.metrics import accuracy_score
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger


def load_data(file_path: str):
    """Load QA data from JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} samples from {file_path}")
    return data


def find_answer_span(answer_text, context):
    """Find start and end positions of answer in context."""
    # Simple string matching
    start_idx = context.find(answer_text)
    if start_idx == -1:
        return 0, 0
    end_idx = start_idx + len(answer_text)
    return start_idx, end_idx


class QADataset(torch.utils.data.Dataset):
    """Dataset for Question Answering task."""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        
        # Use answer as context for now (simplified QA)
        context = answer
        
        # Find answer span in context
        answer_start, answer_end = find_answer_span(question[:50], context)
        
        # Tokenize
        encoding = self.tokenizer(
            question,
            context,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Find token positions for answer
        offset_mapping = encoding['offset_mapping'][0]
        
        start_position = 0
        end_position = 0
        
        for idx, (start, end) in enumerate(offset_mapping):
            if start <= answer_start < end:
                start_position = idx
            if start < answer_end <= end:
                end_position = idx
                break
        
        # Make sure positions are valid
        if end_position < start_position:
            end_position = start_position
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': torch.tensor(start_position, dtype=torch.long),
            'end_positions': torch.tensor(end_position, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Compute QA metrics."""
    start_logits, end_logits = eval_pred.predictions
    start_positions, end_positions = eval_pred.label_ids
    
    # Get predicted positions
    start_preds = np.argmax(start_logits, axis=1)
    end_preds = np.argmax(end_logits, axis=1)
    
    # Compute exact match accuracy
    start_acc = accuracy_score(start_positions, start_preds)
    end_acc = accuracy_score(end_positions, end_preds)
    
    # Exact match: both start and end must be correct
    exact_match = np.mean((start_positions == start_preds) & (end_positions == end_preds))
    
    return {
        'start_accuracy': start_acc,
        'end_accuracy': end_acc,
        'exact_match': exact_match,
    }


def train(args):
    """Main training function."""
    
    # Setup logger
    logger = setup_logger('qa_training', args.output_dir)
    logger.info("=" * 60)
    logger.info("PHOBERT QA TRAINING")
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
    train_dataset = QADataset(train_data, tokenizer, args.max_length)
    val_dataset = QADataset(val_data, tokenizer, args.max_length)
    test_dataset = QADataset(test_data, tokenizer, args.max_length)
    
    logger.info(f"  - Train: {len(train_dataset)} samples")
    logger.info(f"  - Val: {len(val_dataset)} samples")
    logger.info(f"  - Test: {len(test_dataset)} samples")
    
    # Sample question
    if train_data:
        logger.info(f"\n📝 Sample question:")
        logger.info(f"  Q: {train_data[0]['question'][:100]}...")
        logger.info(f"  A: {train_data[0]['answer'][:100]}...")
    
    # Load model
    logger.info(f"\n🤖 Loading model: {args.model_name}")
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name,
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
        metric_for_best_model="exact_match",
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
        data_collator=default_data_collator,
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
    
    # Test inference on a few samples
    logger.info(f"\n🧪 Testing inference on sample questions...")
    model.eval()
    
    for i in range(min(3, len(test_data))):
        sample = test_data[i]
        question = sample['question']
        expected_answer = sample['answer']
        
        # Tokenize
        inputs = tokenizer(
            question,
            expected_answer,
            truncation=True,
            max_length=args.max_length,
            return_tensors='pt'
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs.to(model.device))
        
        # Get answer span
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)
        
        # Decode answer
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        predicted_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        logger.info(f"\n  Sample {i+1}:")
        logger.info(f"    Q: {question[:80]}...")
        logger.info(f"    Expected: {expected_answer[:80]}...")
        logger.info(f"    Predicted: {predicted_answer[:80]}...")
    
    logger.info(f"\n✅ Training complete!")
    logger.info(f"📁 Model saved to: {args.output_dir}/final_model")
    logger.info(f"📊 Metrics saved to: {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Train PhoBERT for QA')
    
    parser.add_argument('--data-dir', type=str,
                       default='data/processed/qa',
                       help='Directory containing train/val/test.json files')
    parser.add_argument('--model-name', type=str,
                       default='vinai/phobert-base',
                       help='Pretrained model name')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/qa',
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
