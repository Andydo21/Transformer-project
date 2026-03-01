"""
Training script for ViT5 Contract Summarization

Fine-tune VietAI/vit5-base on contract summarization data

Usage:
    python scripts/train_vit5_summarizer.py \
        --train-file data/summarization/train.json \
        --val-file data/summarization/val.json \
        --epochs 10 \
        --batch-size 8
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_data(file_path: str):
    """Load summarization data from JSON"""
    print(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} samples")
    return data


def preprocess_function(examples, tokenizer, max_input_length, max_output_length):
    """Preprocess data for ViT5"""
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_input_length,
        truncation=True,
        padding='max_length'
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"],
            max_length=max_output_length,
            truncation=True,
            padding='max_length'
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer):
    """Compute ROUGE metrics"""
    try:
        from datasets import load_metric
        rouge = load_metric("rouge")
    except:
        print("Warning: rouge_score not available")
        return {}
    
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=False
    )
    
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Length metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


def train(args):
    """Main training function"""
    print("=" * 60)
    print("TRAINING ViT5 CONTRACT SUMMARIZATION")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    
    # Load tokenizer and model
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    print(f"✓ Model loaded: {model.num_parameters() / 1e6:.1f}M parameters")
    
    # Load data
    train_data = load_data(args.train_file)
    
    val_data = None
    if args.val_file:
        val_data = load_data(args.val_file)
    
    # Convert to Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None
    
    # Preprocess
    print("\nPreprocessing data...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(
            x,
            tokenizer,
            args.max_input_length,
            args.max_output_length
        ),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    if val_dataset:
        val_dataset = val_dataset.map(
            lambda x: preprocess_function(
                x,
                tokenizer,
                args.max_input_length,
                args.max_output_length
            ),
            batched=True,
            remove_columns=val_dataset.column_names
        )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"✓ Val samples: {len(val_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.1,
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="rouge1" if val_dataset else None,
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=args.max_output_length,
        generation_num_beams=4,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to=["tensorboard"],
        seed=42,
        dataloader_num_workers=0,
        remove_unused_columns=True
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer) if val_dataset else None
    )
    
    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    train_result = trainer.train()
    
    # Save model
    print("\nSaving model...")
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))
    
    # Save metrics
    metrics = train_result.metrics
    metrics_file = output_dir / "train_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Training completed!")
    print(f"  Model saved to: {output_dir / 'final_model'}")
    print(f"  Metrics saved to: {metrics_file}")
    
    # Final evaluation
    if val_dataset:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)
        
        eval_result = trainer.evaluate()
        
        print("\nMetrics:")
        for key, value in eval_result.items():
            print(f"  {key}: {value:.4f}")
        
        # Save eval metrics
        eval_metrics_file = output_dir / "eval_metrics.json"
        with open(eval_metrics_file, 'w', encoding='utf-8') as f:
            json.dump(eval_result, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Train ViT5 Summarization Model")
    
    # Model
    parser.add_argument("--model-name", type=str, default="VietAI/vit5-base",
                       help="ViT5 model name (vit5-base or vit5-large)")
    
    # Data
    parser.add_argument("--train-file", type=str, required=True,
                       help="Training data JSON file")
    parser.add_argument("--val-file", type=str, default=None,
                       help="Validation data JSON file")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size per device")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=3e-5,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")
    
    # Length
    parser.add_argument("--max-input-length", type=int, default=1024,
                       help="Maximum input token length")
    parser.add_argument("--max-output-length", type=int, default=256,
                       help="Maximum output token length")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/vit5_summarizer",
                       help="Output directory")
    
    # Options
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training")
    
    args = parser.parse_args()
    
    # Train
    train(args)


if __name__ == "__main__":
    main()
