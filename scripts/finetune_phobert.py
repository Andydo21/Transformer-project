"""
Fine-tuning PhoBERT for Vietnamese Contract Classification
Script fine-tune chuyên biệt cho PhoBERT với contract data
"""
import os
import sys
import json
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from data.dataset import ContractDataset, prepare_data_splits, load_dataset
from models.model import ContractClassifier, save_model
from training.eval import evaluate_classification
from utils.logger import setup_logger
from utils.metrics import compute_metrics
from data.augmentation import augment_dataset
from data.validation import DataValidator
import argparse


def setup_peft_model(model, task_type: str = "classification"):
    """
    Setup PEFT (LoRA) for efficient fine-tuning
    
    Args:
        model: Base model
        task_type: Task type
    
    Returns:
        PEFT model
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        logger = setup_logger('peft')
        logger.info("Setting up LoRA for efficient fine-tuning...")
        
        # LoRA configuration
        if task_type == "classification":
            task = TaskType.SEQ_CLS
        elif task_type == "ner":
            task = TaskType.TOKEN_CLS
        else:
            task = TaskType.SEQ_2_SEQ_LM
        
        lora_config = LoraConfig(
            task_type=task,
            r=8,  # LoRA rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value"],  # PhoBERT attention layers
            bias="none"
        )
        
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logger.info(f"Total params: {total_params:,}")
        
        return model
    
    except ImportError:
        logger = setup_logger('peft')
        logger.warning("PEFT not installed. Install with: pip install peft")
        logger.warning("Continuing without LoRA (full fine-tuning)...")
        return model


def finetune_phobert(
    train_file: str,
    val_file: str = None,
    test_file: str = None,
    output_dir: str = 'outputs/finetuned_phobert',
    model_name: str = 'vinai/phobert-base',
    task_type: str = 'classification',
    num_labels: int = 3,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    use_peft: bool = False,
    use_augmentation: bool = False,
    validate_data: bool = True,
    device: str = None
):
    """
    Fine-tune PhoBERT cho Vietnamese contract classification
    
    Args:
        train_file: Path to training data JSON
        val_file: Path to validation data JSON
        test_file: Path to test data JSON
        output_dir: Output directory
        model_name: PhoBERT model name
        task_type: Task type
        num_labels: Number of labels
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_length: Max sequence length
        use_peft: Use PEFT (LoRA) for efficient training
        use_augmentation: Apply data augmentation
        validate_data: Validate data before training
        device: Device to use
    """
    logger = setup_logger('finetune')
    
    logger.info("=" * 80)
    logger.info("FINE-TUNING PHOBERT FOR VIETNAMESE CONTRACT CLASSIFICATION")
    logger.info("=" * 80)
    
    # Device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Load data
    logger.info(f"\nLoading training data from {train_file}...")
    train_data = load_dataset(train_file)
    logger.info(f"Loaded {len(train_data)} training samples")
    
    # Validate data
    if validate_data:
        logger.info("\nValidating data...")
        validator = DataValidator()
        result = validator.validate_dataset(train_data)
        
        if not result['valid']:
            logger.warning("Data validation found issues. Attempting to fix...")
            train_data, actions = validator.fix_common_issues(train_data)
            logger.info(f"Fixed data: {len(train_data)} samples")
    
    # Data augmentation
    if use_augmentation:
        logger.info("\nApplying data augmentation...")
        original_size = len(train_data)
        train_data = augment_dataset(train_data, num_aug_per_sample=2)
        logger.info(f"Augmented: {original_size} → {len(train_data)} samples")
    
    # Load validation data
    if val_file:
        logger.info(f"\nLoading validation data from {val_file}...")
        val_data = load_dataset(val_file)
        logger.info(f"Loaded {len(val_data)} validation samples")
    else:
        logger.info("\nNo validation file provided, splitting training data...")
        train_data, val_data, _ = prepare_data_splits(
            train_data,
            train_ratio=0.8,
            val_ratio=0.2,
            test_ratio=0.0
        )
        logger.info(f"Split: {len(train_data)} train, {len(val_data)} val")
    
    # Load tokenizer
    logger.info(f"\nLoading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = ContractDataset(
        data=train_data,
        tokenizer=tokenizer,
        task_type=task_type,
        max_length=max_length
    )
    
    val_dataset = ContractDataset(
        data=val_data,
        tokenizer=tokenizer,
        task_type=task_type,
        max_length=max_length
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Load model
    logger.info(f"\nLoading model: {model_name}...")
    model = ContractClassifier(
        model_name=model_name,
        num_labels=num_labels,
        dropout=0.1
    )
    model.to(device)
    
    # Apply PEFT if requested
    if use_peft:
        model = setup_peft_model(model, task_type)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(output_dir, 'logs'),
        report_to="tensorboard",
        remove_unused_columns=False,
    )
    
    # Trainer
    logger.info("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    train_result = trainer.train()
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Training time: {train_result.metrics['train_runtime']:.2f}s")
    logger.info(f"Training loss: {train_result.metrics['train_loss']:.4f}")
    
    # Evaluate
    logger.info("\nEvaluating on validation set...")
    eval_metrics = trainer.evaluate()
    
    logger.info("\nValidation Results:")
    for key, value in eval_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Test evaluation
    if test_file:
        logger.info(f"\nEvaluating on test set from {test_file}...")
        test_data = load_dataset(test_file)
        test_dataset = ContractDataset(
            data=test_data,
            tokenizer=tokenizer,
            task_type=task_type,
            max_length=max_length
        )
        
        test_metrics = evaluate_classification(
            model=model,
            dataset=test_dataset,
            batch_size=batch_size,
            device=device
        )
        
        logger.info("\nTest Results:")
        for key, value in test_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
    
    # Save model
    logger.info(f"\nSaving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save with trainer (includes tokenizer)
    trainer.save_model(output_dir)
    
    # Also save in our format
    model_path = os.path.join(output_dir, 'best_model.pt')
    save_model(model, model_path)
    
    logger.info("✅ Model saved successfully!")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    all_metrics = {
        'train': train_result.metrics,
        'val': eval_metrics
    }
    if test_file:
        all_metrics['test'] = test_metrics
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Metrics saved to {metrics_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("FINE-TUNING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    
    return trainer, all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune PhoBERT')
    parser.add_argument('--train-file', type=str, required=True, help='Training data JSON')
    parser.add_argument('--val-file', type=str, help='Validation data JSON')
    parser.add_argument('--test-file', type=str, help='Test data JSON')
    parser.add_argument('--output-dir', type=str, default='outputs/finetuned_phobert')
    parser.add_argument('--model-name', type=str, default='vinai/phobert-base')
    parser.add_argument('--num-labels', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--use-peft', action='store_true', help='Use LoRA (PEFT)')
    parser.add_argument('--augmentation', action='store_true', help='Use data augmentation')
    parser.add_argument('--no-validation', action='store_true', help='Skip data validation')
    
    args = parser.parse_args()
    
    finetune_phobert(
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_labels=args.num_labels,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        use_peft=args.use_peft,
        use_augmentation=args.augmentation,
        validate_data=not args.no_validation
    )
