"""
Training script for PhoBERT-based contract processing models
Script training cho các model xử lý hợp đồng sử dụng PhoBERT
"""
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from tqdm import tqdm
import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import load_model, save_model, count_parameters
from data.dataset import load_dataset
from utils.tokenizer import load_tokenizer
from utils.metrics import compute_metrics
from utils.logger import setup_logger
from training.loss import get_loss_function
from training.eval import evaluate


def load_config(config_path: str = 'configs/config.yaml'):
    """Load cấu hình từ file YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_with_hf_trainer(config):
    """
    Training sử dụng HuggingFace Trainer (Khuyên dùng)
    
    Args:
        config: Configuration dictionary
    """
    logger = setup_logger('train', config['logging']['log_file'])
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU TRAINING VỚI HUGGINGFACE TRAINER")
    logger.info("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(config['model']['name'])
    logger.info(f"Tokenizer: {config['model']['name']}")
    
    # Load datasets
    logger.info("Đang load datasets...")
    train_dataset = load_dataset(
        config['data']['train_file'],
        tokenizer,
        config['model']['max_length'],
        config['model']['task_type']
    )
    
    eval_dataset = load_dataset(
        config['data']['val_file'],
        tokenizer,
        config['model']['max_length'],
        config['model']['task_type']
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Load model
    logger.info(f"Đang load model: {config['model']['name']}")
    model = load_model(
        model_name=config['model']['name'],
        task_type=config['model']['task_type'],
        num_labels=config['model']['num_labels'],
        dropout=config['model']['dropout'],
        device=str(device)
    )
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        logging_dir=config['training']['logging_dir'],
        logging_steps=config['training']['logging_steps'],
        save_strategy=config['training'].get('save_strategy', 'steps'),
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        save_total_limit=config['training']['save_total_limit'],
        eval_strategy=config['training']['evaluation_strategy'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        fp16=config['training']['fp16'],
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        remove_unused_columns=False
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda pred: compute_metrics(pred, config['model']['task_type']),
        tokenizer=tokenizer
    )
    
    # Train
    logger.info("Bắt đầu training...")
    trainer.train()
    
    # Evaluate
    logger.info("Đánh giá model...")
    results = trainer.evaluate()
    logger.info(f"Evaluation results: {results}")
    
    # Save final model
    logger.info(f"Lưu model vào {config['training']['output_dir']}")
    trainer.save_model(config['training']['output_dir'])
    tokenizer.save_pretrained(config['training']['output_dir'])
    
    logger.info("=" * 60)
    logger.info("HOÀN THÀNH TRAINING!")
    logger.info("=" * 60)
    
    return results


def train_custom_loop(config):
    """
    Custom training loop với nhiều control hơn
    
    Args:
        config: Configuration dictionary
    """
    logger = setup_logger('train', config['logging']['log_file'])
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU CUSTOM TRAINING LOOP")
    logger.info("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(config['model']['name'])
    
    # Load datasets
    logger.info("Đang load datasets...")
    train_dataset = load_dataset(
        config['data']['train_file'],
        tokenizer,
        config['model']['max_length'],
        config['model']['task_type']
    )
    
    eval_dataset = load_dataset(
        config['data']['val_file'],
        tokenizer,
        config['model']['max_length'],
        config['model']['task_type']
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['per_device_train_batch_size'],
        shuffle=True,
        num_workers=config['training']['dataloader_num_workers']
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['training']['per_device_eval_batch_size'],
        shuffle=False,
        num_workers=config['training']['dataloader_num_workers']
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Eval batches: {len(eval_loader)}")
    
    # Load model
    model = load_model(
        model_name=config['model']['name'],
        task_type=config['model']['task_type'],
        num_labels=config['model']['num_labels'],
        dropout=config['model']['dropout'],
        device=str(device)
    )
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Parameters: {trainable_params:,} / {total_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    total_steps = len(train_loader) * config['training']['num_train_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Loss function
    loss_fn = get_loss_function(
        config['loss']['type'],
        num_classes=config['model']['num_labels'],
        alpha=config['loss'].get('focal_alpha', 0.25),
        gamma=config['loss'].get('focal_gamma', 2.0),
        smoothing=config['loss'].get('smoothing', 0.1)
    )
    
    logger.info(f"Loss function: {config['loss']['type']}")
    
    # Training loop
    best_eval_f1 = 0.0
    global_step = 0
    
    for epoch in range(config['training']['num_train_epochs']):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch + 1}/{config['training']['num_train_epochs']}")
        logger.info(f"{'='*60}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['max_grad_norm']
            )
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Logging
            if global_step % config['training']['logging_steps'] == 0:
                avg_loss = train_loss / train_steps
                logger.info(f"Step {global_step}: loss = {avg_loss:.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Evaluation phase
        logger.info("Đánh giá model...")
        eval_loss, eval_metrics = evaluate(
            model,
            eval_loader,
            device,
            loss_fn,
            config['model']['task_type']
        )
        
        logger.info(f"Evaluation loss: {eval_loss:.4f}")
        logger.info(f"Metrics: {eval_metrics}")
        
        # Save best model
        current_f1 = eval_metrics.get('f1', 0.0)
        if current_f1 > best_eval_f1:
            best_eval_f1 = current_f1
            best_model_path = os.path.join(config['training']['output_dir'], 'best_model')
            save_model(model, best_model_path, tokenizer)
            logger.info(f"✓ Saved best model (F1: {best_eval_f1:.4f})")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"HOÀN THÀNH TRAINING! Best F1: {best_eval_f1:.4f}")
    logger.info("=" * 60)
    
    return {'best_f1': best_eval_f1}


if __name__ == '__main__':
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Chọn phương thức training
    USE_HF_TRAINER = True  # Set False để dùng custom loop
    
    if USE_HF_TRAINER:
        results = train_with_hf_trainer(config)
    else:
        results = train_custom_loop(config)
    
    print("\n✓ Training hoàn tất!")
    print(f"Results: {results}")
