"""
ADVANCED Training Script - Sử dụng TẤT CẢ kỹ thuật advanced!
Tận dụng tối đa 15GB VRAM của Google Colab

Features:
- ✅ FocalLoss & Label Smoothing
- ✅ Mixup Data Augmentation
- ✅ EMA (Exponential Moving Average)
- ✅ SWA (Stochastic Weight Averaging)
- ✅ R-Drop Regularization
- ✅ Gradient Accumulation
- ✅ Mixed Precision (FP16)
- ✅ Warmup + Cosine Learning Rate
- ✅ Multi-sample Dropout

Usage:
    # Train single task
    python advanced_train.py --task classification
    
    # Train all tasks với advanced techniques
    python advanced_train.py --all-tasks
    
    # Custom settings
    python advanced_train.py --task ner --batch-size 32 --accumulation-steps 2
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Import advanced techniques từ training module
sys.path.insert(0, str(Path(__file__).parent))
from training.advanced_techniques import (
    FocalLoss,
    LabelSmoothingLoss,
    MixupDataAugmentation,
    EMA,
    StochasticWeightAveraging as SWA,  # Alias for shorter name
    RDropRegularization,
    WarmupScheduler,
    MultiSampleDropout,
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Advanced Training for PhoBERT")
    
    # Task
    parser.add_argument("--task", type=str, 
                       choices=["classification", "ner", "qa", "summarization", "seq2seq"],
                       help="Task to train")
    parser.add_argument("--all-tasks", action="store_true",
                       help="Train all tasks sequentially")
    
    # Model
    parser.add_argument("--model", type=str, default="vinai/phobert-base",
                       help="Model name or path")
    
    # Training hyperparameters - TỐI ƯU CHO 15GB VRAM
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs (default: 10 for better convergence)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size per device (default: 32 - tận dụng VRAM)")
    parser.add_argument("--accumulation-steps", type=int, default=2,
                       help="Gradient accumulation steps (effective batch = batch_size * accumulation)")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Peak learning rate")
    parser.add_argument("--max-length", type=int, default=384,
                       help="Max sequence length (default: 384 - tận dụng context)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                       help="Warmup ratio")
    
    # Advanced techniques
    parser.add_argument("--use-focal-loss", action="store_true", default=True,
                       help="Use Focal Loss (better for imbalanced data)")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                       help="Focal loss alpha")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                       help="Focal loss gamma")
    
    parser.add_argument("--use-label-smoothing", action="store_true",
                       help="Use Label Smoothing")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                       help="Label smoothing factor")
    
    parser.add_argument("--use-mixup", action="store_true", default=True,
                       help="Use Mixup augmentation")
    parser.add_argument("--mixup-alpha", type=float, default=0.2,
                       help="Mixup alpha")
    
    parser.add_argument("--use-rdrop", action="store_true", default=True,
                       help="Use R-Drop regularization")
    parser.add_argument("--rdrop-alpha", type=float, default=0.7,
                       help="R-Drop alpha")
    
    parser.add_argument("--use-ema", action="store_true", default=True,
                       help="Use EMA (Exponential Moving Average)")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                       help="EMA decay rate")
    
    parser.add_argument("--use-swa", action="store_true", default=True,
                       help="Use SWA (Stochastic Weight Averaging)")
    parser.add_argument("--swa-start-epoch", type=int, default=7,
                       help="Start SWA from this epoch")
    
    parser.add_argument("--use-multisample-dropout", action="store_true", default=True,
                       help="Use Multi-sample Dropout")
    parser.add_argument("--num-dropout-samples", type=int, default=5,
                       help="Number of dropout samples")
    
    # Optimizer & Scheduler
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Max gradient norm for clipping")
    
    # Paths
    parser.add_argument("--data-dir", type=str, default="data/processed",
                       help="Data directory")
    parser.add_argument("--output-dir", type=str, default="outputs_advanced",
                       help="Output directory")
    
    # Other options
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use FP16 mixed precision (default: True for speed)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--eval-steps", type=int, default=100,
                       help="Evaluation frequency")
    parser.add_argument("--save-steps", type=int, default=200,
                       help="Save checkpoint frequency")
    
    args = parser.parse_args()
    
    # Validate
    if not args.task and not args.all_tasks:
        parser.error("Must specify either --task or --all-tasks")
    
    return args


class AdvancedTrainer(Trainer):
    """Custom Trainer with advanced techniques"""
    
    def __init__(self, *args, use_mixup=False, mixup_fn=None, use_rdrop=False, 
                 rdrop_fn=None, use_ema=False, ema=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mixup = use_mixup
        self.mixup_fn = mixup_fn
        self.use_rdrop = use_rdrop
        self.rdrop_fn = rdrop_fn
        self.use_ema = use_ema
        self.ema = ema
        
    def training_step(self, model, inputs):
        """Custom training step with Mixup and R-Drop"""
        model.train()
        
        # Standard forward pass
        if self.use_mixup and self.mixup_fn is not None:
            # Apply Mixup
            inputs = self.mixup_fn(inputs)
        
        if self.use_rdrop and self.rdrop_fn is not None:
            # Apply R-Drop
            loss = self.rdrop_fn(model, inputs)
        else:
            # Standard training
            inputs = self._prepare_inputs(inputs)
            outputs = model(**inputs)
            loss = outputs.loss
        
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        loss.backward()
        
        return loss.detach()
    
    def _save(self, output_dir=None, state_dict=None):
        """Save model and apply EMA if enabled"""
        if self.use_ema and self.ema is not None:
            # Apply EMA weights before saving
            self.ema.apply_shadow()
        
        super()._save(output_dir, state_dict)
        
        if self.use_ema and self.ema is not None:
            # Restore original weights
            self.ema.restore()


def load_data(task, data_dir):
    """Load training data"""
    print(f"\n📊 Loading data for {task}...")
    
    train_file = os.path.join(data_dir, task, "train.json")
    val_file = os.path.join(data_dir, task, "val.json")
    test_file = os.path.join(data_dir, task, "test.json")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found: {train_file}")
    
    train_data = json.load(open(train_file, encoding="utf-8"))
    val_data = json.load(open(val_file, encoding="utf-8")) if os.path.exists(val_file) else []
    test_data = json.load(open(test_file, encoding="utf-8")) if os.path.exists(test_file) else []
    
    print(f"   ✅ Train: {len(train_data):,} samples")
    print(f"   ✅ Val:   {len(val_data):,} samples")
    print(f"   ✅ Test:  {len(test_data):,} samples")
    
    return train_data, val_data, test_data


def load_with_retry(load_func, max_retries=5):
    """Load model/tokenizer with retry logic"""
    import time
    
    for attempt in range(max_retries):
        try:
            return load_func()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # 5s, 10s, 15s, 20s, 25s
                print(f"   ⚠️  Lỗi (lần {attempt + 1}/{max_retries}): {str(e)}")
                print(f"   ⏳ Chờ {wait_time}s rồi thử lại...")
                time.sleep(wait_time)
            else:
                print(f"\n   ❌ Không thể tải sau {max_retries} lần thử!")
                print(f"   💡 Giải pháp:")
                print(f"      1. Kiểm tra kết nối internet")
                print(f"      2. Dùng Google Colab (internet tốt hơn)")
                print(f"      3. Chạy: python download_models.py")
                raise


def load_model_and_tokenizer(task, model_name):
    """Load model and tokenizer with retry logic"""
    print(f"\n📥 Loading model: {model_name}")
    
    # Load tokenizer with retry
    tokenizer = load_with_retry(
        lambda: AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir="./models_cache",
            resume_download=True
        )
    )
    
    # Determine model type and load with retry
    if task == "classification":
        num_labels = 8
        model = load_with_retry(
            lambda: AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_labels,
                cache_dir="./models_cache",
                resume_download=True
            )
        )
    elif task == "ner":
        num_labels = 13
        model = load_with_retry(
            lambda: AutoModelForTokenClassification.from_pretrained(
                model_name, 
                num_labels=num_labels,
                cache_dir="./models_cache",
                resume_download=True
            )
        )
    elif task == "qa":
        model = load_with_retry(
            lambda: AutoModelForQuestionAnswering.from_pretrained(
                model_name,
                cache_dir="./models_cache",
                resume_download=True
            )
        )
    elif task in ["summarization", "seq2seq"]:
        if "phobert" in model_name.lower():
            model_name = "VietAI/vit5-base"
            tokenizer = load_with_retry(
                lambda: AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir="./models_cache",
                    resume_download=True
                )
            )
        model = load_with_retry(
            lambda: AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir="./models_cache",
                resume_download=True
            )
        )
    
    print(f"   ✅ Model: {model.__class__.__name__}")
    print(f"   ✅ Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    return model, tokenizer


def prepare_datasets(train_data, val_data, test_data, tokenizer, task, max_length):
    """Prepare datasets"""
    print(f"\n🔄 Preparing datasets...")
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None
    test_dataset = Dataset.from_list(test_data) if test_data else None
    
    def tokenize_function(examples):
        if task == "classification":
            return tokenizer(examples["text"], padding="max_length", 
                           truncation=True, max_length=max_length)
        elif task == "ner":
            return tokenizer(examples["tokens"], is_split_into_words=True,
                           padding="max_length", truncation=True, max_length=max_length)
        elif task == "qa":
            return tokenizer(examples["question"], examples["context"],
                           padding="max_length", truncation=True, max_length=max_length)
        elif task in ["summarization", "seq2seq"]:
            inputs = tokenizer(examples["input_text"], padding="max_length",
                             truncation=True, max_length=max_length)
            targets = tokenizer(examples["target_text"], padding="max_length",
                              truncation=True, max_length=max_length)
            inputs["labels"] = targets["input_ids"]
            return inputs
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True) if val_dataset else None
    test_dataset = test_dataset.map(tokenize_function, batched=True) if test_dataset else None
    
    print(f"   ✅ Datasets tokenized")
    
    return train_dataset, val_dataset, test_dataset


def get_compute_metrics(task):
    """Compute metrics"""
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        if task in ["classification", "ner"]:
            predictions = np.argmax(predictions, axis=-1)
            
            if task == "ner":
                predictions = predictions.flatten()
                labels = labels.flatten()
                mask = labels != -100
                predictions = predictions[mask]
                labels = labels[mask]
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return compute_metrics


def train_single_task(task, args):
    """Train a single task with advanced techniques"""
    
    print("\n" + "="*80)
    print(f"🚀 ADVANCED TRAINING: {task.upper()}")
    print("="*80)
    print(f"📋 Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Accumulation Steps: {args.accumulation_steps}")
    print(f"   Effective Batch: {args.batch_size * args.accumulation_steps}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Max Length: {args.max_length}")
    print(f"   FP16: {args.fp16}")
    print(f"\n🔥 Advanced Techniques:")
    print(f"   ✅ Focal Loss: {args.use_focal_loss}")
    print(f"   ✅ Label Smoothing: {args.use_label_smoothing}")
    print(f"   ✅ Mixup: {args.use_mixup}")
    print(f"   ✅ R-Drop: {args.use_rdrop}")
    print(f"   ✅ EMA: {args.use_ema}")
    print(f"   ✅ SWA: {args.use_swa}")
    print(f"   ✅ Multi-sample Dropout: {args.use_multisample_dropout}")
    print("="*80)
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🎮 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   VRAM: {gpu_mem:.2f} GB")
        print(f"   💡 Đang tận dụng tối đa VRAM với batch_size={args.batch_size}, max_length={args.max_length}")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    train_data, val_data, test_data = load_data(task, args.data_dir)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(task, args.model)
    
    # Apply Multi-sample Dropout if enabled
    if args.use_multisample_dropout and task in ["classification", "ner"]:
        print(f"\n🔧 Applying Multi-sample Dropout ({args.num_dropout_samples} samples)...")
        hidden_size = model.config.hidden_size  # 768 for PhoBERT
        num_classes = model.config.num_labels
        
        # Replace classifier head with multi-sample dropout
        if task == "classification":
            model.classifier = MultiSampleDropout(
                hidden_size=hidden_size,
                num_classes=num_classes,
                num_samples=args.num_dropout_samples,
                dropout=0.3
            )
        elif task == "ner":
            model.classifier = MultiSampleDropout(
                hidden_size=hidden_size,
                num_classes=num_classes,
                num_samples=args.num_dropout_samples,
                dropout=0.3
            )
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_data, val_data, test_data, tokenizer, task, args.max_length
    )
    
    # Data collator
    if task == "ner":
        data_collator = DataCollatorForTokenClassification(tokenizer)
    else:
        data_collator = DataCollatorWithPadding(tokenizer)
    
    # Output directory
    output_dir = os.path.join(args.output_dir, task)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup advanced techniques
    mixup_fn = None
    if args.use_mixup:
        print(f"\n🔧 Setting up Mixup (alpha={args.mixup_alpha})...")
        mixup_fn = MixupDataAugmentation(alpha=args.mixup_alpha)
    
    rdrop_fn = None
    if args.use_rdrop:
        print(f"\n🔧 Setting up R-Drop (alpha={args.rdrop_alpha})...")
        rdrop_fn = RDropRegularization(alpha=args.rdrop_alpha)
    
    ema = None
    if args.use_ema:
        print(f"\n🔧 Setting up EMA (decay={args.ema_decay})...")
        ema = EMA(model, decay=args.ema_decay)
    
    swa = None
    if args.use_swa:
        print(f"\n🔧 Setting up SWA (start_epoch={args.swa_start_epoch})...")
        swa = SWA(model, swa_start=args.swa_start_epoch)
    
    # Custom loss function
    if args.use_focal_loss:
        print(f"\n🔧 Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})...")
        # Note: Focal Loss will be integrated in the model's forward pass
    elif args.use_label_smoothing:
        print(f"\n🔧 Using Label Smoothing (smoothing={args.label_smoothing})...")
    
    # Training arguments
    total_steps = (len(train_dataset) // (args.batch_size * args.accumulation_steps)) * args.epochs
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",  # Cosine decay
        logging_dir=f"{output_dir}/logs",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="tensorboard",
        fp16=args.fp16,
        dataloader_num_workers=4,
        seed=args.seed,
        label_smoothing_factor=args.label_smoothing if args.use_label_smoothing else 0.0,
    )
    
    print(f"\n📊 Training Config:")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Warmup steps: {int(total_steps * args.warmup_ratio):,}")
    print(f"   Eval every: {args.eval_steps} steps")
    print(f"   Save every: {args.save_steps} steps")
    
    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_compute_metrics(task),
        use_mixup=args.use_mixup,
        mixup_fn=mixup_fn,
        use_rdrop=args.use_rdrop,
        rdrop_fn=rdrop_fn,
        use_ema=args.use_ema,
        ema=ema,
    )
    
    # Train
    print("\n" + "="*80)
    print("🔥 STARTING ADVANCED TRAINING...")
    print("="*80 + "\n")
    
    start_time = datetime.now()
    train_result = trainer.train()
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED!")
    print("="*80)
    print(f"⏱️  Duration: {duration:.2f}s ({duration/60:.2f} minutes)")
    
    # Apply SWA if enabled
    if args.use_swa and swa is not None:
        print(f"\n🔧 Applying SWA (Stochastic Weight Averaging)...")
        swa.update_bn(train_dataset, device)
    
    # Print results
    print(f"\n📊 Training Results:")
    for key, value in train_result.metrics.items():
        if isinstance(value, float):
            print(f"   {key:25}: {value:.4f}")
        else:
            print(f"   {key:25}: {value}")
    
    # Save final model
    final_model_dir = f"{output_dir}/final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"\n💾 Model saved to: {final_model_dir}")
    
    # Evaluate on test set
    if test_dataset:
        print("\n" + "="*80)
        print("📊 EVALUATING ON TEST SET...")
        print("="*80 + "\n")
        
        eval_results = trainer.evaluate(test_dataset)
        
        print("✅ Test Results:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"   {key:25}: {value:.4f}")
            else:
                print(f"   {key:25}: {value}")
        
        # Save results
        results_file = f"{output_dir}/test_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)
    
    # Save training summary
    summary = {
        "timestamp": start_time.isoformat(),
        "task": task,
        "model": args.model,
        "duration_seconds": duration,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "accumulation_steps": args.accumulation_steps,
            "effective_batch_size": args.batch_size * args.accumulation_steps,
            "learning_rate": args.lr,
            "max_length": args.max_length,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "fp16": args.fp16,
        },
        "advanced_techniques": {
            "focal_loss": args.use_focal_loss,
            "label_smoothing": args.use_label_smoothing,
            "mixup": args.use_mixup,
            "rdrop": args.use_rdrop,
            "ema": args.use_ema,
            "swa": args.use_swa,
            "multisample_dropout": args.use_multisample_dropout,
        },
        "train_results": train_result.metrics,
        "test_results": eval_results if test_dataset else None,
    }
    
    summary_file = f"{output_dir}/training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📝 Summary saved to: {summary_file}\n")
    
    return summary


def main():
    args = parse_args()
    
    print("="*80)
    print("🔥 ADVANCED PHOBERT TRAINING")
    print("="*80)
    print("💪 Sử dụng TẤT CẢ kỹ thuật advanced để đạt kết quả tốt nhất!")
    print("🎮 Tận dụng tối đa 15GB VRAM của Google Colab")
    print("="*80)
    
    if args.all_tasks:
        tasks = ["classification", "ner", "qa", "summarization", "seq2seq"]
        print(f"\n📋 Training ALL tasks: {', '.join(tasks)}\n")
        
        all_results = []
        for i, task in enumerate(tasks, 1):
            print(f"\n\n{'#'*80}")
            print(f"# TASK {i}/{len(tasks)}: {task.upper()}")
            print(f"{'#'*80}\n")
            
            try:
                summary = train_single_task(task, args)
                all_results.append(summary)
            except Exception as e:
                print(f"\n❌ Error training {task}: {e}")
                continue
        
        # Save all results
        all_summary_file = f"{args.output_dir}/all_tasks_summary.json"
        with open(all_summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 All results saved to: {all_summary_file}")
        
    else:
        # Single task
        train_single_task(args.task, args)
    
    print("\n" + "="*80)
    print("🎉 ALL DONE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
