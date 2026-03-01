"""
All-in-One Training Launcher
Script tổng hợp để train tất cả models: Classification, NER, QA, Summarization
"""

import argparse
import subprocess
import sys
from pathlib import Path


TRAINING_CONFIGS = {
    'classification': {
        'script': 'scripts/train_classification.py',
        'data_dir': 'data/processed/classification',
        'output_dir': 'outputs/classification',
        'description': 'PhoBERT Classification (8 legal types)',
        'epochs': 5,
        'batch_size': 16,
        'learning_rate': 2e-5
    },
    'ner': {
        'script': 'scripts/train_ner.py',
        'data_dir': 'data/processed/ner',
        'output_dir': 'outputs/ner',
        'description': 'PhoBERT NER (LAW, ARTICLE, ORG, PER, DATE, LOC)',
        'epochs': 5,
        'batch_size': 16,
        'learning_rate': 3e-5
    },
    'qa': {
        'script': 'scripts/train_qa.py',
        'data_dir': 'data/processed/qa',
        'output_dir': 'outputs/qa',
        'description': 'PhoBERT Question Answering',
        'epochs': 5,
        'batch_size': 16,
        'learning_rate': 3e-5
    },
    'summarization': {
        'script': 'scripts/train_vit5_summarizer.py',
        'data_dir': 'data/processed/summarization',
        'output_dir': 'outputs/summarization',
        'description': 'ViT5 Abstractive Summarization',
        'epochs': 5,
        'batch_size': 8,
        'learning_rate': 5e-5
    }
}


def print_banner():
    """Print training banner."""
    print("\n" + "=" * 70)
    print("🚀 ALL-IN-ONE MODEL TRAINING LAUNCHER")
    print("=" * 70)
    print("\n📋 Available training tasks:")
    for i, (task, config) in enumerate(TRAINING_CONFIGS.items(), 1):
        print(f"  {i}. {task:15s} - {config['description']}")
    print("\n" + "=" * 70 + "\n")


def run_training(task_name, args):
    """Run training for a specific task."""
    
    config = TRAINING_CONFIGS[task_name]
    
    print(f"\n{'='*70}")
    print(f"🎯 TRAINING: {task_name.upper()}")
    print(f"{'='*70}")
    print(f"📄 Description: {config['description']}")
    print(f"📁 Data: {config['data_dir']}")
    print(f"💾 Output: {config['output_dir']}")
    print(f"⚙️  Epochs: {config['epochs']}, Batch: {config['batch_size']}, LR: {config['learning_rate']}")
    print(f"{'='*70}\n")
    
    # Build command
    cmd = [
        sys.executable,
        config['script'],
        '--data-dir', config['data_dir'],
        '--output-dir', config['output_dir'],
        '--epochs', str(args.epochs if args.epochs else config['epochs']),
        '--batch-size', str(args.batch_size if args.batch_size else config['batch_size']),
        '--learning-rate', str(args.learning_rate if args.learning_rate else config['learning_rate']),
        '--max-length', str(args.max_length),
        '--logging-steps', str(args.logging_steps),
    ]
    
    # Add FP16 if specified
    if args.fp16:
        cmd.append('--fp16')
    
    # Add model name if specified
    if args.model_name:
        cmd.extend(['--model-name', args.model_name])
    
    print(f"🔧 Command: {' '.join(cmd)}\n")
    
    try:
        # Run training
        result = subprocess.run(cmd, check=True)
        
        print(f"\n✅ {task_name.upper()} training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {task_name.upper()} training failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ {task_name.upper()} training failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Train all models for Vietnamese legal contract processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models with default settings
  python scripts/train_all.py --tasks all
  
  # Train specific models
  python scripts/train_all.py --tasks classification ner
  
  # Train with custom settings
  python scripts/train_all.py --tasks all --epochs 10 --batch-size 32 --fp16
  
  # Train classification only with custom learning rate
  python scripts/train_all.py --tasks classification --learning-rate 1e-5
        """
    )
    
    parser.add_argument('--tasks', nargs='+', 
                       choices=['all', 'classification', 'ner', 'qa', 'summarization'],
                       default=['all'],
                       help='Tasks to train (default: all)')
    parser.add_argument('--model-name', type=str,
                       help='Pretrained model name (default: vinai/phobert-base for PhoBERT, VietAI/vit5-base for ViT5)')
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs (overrides task defaults)')
    parser.add_argument('--batch-size', type=int,
                       help='Training batch size (overrides task defaults)')
    parser.add_argument('--learning-rate', type=float,
                       help='Learning rate (overrides task defaults)')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Max sequence length (default: 512)')
    parser.add_argument('--logging-steps', type=int, default=50,
                       help='Logging frequency (default: 50)')
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision training (FP16)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip tasks that already have trained models')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Determine tasks to run
    if 'all' in args.tasks:
        tasks_to_run = list(TRAINING_CONFIGS.keys())
    else:
        tasks_to_run = args.tasks
    
    print(f"🎯 Tasks to train: {', '.join(tasks_to_run)}\n")
    
    # Check if we should skip existing models
    if args.skip_existing:
        filtered_tasks = []
        for task in tasks_to_run:
            output_dir = Path(TRAINING_CONFIGS[task]['output_dir']) / 'final_model'
            if output_dir.exists():
                print(f"⏭️  Skipping {task} (model already exists at {output_dir})")
            else:
                filtered_tasks.append(task)
        tasks_to_run = filtered_tasks
        print()
    
    if not tasks_to_run:
        print("ℹ️  No tasks to run (all models already trained or no tasks selected)")
        return
    
    # Track results
    results = {}
    
    # Train each task
    for i, task in enumerate(tasks_to_run, 1):
        print(f"\n{'#'*70}")
        print(f"# Task {i}/{len(tasks_to_run)}: {task.upper()}")
        print(f"{'#'*70}")
        
        success = run_training(task, args)
        results[task] = success
        
        if not success and i < len(tasks_to_run):
            print(f"\n⚠️  {task} failed, but continuing with remaining tasks...")
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for task, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {task:20s} {status}")
    
    print(f"\n🎯 Overall: {success_count}/{total_count} tasks completed successfully")
    
    if success_count == total_count:
        print("\n🎉 All training tasks completed successfully!")
        print("\n📁 Models saved in:")
        for task in results.keys():
            output_dir = TRAINING_CONFIGS[task]['output_dir']
            print(f"  - {task:20s} → {output_dir}/final_model")
    else:
        print(f"\n⚠️  {total_count - success_count} task(s) failed. Please check the logs above.")
    
    print("\n" + "=" * 70 + "\n")
    
    # Exit with appropriate code
    sys.exit(0 if success_count == total_count else 1)


if __name__ == '__main__':
    main()
