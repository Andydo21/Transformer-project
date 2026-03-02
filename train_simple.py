#!/usr/bin/env python
"""
Train cả 4 tasks với 1 CONFIG DUY NHẤT
Đơn giản hơn, chỉ override parameters cho từng task
"""
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import load_config
from training.train import train_with_hf_trainer


def train_with_universal_config():
    """Train cả 4 tasks với 1 config, override từng task"""
    
    # Load base config
    base_config = load_config('configs/universal_config.yaml')
    
    tasks = ['classification', 'ner', 'qa', 'summarization']
    
    print("🚀 TRAINING ALL 4 TASKS WITH UNIVERSAL CONFIG\n")
    
    for task_name in tasks:
        print(f"\n{'='*60}")
        print(f"📌 TRAINING: {task_name.upper()}")
        print(f"{'='*60}\n")
        
        # Clone base config
        config = base_config.copy()
        task_overrides = base_config['tasks'][task_name]
        
        # Override task-specific params
        config['model']['task_type'] = task_name
        config['model']['num_labels'] = task_overrides['num_labels']
        
        # Override data paths
        data_path = task_overrides['data_path']
        config['data']['train_file'] = f"{data_path}/train.json"
        config['data']['val_file'] = f"{data_path}/val.json"
        config['data']['test_file'] = f"{data_path}/test.json"
        
        # Override output
        config['training']['output_dir'] = task_overrides['output_dir']
        config['training']['logging_dir'] = config['training']['output_dir'].replace('outputs', 'logs')
        
        # Override other task-specific settings
        if 'learning_rate' in task_overrides:
            config['training']['learning_rate'] = task_overrides['learning_rate']
        if 'batch_size' in task_overrides:
            config['training']['batch_size'] = task_overrides['batch_size']
        if 'gradient_accumulation_steps' in task_overrides:
            config['training']['gradient_accumulation_steps'] = task_overrides['gradient_accumulation_steps']
        if 'num_epochs' in task_overrides:
            config['training']['num_epochs'] = task_overrides['num_epochs']
        if 'model_name' in task_overrides:
            config['model']['name'] = task_overrides['model_name']
        
        print(f"📊 Config:")
        print(f"  Model: {config['model']['name']}")
        print(f"  Task: {config['model']['task_type']}")
        print(f"  Labels: {config['model']['num_labels']}")
        print(f"  Data: {config['data']['train_file']}")
        print(f"  Output: {config['training']['output_dir']}\n")
        
        # Train
        try:
            train_with_hf_trainer(config)
            print(f"\n✅ {task_name.upper()} completed!")
        except Exception as e:
            print(f"\n❌ {task_name.upper()} failed: {e}")
            cont = input("\nContinue? (y/n): ")
            if cont.lower() != 'y':
                break
    
    print("\n🎉 ALL TASKS COMPLETED!")


if __name__ == "__main__":
    train_with_universal_config()
