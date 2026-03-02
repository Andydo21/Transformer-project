#!/usr/bin/env python
"""
Train all 4 models: Classification, NER, QA, Summarization
Chạy tuần tự từng model trên Google Colab hoặc local
"""
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import load_config
from training.train import train_with_hf_trainer


def train_all_tasks():
    """Train tất cả 4 tasks"""
    
    tasks = [
        {
            'name': 'Classification',
            'config': 'configs/classification_config.yaml',
            'icon': '📌',
            'description': 'Phân loại 8 loại luật',
            'data_size': '1,876 samples',
            'time_est': '~30-40 phút'
        },
        {
            'name': 'NER',
            'config': 'configs/ner_config.yaml',
            'icon': '🏷️',
            'description': 'Nhận diện LAW, ARTICLE',
            'data_size': '1,876 samples',
            'time_est': '~30-40 phút'
        },
        {
            'name': 'QA',
            'config': 'configs/qa_config.yaml',
            'icon': '❓',
            'description': 'Hỏi đáp về luật',
            'data_size': '1,876 samples',
            'time_est': '~40-50 phút'
        },
        {
            'name': 'Summarization',
            'config': 'configs/summarization_config.yaml',
            'icon': '📝',
            'description': 'Tóm tắt văn bản',
            'data_size': '3,504 samples',
            'time_est': '~60-80 phút'
        }
    ]
    
    print("=" * 80)
    print("🚀 TRAINING ALL 4 MODELS")
    print("=" * 80)
    print("\n📋 Tasks to train:\n")
    
    for i, task in enumerate(tasks, 1):
        print(f"{i}. {task['icon']} {task['name']}")
        print(f"   └─ {task['description']}")
        print(f"   └─ Data: {task['data_size']}")
        print(f"   └─ Time: {task['time_est']}\n")
    
    total_time = sum([40, 40, 45, 70])  # Estimated minutes
    print(f"⏱️  Total estimated time: ~{total_time//60}h {total_time%60}min")
    print(f"💾 Models will be saved to: outputs/<task_name>/")
    print("=" * 80)
    
    input("\n✅ Press ENTER to start training all 4 models, or Ctrl+C to cancel...")
    
    # Train each task
    results = []
    start_time = time.time()
    
    for i, task in enumerate(tasks, 1):
        print("\n" + "=" * 80)
        print(f"{task['icon']} TASK {i}/4: {task['name'].upper()}")
        print("=" * 80)
        
        task_start = time.time()
        
        try:
            # Load config
            print(f"\n📂 Loading config: {task['config']}")
            config = load_config(task['config'])
            
            # Train
            print(f"🚀 Starting training...")
            train_with_hf_trainer(config)
            
            task_time = (time.time() - task_start) / 60
            print(f"\n✅ {task['name']} completed in {task_time:.1f} minutes")
            
            results.append({
                'task': task['name'],
                'status': 'SUCCESS',
                'time': task_time,
                'output': config['training']['output_dir']
            })
            
        except Exception as e:
            task_time = (time.time() - task_start) / 60
            print(f"\n❌ {task['name']} FAILED after {task_time:.1f} minutes")
            print(f"Error: {str(e)}")
            
            results.append({
                'task': task['name'],
                'status': 'FAILED',
                'time': task_time,
                'error': str(e)
            })
            
            # Ask if continue
            cont = input("\nContinue with next task? (y/n): ")
            if cont.lower() != 'y':
                break
    
    # Summary
    total_time = (time.time() - start_time) / 60
    
    print("\n" + "=" * 80)
    print("📊 TRAINING SUMMARY")
    print("=" * 80)
    
    for result in results:
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"\n{status_icon} {result['task']}")
        print(f"   Status: {result['status']}")
        print(f"   Time: {result['time']:.1f} min")
        if result['status'] == 'SUCCESS':
            print(f"   Output: {result['output']}")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    
    print("\n" + "=" * 80)
    print(f"🎯 Completed: {success_count}/{len(results)} tasks")
    print(f"⏱️  Total time: {total_time//60:.0f}h {total_time%60:.0f}min")
    print("=" * 80)
    
    if success_count == len(tasks):
        print("\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        print(f"\n📁 Models saved to:")
        for result in results:
            if result['status'] == 'SUCCESS':
                print(f"   - {result['output']}")
    else:
        print(f"\n⚠️  {len(tasks) - success_count} task(s) failed. Check logs for details.")


if __name__ == "__main__":
    train_all_tasks()
