"""
Visualize training metrics
Script vẽ biểu đồ training metrics
"""
import os
import sys
import json
import matplotlib.pyplot as plt
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_training_metrics(metrics_file: str, output_dir: str = 'outputs'):
    """
    Plot training metrics from JSON file
    
    Args:
        metrics_file: Path to metrics JSON file
        output_dir: Output directory for plots
    """
    # Load metrics
    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training loss
    if 'train_loss' in metrics and metrics['train_loss']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['train_loss'], label='Train Loss', marker='o')
        if 'eval_loss' in metrics and metrics['eval_loss']:
            plt.plot(metrics['eval_loss'], label='Eval Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Evaluation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {os.path.join(output_dir, 'loss_curve.png')}")
    
    # Plot metrics (F1, Accuracy, etc.)
    if 'eval_metrics' in metrics and metrics['eval_metrics']:
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        
        plt.figure(figsize=(12, 6))
        for metric_name in metrics_to_plot:
            values = [m.get(metric_name, 0) for m in metrics['eval_metrics'] if metric_name in m]
            if values:
                plt.plot(values, label=metric_name.capitalize(), marker='o')
        
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Evaluation Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.savefig(os.path.join(output_dir, 'metrics_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {os.path.join(output_dir, 'metrics_curve.png')}")
    
    print("\n✓ All plots saved successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('--metrics-file', type=str, required=True, help='Path to metrics JSON')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    plot_training_metrics(args.metrics_file, args.output_dir)
