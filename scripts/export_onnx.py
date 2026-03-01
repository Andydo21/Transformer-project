"""
Export model to ONNX format for production deployment
Script xuất model sang format ONNX để deploy
"""
import os
import sys
import torch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import load_model
from utils.tokenizer import load_tokenizer
from utils.logger import setup_logger


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    model_name: str = "vinai/phobert-base",
    task_type: str = "classification",
    num_labels: int = 3,
    max_length: int = 512
):
    """
    Export PyTorch model to ONNX format
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_path: Output ONNX file path
        model_name: Model name
        task_type: Task type
        num_labels: Number of labels
        max_length: Max sequence length
    """
    logger = setup_logger('onnx_export')
    logger.info("Starting ONNX export...")
    
    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    model = load_model(
        model_name=model_name,
        task_type=task_type,
        num_labels=num_labels,
        checkpoint_path=checkpoint_path,
        device='cpu'
    )
    model.eval()
    
    # Create dummy input
    batch_size = 1
    dummy_input_ids = torch.randint(0, 1000, (batch_size, max_length), dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, max_length, dtype=torch.long)
    
    # Export
    logger.info(f"Exporting to {output_path}")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    logger.info("✓ ONNX export completed!")
    logger.info(f"Model saved to: {output_path}")
    
    # Verify export
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✓ ONNX model verification passed!")
    except ImportError:
        logger.warning("onnx package not installed, skipping verification")
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output', type=str, default='model.onnx', help='Output ONNX file')
    parser.add_argument('--model-name', type=str, default='vinai/phobert-base')
    parser.add_argument('--task-type', type=str, default='classification')
    parser.add_argument('--num-labels', type=int, default=3)
    parser.add_argument('--max-length', type=int, default=512)
    
    args = parser.parse_args()
    
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_name=args.model_name,
        task_type=args.task_type,
        num_labels=args.num_labels,
        max_length=args.max_length
    )
