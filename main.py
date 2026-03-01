"""
Main entry point for PhoBERT Contract Processing Project
Điểm vào chính cho dự án xử lý hợp đồng với PhoBERT
"""
import argparse
import yaml
import os
import sys

from training.train import train_with_hf_trainer, train_custom_loop, load_config
from inference.predict import ContractPredictor, predict_from_file
from data.dataset import prepare_data_splits
from utils.logger import setup_logger


def train_mode(args):
    """Chế độ training"""
    logger = setup_logger('main')
    logger.info("="*60)
    logger.info("MODE: TRAINING")
    logger.info("="*60)
    
    # Load config
    config = load_config(args.config)
    
    # Train
    if args.use_custom_loop:
        logger.info("Sử dụng custom training loop")
        results = train_custom_loop(config)
    else:
        logger.info("Sử dụng HuggingFace Trainer")
        results = train_with_hf_trainer(config)
    
    logger.info(f"Training completed! Results: {results}")


def predict_mode(args):
    """Chế độ prediction"""
    logger = setup_logger('main')
    logger.info("="*60)
    logger.info("MODE: PREDICTION")
    logger.info("="*60)
    
    if args.input_file:
        # Predict từ file
        logger.info(f"Predicting from file: {args.input_file}")
        predict_from_file(
            input_file=args.input_file,
            output_file=args.output_file,
            checkpoint_path=args.checkpoint,
            config_path=args.config
        )
    else:
        # Interactive prediction
        logger.info("Interactive prediction mode")
        predictor = ContractPredictor(
            checkpoint_path=args.checkpoint,
            config_path=args.config
        )
        
        print("\n" + "="*60)
        print("INTERACTIVE PREDICTION")
        print("Nhập 'quit' để thoát")
        print("="*60)
        
        while True:
            text = input("\nNhập văn bản hợp đồng: ")
            if text.lower() == 'quit':
                break
            
            result = predictor.predict(text, return_probs=True)
            print(f"\nKết quả: {result}")


def prepare_data_mode(args):
    """Chế độ chuẩn bị dữ liệu"""
    logger = setup_logger('main')
    logger.info("="*60)
    logger.info("MODE: DATA PREPARATION")
    logger.info("="*60)
    
    logger.info(f"Input: {args.input_file}")
    logger.info(f"Output: {args.output_dir}")
    
    prepare_data_splits(
        raw_data_path=args.input_file,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    
    logger.info("Data preparation completed!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='PhoBERT Contract Processing - Xử lý hợp đồng pháp lý tiếng Việt'
    )
    
    # Common arguments
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Đường dẫn file config'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Chế độ hoạt động')
    
    # Train mode
    train_parser = subparsers.add_parser('train', help='Training model')
    train_parser.add_argument(
        '--use-custom-loop',
        action='store_true',
        help='Sử dụng custom training loop thay vì HF Trainer'
    )
    
    # Predict mode
    predict_parser = subparsers.add_parser('predict', help='Dự đoán với model đã train')
    predict_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Đường dẫn model checkpoint'
    )
    predict_parser.add_argument(
        '--input-file',
        type=str,
        help='File input (JSON)'
    )
    predict_parser.add_argument(
        '--output-file',
        type=str,
        default='predictions.json',
        help='File output'
    )
    
    # Data preparation mode
    data_parser = subparsers.add_parser('prepare-data', help='Chuẩn bị dữ liệu')
    data_parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='File dữ liệu gốc'
    )
    data_parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Thư mục output'
    )
    data_parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Tỷ lệ train set'
    )
    data_parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Tỷ lệ validation set'
    )
    data_parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Tỷ lệ test set'
    )
    data_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Execute mode
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)
    elif args.mode == 'prepare-data':
        prepare_data_mode(args)
    else:
        parser.print_help()
        print("\nVí dụ sử dụng:")
        print("  python main.py train")
        print("  python main.py predict --checkpoint outputs/best_model")
        print("  python main.py prepare-data --input-file data/raw/contracts.json")


if __name__ == '__main__':
    main()
