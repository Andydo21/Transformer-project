"""
Quick training script - Train nhanh với default settings
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train import train_with_hf_trainer, load_config
from utils.logger import setup_logger


def quick_train():
    """Quick training với default config"""
    logger = setup_logger('quick_train')
    
    logger.info("=" * 60)
    logger.info("QUICK TRAINING - SỬ DỤNG SAMPLE DATA")
    logger.info("=" * 60)
    
    # Load config
    config = load_config('configs/config.yaml')
    
    # Override với sample data nếu file chính không tồn tại
    if not os.path.exists(config['data']['train_file']):
        logger.warning("Train file không tồn tại, sử dụng sample data")
        
        # Tạo sample data nếu chưa có
        from data.dataset import prepare_data_splits
        sample_file = 'data/sample_data.json'
        
        if os.path.exists(sample_file):
            logger.info(f"Chuẩn bị dữ liệu từ {sample_file}")
            prepare_data_splits(
                sample_file,
                'data/processed',
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=42
            )
        else:
            logger.error("Sample data không tồn tại! Tạo file data/sample_data.json trước")
            return
    
    # Train
    logger.info("Bắt đầu training...")
    try:
        results = train_with_hf_trainer(config)
        logger.info(f"Training hoàn tất! Kết quả: {results}")
    except Exception as e:
        logger.error(f"Lỗi khi training: {e}")
        raise


if __name__ == '__main__':
    quick_train()
