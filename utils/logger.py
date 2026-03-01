"""
Logging utilities
Module tiện ích cho logging
"""
import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger với console và file handlers
    
    Args:
        name: Tên logger
        log_file: Đường dẫn file log (optional)
        level: Logging level
        format_string: Custom format string (optional)
    
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        # Create log directory if not exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger by name
    
    Args:
        name: Tên logger
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TrainingLogger:
    """
    Custom logger cho training process với metrics tracking
    """
    
    def __init__(
        self,
        log_dir: str = 'logs',
        experiment_name: str = None
    ):
        """
        Args:
            log_dir: Thư mục chứa logs
            experiment_name: Tên experiment (optional)
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f'exp_{timestamp}'
        
        self.experiment_name = experiment_name
        
        # Setup logger
        log_file = os.path.join(log_dir, f'{experiment_name}.log')
        self.logger = setup_logger(experiment_name, log_file)
        
        # Metrics storage
        self.metrics = {
            'train_loss': [],
            'eval_loss': [],
            'train_metrics': [],
            'eval_metrics': []
        }
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        eval_loss: float,
        train_metrics: dict = None,
        eval_metrics: dict = None
    ):
        """
        Log thông tin epoch
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            eval_loss: Evaluation loss
            train_metrics: Training metrics (optional)
            eval_metrics: Evaluation metrics (optional)
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"EPOCH {epoch}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Train Loss: {train_loss:.4f}")
        self.logger.info(f"Eval Loss: {eval_loss:.4f}")
        
        if train_metrics:
            self.logger.info(f"Train Metrics: {train_metrics}")
        
        if eval_metrics:
            self.logger.info(f"Eval Metrics: {eval_metrics}")
        
        # Store metrics
        self.metrics['train_loss'].append(train_loss)
        self.metrics['eval_loss'].append(eval_loss)
        if train_metrics:
            self.metrics['train_metrics'].append(train_metrics)
        if eval_metrics:
            self.metrics['eval_metrics'].append(eval_metrics)
    
    def log_step(
        self,
        step: int,
        loss: float,
        lr: float = None
    ):
        """
        Log thông tin training step
        
        Args:
            step: Step number
            loss: Loss value
            lr: Learning rate (optional)
        """
        msg = f"Step {step}: loss = {loss:.4f}"
        if lr is not None:
            msg += f", lr = {lr:.6f}"
        self.logger.info(msg)
    
    def log_metrics(self, metrics: dict, prefix: str = ''):
        """
        Log dictionary metrics
        
        Args:
            metrics: Dictionary chứa metrics
            prefix: Prefix cho log message
        """
        msg = prefix
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f" {key}={value:.4f}"
            else:
                msg += f" {key}={value}"
        self.logger.info(msg)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def save_metrics(self, output_file: str = None):
        """
        Lưu metrics vào file
        
        Args:
            output_file: Đường dẫn file output (optional)
        """
        import json
        
        if output_file is None:
            output_file = os.path.join(self.log_dir, f'{self.experiment_name}_metrics.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Metrics saved to {output_file}")


def log_model_info(model, logger: logging.Logger):
    """
    Log thông tin model
    
    Args:
        model: Model instance
        logger: Logger instance
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("=" * 60)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    logger.info("=" * 60)


def log_config(config: dict, logger: logging.Logger):
    """
    Log configuration
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    import json
    
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    logger.info(json.dumps(config, ensure_ascii=False, indent=2))
    logger.info("=" * 60)


def log_dataset_info(
    train_dataset,
    eval_dataset,
    test_dataset=None,
    logger: logging.Logger = None
):
    """
    Log thông tin datasets
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        test_dataset: Test dataset (optional)
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("DATASET INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    if test_dataset:
        logger.info(f"Test samples: {len(test_dataset)}")
    logger.info("=" * 60)
