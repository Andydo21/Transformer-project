"""
Configuration loader
Load configuration từ YAML và environment variables
"""
import os
import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration từ file YAML
    
    Args:
        config_path: Path to config YAML file
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default config path
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'configs',
            'config.yaml'
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Override với environment variables nếu có
    config = override_with_env(config)
    
    return config


def override_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override config với environment variables
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Updated configuration
    """
    # Model configuration
    if os.getenv('MODEL_NAME'):
        config['model']['name'] = os.getenv('MODEL_NAME')
    if os.getenv('TASK_TYPE'):
        config['model']['task_type'] = os.getenv('TASK_TYPE')
    if os.getenv('NUM_LABELS'):
        config['model']['num_labels'] = int(os.getenv('NUM_LABELS'))
    
    # Training configuration
    if os.getenv('LEARNING_RATE'):
        config['training']['learning_rate'] = float(os.getenv('LEARNING_RATE'))
    if os.getenv('NUM_EPOCHS'):
        config['training']['num_epochs'] = int(os.getenv('NUM_EPOCHS'))
    if os.getenv('BATCH_SIZE'):
        config['training']['batch_size'] = int(os.getenv('BATCH_SIZE'))
    
    # Paths
    if os.getenv('DATA_DIR'):
        config['data']['data_dir'] = os.getenv('DATA_DIR')
    if os.getenv('OUTPUT_DIR'):
        config['training']['output_dir'] = os.getenv('OUTPUT_DIR')
    
    # Device
    if os.getenv('DEVICE'):
        config['training']['device'] = os.getenv('DEVICE')
    
    return config


def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent


def ensure_dirs(config: Dict[str, Any]):
    """
    Ensure required directories exist
    
    Args:
        config: Configuration dictionary
    """
    dirs_to_create = [
        config['training']['output_dir'],
        config['training']['log_dir'],
        config['data']['data_dir'],
        os.path.join(config['data']['data_dir'], 'raw'),
        os.path.join(config['data']['data_dir'], 'processed')
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)


if __name__ == '__main__':
    # Test config loader
    config = load_config()
    print("Configuration loaded successfully:")
    print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
