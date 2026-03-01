"""
Test configuration
"""
import pytest


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'model': {
            'name': 'vinai/phobert-base',
            'task_type': 'classification',
            'num_labels': 3,
            'max_length': 512,
            'dropout': 0.1
        },
        'training': {
            'num_train_epochs': 1,
            'per_device_train_batch_size': 2,
            'learning_rate': 2e-5
        },
        'data': {
            'train_file': 'data/processed/train.json',
            'val_file': 'data/processed/val.json'
        }
    }


@pytest.fixture
def sample_data():
    """Sample data for testing"""
    return [
        {"text": "Sample contract 1", "label": 0},
        {"text": "Sample contract 2", "label": 1},
        {"text": "Sample contract 3", "label": 2}
    ]
