"""
Unit tests for utils modules
"""
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import compute_multiclass_metrics, compute_binary_metrics
from utils.logger import setup_logger
import numpy as np


class TestMetrics:
    """Test metrics computation"""
    
    def test_multiclass_metrics(self):
        """Test multiclass classification metrics"""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1])
        
        metrics = compute_multiclass_metrics(y_true, y_pred, average='weighted')
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_binary_metrics(self):
        """Test binary classification metrics"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = compute_binary_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
    
    def test_metrics_with_scores(self):
        """Test metrics with probability scores"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_scores = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
        
        metrics = compute_binary_metrics(y_true, y_pred, y_scores)
        
        assert 'auc' in metrics or 'accuracy' in metrics


class TestLogger:
    """Test logging utilities"""
    
    def test_setup_logger(self, tmp_path):
        """Test logger setup"""
        log_file = tmp_path / "test.log"
        logger = setup_logger('test', str(log_file))
        
        assert logger is not None
        assert logger.name == 'test'
        
        # Test logging
        logger.info("Test message")
        
        # Check log file created
        assert os.path.exists(log_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
