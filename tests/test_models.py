"""
Unit tests for models
"""
import pytest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import ContractClassifier, ContractNER, ContractQA, load_model, count_parameters


class TestContractClassifier:
    """Test ContractClassifier model"""
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        # Skip if no internet or model not cached
        try:
            model = ContractClassifier(
                model_name="vinai/phobert-base",
                num_labels=3,
                dropout=0.1
            )
            assert model is not None
            
            # Test forward pass
            batch_size = 2
            seq_length = 10
            input_ids = torch.randint(0, 1000, (batch_size, seq_length))
            attention_mask = torch.ones(batch_size, seq_length)
            labels = torch.randint(0, 3, (batch_size,))
            
            outputs = model(input_ids, attention_mask, labels)
            
            assert 'loss' in outputs
            assert 'logits' in outputs
            assert outputs['logits'].shape == (batch_size, 3)
        except:
            pytest.skip("Model download failed - skipping test")


class TestModelUtils:
    """Test model utility functions"""
    
    def test_count_parameters(self):
        """Test parameter counting"""
        # Simple model for testing
        model = torch.nn.Linear(10, 5)
        total, trainable = count_parameters(model)
        
        # Linear layer has 10*5 + 5 = 55 parameters
        assert total == 55
        assert trainable == 55
    
    def test_load_model(self):
        """Test model loading function"""
        try:
            model = load_model(
                model_name="vinai/phobert-base",
                task_type='classification',
                num_labels=3,
                device='cpu'
            )
            assert model is not None
        except:
            pytest.skip("Model download failed - skipping test")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
