"""
Unit tests for data module
"""
import pytest
import os
import json
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ContractDataset, ContractQADataset, prepare_data_splits


class TestContractDataset:
    """Test ContractDataset class"""
    
    def test_load_data(self, tmp_path):
        """Test loading data from file"""
        # Create temp data file
        data = [
            {"text": "Sample contract", "label": 0},
            {"text": "Another contract", "label": 1}
        ]
        data_file = tmp_path / "test_data.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        
        # Mock tokenizer
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                import torch
                return {
                    'input_ids': torch.zeros(1, 10, dtype=torch.long),
                    'attention_mask': torch.ones(1, 10, dtype=torch.long)
                }
        
        tokenizer = MockTokenizer()
        dataset = ContractDataset(str(data_file), tokenizer, task_type='classification')
        
        assert len(dataset) == 2
    
    def test_classification_task(self, tmp_path):
        """Test classification data processing"""
        data = [{"text": "Test", "label": 1}]
        data_file = tmp_path / "test.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                import torch
                return {
                    'input_ids': torch.zeros(1, 10, dtype=torch.long),
                    'attention_mask': torch.ones(1, 10, dtype=torch.long)
                }
        
        tokenizer = MockTokenizer()
        dataset = ContractDataset(str(data_file), tokenizer, task_type='classification')
        item = dataset[0]
        
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item


class TestPrepareDataSplits:
    """Test data preparation functions"""
    
    def test_prepare_splits(self, tmp_path):
        """Test train/val/test split"""
        # Create sample data
        data = [{"text": f"Contract {i}", "label": i % 3} for i in range(10)]
        input_file = tmp_path / "input.json"
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        
        output_dir = tmp_path / "output"
        
        prepare_data_splits(
            str(input_file),
            str(output_dir),
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42
        )
        
        # Check output files exist
        assert os.path.exists(output_dir / "train.json")
        assert os.path.exists(output_dir / "val.json")
        assert os.path.exists(output_dir / "test.json")
        
        # Check data split correctly
        with open(output_dir / "train.json", 'r') as f:
            train_data = json.load(f)
        with open(output_dir / "val.json", 'r') as f:
            val_data = json.load(f)
        with open(output_dir / "test.json", 'r') as f:
            test_data = json.load(f)
        
        assert len(train_data) == 6
        assert len(val_data) == 2
        assert len(test_data) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
