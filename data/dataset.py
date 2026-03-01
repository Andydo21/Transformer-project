"""
Dataset module for Vietnamese legal contract processing
Module xử lý dữ liệu hợp đồng pháp lý tiếng Việt
"""
import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path


class ContractDataset(Dataset):
    """
    Dataset cho dữ liệu hợp đồng pháp lý
    Hỗ trợ các tác vụ: classification, NER, QA
    """
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer, 
        max_length: int = 512,
        task_type: str = 'classification'
    ):
        """
        Args:
            data_path: Đường dẫn đến file JSON chứa dữ liệu
            tokenizer: Tokenizer instance (PhoBERT tokenizer)
            max_length: Độ dài tối đa của sequence
            task_type: Loại tác vụ ('classification', 'ner', 'qa')
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """Load dữ liệu từ file JSON"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Đã load {len(data)} samples từ {self.data_path}")
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item tại index
        
        Returns:
            Dictionary chứa:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - labels: Label IDs (tùy loại task)
        """
        item = self.data[idx]
        
        if self.task_type == 'classification':
            return self._process_classification(item)
        elif self.task_type == 'ner':
            return self._process_ner(item)
        elif self.task_type == 'qa':
            return self._process_qa(item)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
    
    def _process_classification(self, item: Dict) -> Dict[str, torch.Tensor]:
        """Xử lý dữ liệu phân loại hợp đồng"""
        text = item.get('text', '')
        label = item.get('label', 0)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def _process_ner(self, item: Dict) -> Dict[str, torch.Tensor]:
        """Xử lý dữ liệu NER (Named Entity Recognition)"""
        tokens = item.get('tokens', [])
        labels = item.get('labels', [])
        
        # Tokenize với word-level labels
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Align labels với subword tokens
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # Ignore index
            elif word_idx != previous_word_idx:
                aligned_labels.append(labels[word_idx] if word_idx < len(labels) else 0)
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }
    
    def _process_qa(self, item: Dict) -> Dict[str, torch.Tensor]:
        """Xử lý dữ liệu Question Answering"""
        question = item.get('question', '')
        context = item.get('context', '')
        answer = item.get('answer', '')
        answer_start = item.get('answer_start', 0)
        
        # Tokenize question và context
        encoding = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation='only_second',  # Chỉ truncate context
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # Tính start và end position của answer
        offset_mapping = encoding['offset_mapping'].squeeze(0)
        start_position = 0
        end_position = 0
        
        # Tìm vị trí start và end của answer trong tokens
        for idx, (start, end) in enumerate(offset_mapping):
            if start <= answer_start < end:
                start_position = idx
            if start < answer_start + len(answer) <= end:
                end_position = idx
                break
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'start_positions': torch.tensor(start_position, dtype=torch.long),
            'end_positions': torch.tensor(end_position, dtype=torch.long)
        }


class ContractQADataset(Dataset):
    """
    Specialized Dataset cho Question Answering trên hợp đồng
    Tương thích với format dữ liệu từ training_data/qa/
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_and_process_data()
    
    def _load_and_process_data(self) -> List[Dict]:
        """Load và xử lý dữ liệu QA từ file"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            contract_text = item.get('contract_text', '')
            questions = item.get('questions', [])
            
            for qa in questions:
                examples.append({
                    'context': contract_text,
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'answer_start': qa.get('answer_start', 0)
                })
        
        print(f"Đã load {len(examples)} QA pairs từ {self.data_path}")
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        encoding = self.tokenizer(
            example['question'],
            example['context'],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation='only_second',
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        offset_mapping = encoding.pop('offset_mapping').squeeze(0)
        
        # Find start and end positions
        start_position = 0
        end_position = 0
        answer_start = example['answer_start']
        answer_end = answer_start + len(example['answer'])
        
        for idx, (start, end) in enumerate(offset_mapping):
            if start <= answer_start < end:
                start_position = idx
            if start < answer_end <= end:
                end_position = idx
                break
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'start_positions': torch.tensor(start_position, dtype=torch.long),
            'end_positions': torch.tensor(end_position, dtype=torch.long)
        }


def load_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    task_type: str = 'classification'
) -> Dataset:
    """
    Factory function để load dataset phù hợp
    
    Args:
        data_path: Đường dẫn file dữ liệu
        tokenizer: PhoBERT tokenizer
        max_length: Độ dài tối đa
        task_type: Loại tác vụ
    
    Returns:
        Dataset instance
    """
    if task_type == 'qa':
        return ContractQADataset(data_path, tokenizer, max_length)
    else:
        return ContractDataset(data_path, tokenizer, max_length, task_type)


def prepare_data_splits(
    raw_data_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
):
    """
    Chia dữ liệu thành train/val/test sets
    
    Args:
        raw_data_path: Đường dẫn file dữ liệu gốc
        output_dir: Thư mục output
        train_ratio: Tỷ lệ train set
        val_ratio: Tỷ lệ validation set
        test_ratio: Tỷ lệ test set
        random_seed: Random seed
    """
    import random
    
    random.seed(random_seed)
    
    # Load data
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Shuffle
    random.shuffle(data)
    
    # Split
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
