"""
Data validation utilities
Kiểm tra và validate dữ liệu training
"""
import json
from typing import List, Dict, Tuple
from collections import Counter
import re


class DataValidator:
    """Validator cho contract data"""
    
    def __init__(self, min_text_length: int = 10, max_text_length: int = 10000):
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.issues = []
    
    def validate_sample(self, sample: dict, index: int) -> List[str]:
        """
        Validate một sample
        
        Args:
            sample: Dict chứa 'text' và 'label'
            index: Index của sample
        
        Returns:
            List các vấn đề tìm thấy
        """
        issues = []
        
        # Check required fields
        if 'text' not in sample:
            issues.append(f"Sample {index}: Missing 'text' field")
            return issues
        
        if 'label' not in sample:
            issues.append(f"Sample {index}: Missing 'label' field")
        
        text = sample['text']
        
        # Check text type
        if not isinstance(text, str):
            issues.append(f"Sample {index}: 'text' must be string, got {type(text)}")
            return issues
        
        # Check text length
        text_len = len(text.strip())
        if text_len < self.min_text_length:
            issues.append(
                f"Sample {index}: Text too short ({text_len} chars, min={self.min_text_length})"
            )
        
        if text_len > self.max_text_length:
            issues.append(
                f"Sample {index}: Text too long ({text_len} chars, max={self.max_text_length})"
            )
        
        # Check empty text
        if not text.strip():
            issues.append(f"Sample {index}: Empty text")
        
        # Check label type
        label = sample['label']
        if not isinstance(label, (int, str)):
            issues.append(f"Sample {index}: 'label' must be int or str, got {type(label)}")
        
        return issues
    
    def validate_dataset(
        self,
        data: List[dict],
        check_balance: bool = True
    ) -> Dict[str, any]:
        """
        Validate toàn bộ dataset
        
        Args:
            data: List các samples
            check_balance: Check class imbalance hay không
        
        Returns:
            Dict chứa kết quả validation
        """
        print("=" * 60)
        print("DATA VALIDATION")
        print("=" * 60)
        
        self.issues = []
        
        # Basic checks
        if not data:
            self.issues.append("Dataset is empty!")
            return {'valid': False, 'issues': self.issues}
        
        if not isinstance(data, list):
            self.issues.append(f"Data must be list, got {type(data)}")
            return {'valid': False, 'issues': self.issues}
        
        # Validate each sample
        for i, sample in enumerate(data):
            sample_issues = self.validate_sample(sample, i)
            self.issues.extend(sample_issues)
        
        # Collect statistics
        labels = [s.get('label') for s in data if 'label' in s]
        label_counts = Counter(labels)
        
        text_lengths = [len(s.get('text', '')) for s in data if 'text' in s]
        avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        
        # Check class balance
        if check_balance and len(label_counts) > 1:
            max_count = max(label_counts.values())
            min_count = min(label_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 3:
                self.issues.append(
                    f"Class imbalance detected! Ratio: {imbalance_ratio:.2f}:1"
                )
        
        # Check duplicates
        texts = [s.get('text', '') for s in data]
        unique_texts = set(texts)
        if len(texts) != len(unique_texts):
            duplicates = len(texts) - len(unique_texts)
            self.issues.append(f"Found {duplicates} duplicate texts")
        
        # Summary
        print(f"\nTotal samples: {len(data)}")
        print(f"Unique texts: {len(unique_texts)}")
        print(f"Average text length: {avg_length:.0f} chars")
        print(f"\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            percentage = (count / len(data)) * 100
            print(f"  Label {label}: {count} ({percentage:.1f}%)")
        
        if self.issues:
            print(f"\n⚠️  Found {len(self.issues)} issues:")
            for issue in self.issues[:10]:  # Show first 10
                print(f"  - {issue}")
            if len(self.issues) > 10:
                print(f"  ... and {len(self.issues) - 10} more")
        else:
            print("\n✅ All checks passed!")
        
        print("=" * 60)
        
        return {
            'valid': len(self.issues) == 0,
            'issues': self.issues,
            'stats': {
                'total_samples': len(data),
                'unique_texts': len(unique_texts),
                'avg_text_length': avg_length,
                'label_distribution': dict(label_counts),
                'duplicates': len(texts) - len(unique_texts)
            }
        }
    
    def fix_common_issues(self, data: List[dict]) -> Tuple[List[dict], List[str]]:
        """
        Tự động fix các vấn đề phổ biến
        
        Args:
            data: Dataset gốc
        
        Returns:
            (cleaned_data, actions_taken)
        """
        cleaned_data = []
        actions = []
        
        seen_texts = set()
        
        for i, sample in enumerate(data):
            # Skip invalid samples
            if not isinstance(sample, dict):
                actions.append(f"Removed sample {i}: Not a dict")
                continue
            
            if 'text' not in sample or 'label' not in sample:
                actions.append(f"Removed sample {i}: Missing required fields")
                continue
            
            # Clean text
            text = str(sample['text']).strip()
            
            # Skip empty or too short
            if len(text) < self.min_text_length:
                actions.append(f"Removed sample {i}: Text too short")
                continue
            
            # Truncate if too long
            if len(text) > self.max_text_length:
                text = text[:self.max_text_length]
                actions.append(f"Truncated sample {i}: Text too long")
            
            # Remove duplicates
            if text in seen_texts:
                actions.append(f"Removed sample {i}: Duplicate text")
                continue
            
            seen_texts.add(text)
            
            # Add cleaned sample
            cleaned_data.append({
                'text': text,
                'label': sample['label']
            })
        
        return cleaned_data, actions


def validate_json_file(file_path: str) -> bool:
    """
    Validate JSON file format và content
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        True if valid
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        validator = DataValidator()
        result = validator.validate_dataset(data)
        
        return result['valid']
    
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == '__main__':
    # Test validation
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Validating {file_path}...")
        validate_json_file(file_path)
    else:
        # Test with sample data
        sample_data = [
            {"text": "Hợp đồng mua bán", "label": 0},
            {"text": "Hợp đồng thuê nhà", "label": 1},
            {"text": "Hợp đồng dịch vụ", "label": 2},
            {"text": "", "label": 0},  # Empty - should flag
            {"text": "Short", "label": 1},  # Too short - should flag
        ]
        
        validator = DataValidator()
        result = validator.validate_dataset(sample_data)
        
        print("\n\nAuto-fixing issues...")
        cleaned_data, actions = validator.fix_common_issues(sample_data)
        print(f"\nActions taken:")
        for action in actions:
            print(f"  - {action}")
        print(f"\nCleaned data: {len(cleaned_data)} samples")
