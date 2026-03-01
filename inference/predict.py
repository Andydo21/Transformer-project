"""
Inference/Prediction module for trained models
Module inference/dự đoán cho các model đã train
"""
import os
import sys
import yaml
import torch
from typing import List, Dict, Union, Tuple
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import load_model
from utils.tokenizer import load_tokenizer
from utils.logger import setup_logger


class ContractPredictor:
    """
    Predictor class cho các tác vụ xử lý hợp đồng
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = 'configs/config.yaml',
        device: str = None
    ):
        """
        Args:
            checkpoint_path: Đường dẫn đến model checkpoint
            config_path: Đường dẫn đến file config
            device: Device để sử dụng ('cuda', 'cpu', hoặc None để auto-detect)
        """
        self.logger = setup_logger('predictor')
        
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = load_tokenizer(self.config['model']['name'])
        self.logger.info(f"Loaded tokenizer: {self.config['model']['name']}")
        
        # Load model
        self.model = load_model(
            model_name=self.config['model']['name'],
            task_type=self.config['model']['task_type'],
            num_labels=self.config['model']['num_labels'],
            checkpoint_path=checkpoint_path,
            device=str(self.device)
        )
        self.model.eval()
        self.logger.info(f"Loaded model from: {checkpoint_path}")
        
        self.task_type = self.config['model']['task_type']
        self.max_length = self.config['model']['max_length']
    
    def predict(
        self,
        text: Union[str, List[str]],
        return_probs: bool = False
    ) -> Union[Dict, List[Dict]]:
        """
        Dự đoán cho text hoặc danh sách texts
        
        Args:
            text: Văn bản hoặc danh sách văn bản
            return_probs: Có trả về probabilities không
        
        Returns:
            Kết quả dự đoán (dict hoặc list of dicts)
        """
        single_input = isinstance(text, str)
        if single_input:
            text = [text]
        
        if self.task_type == 'classification':
            results = self._predict_classification(text, return_probs)
        elif self.task_type == 'ner':
            results = self._predict_ner(text)
        elif self.task_type == 'qa':
            raise NotImplementedError("Use predict_qa() method for QA tasks")
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
        
        return results[0] if single_input else results
    
    def _predict_classification(
        self,
        texts: List[str],
        return_probs: bool = False
    ) -> List[Dict]:
        """Dự đoán phân loại hợp đồng"""
        results = []
        
        # Get contract type names from config
        contract_types = self.config.get('contract', {}).get('contract_types', [])
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Predict
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=-1)[0]
                pred_label = torch.argmax(probs).item()
                confidence = probs[pred_label].item()
                
                result = {
                    'predicted_label': pred_label,
                    'confidence': confidence
                }
                
                # Add label name if available
                if contract_types and pred_label < len(contract_types):
                    result['label_name'] = contract_types[pred_label]
                
                if return_probs:
                    result['probabilities'] = probs.cpu().numpy().tolist()
                
                results.append(result)
        
        return results
    
    def _predict_ner(self, texts: List[str]) -> List[Dict]:
        """Dự đoán NER cho hợp đồng"""
        results = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    return_offsets_mapping=True
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                offset_mapping = encoding['offset_mapping'][0]
                
                # Predict
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs['logits'][0]
                predictions = torch.argmax(logits, dim=-1)
                
                # Extract entities
                entities = []
                current_entity = None
                
                for idx, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
                    if start == end:  # Skip special tokens
                        continue
                    
                    pred_label = pred.item()
                    
                    # Simple entity extraction (customize based on your label scheme)
                    if pred_label > 0:  # Not O (Outside)
                        entity_text = text[start:end]
                        entities.append({
                            'text': entity_text,
                            'label': pred_label,
                            'start': start.item(),
                            'end': end.item()
                        })
                
                results.append({
                    'text': text,
                    'entities': entities
                })
        
        return results
    
    def predict_qa(
        self,
        question: str,
        context: str
    ) -> Dict:
        """
        Dự đoán cho Question Answering
        
        Args:
            question: Câu hỏi
            context: Ngữ cảnh (văn bản hợp đồng)
        
        Returns:
            Dictionary chứa answer và positions
        """
        if self.task_type != 'qa':
            raise ValueError("Model task_type must be 'qa' for this method")
        
        with torch.no_grad():
            # Tokenize
            encoding = self.tokenizer(
                question,
                context,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation='only_second',
                return_tensors='pt',
                return_offsets_mapping=True
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            offset_mapping = encoding['offset_mapping'][0]
            
            # Predict
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            start_logits = outputs['start_logits'][0]
            end_logits = outputs['end_logits'][0]
            
            start_idx = torch.argmax(start_logits).item()
            end_idx = torch.argmax(end_logits).item()
            
            # Extract answer từ context
            if start_idx <= end_idx:
                start_char = offset_mapping[start_idx][0].item()
                end_char = offset_mapping[end_idx][1].item()
                answer = context[start_char:end_char]
            else:
                answer = ""
            
            # Calculate confidence
            start_prob = torch.softmax(start_logits, dim=0)[start_idx].item()
            end_prob = torch.softmax(end_logits, dim=0)[end_idx].item()
            confidence = (start_prob + end_prob) / 2
            
            return {
                'question': question,
                'answer': answer,
                'confidence': confidence,
                'start_position': start_idx,
                'end_position': end_idx
            }
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 16
    ) -> List[Dict]:
        """
        Dự đoán theo batch cho hiệu quả cao hơn
        
        Args:
            texts: Danh sách văn bản
            batch_size: Kích thước batch
        
        Returns:
            List kết quả dự đoán
        """
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self.predict(batch_texts)
            all_results.extend(batch_results)
        
        return all_results
    
    def predict_ner(self, text: str) -> Dict:
        """
        Public method for Named Entity Recognition
        
        Args:
            text: Input text for NER
        
        Returns:
            Dictionary with entities found
        """
        if self.task_type != 'ner':
            raise ValueError("Model task_type must be 'ner' for this method")
        
        results = self._predict_ner([text])
        return results[0] if results else {'text': text, 'entities': []}
    
    def predict_summary(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 50
    ) -> Dict:
        """
        Public method for text summarization
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
        
        Returns:
            Dictionary with summary
        """
        if self.task_type != 'summarization':
            raise ValueError("Model task_type must be 'summarization' for this method")
        
        # This is a placeholder - actual implementation depends on your summarization model
        # For ViT5 or similar models, you'd use generate() method
        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Generate summary
            summary_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True
            )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return {
                'text': text,
                'summary': summary,
                'max_length': max_length,
                'min_length': min_length
            }


def predict_from_file(
    input_file: str,
    output_file: str,
    checkpoint_path: str,
    config_path: str = 'configs/config.yaml'
):
    """
    Dự đoán từ file input và lưu kết quả vào file output
    
    Args:
        input_file: File chứa dữ liệu input (JSON)
        output_file: File để lưu kết quả
        checkpoint_path: Đường dẫn model checkpoint
        config_path: Đường dẫn config file
    """
    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize predictor
    predictor = ContractPredictor(checkpoint_path, config_path)
    
    # Extract texts
    if isinstance(data, list):
        texts = [item.get('text', '') for item in data]
    else:
        texts = [data.get('text', '')]
    
    # Predict
    results = predictor.predict_batch(texts)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Đã lưu {len(results)} kết quả vào {output_file}")


if __name__ == '__main__':
    # Example usage
    checkpoint_path = 'outputs/best_model'
    config_path = 'configs/config.yaml'
    
    # Initialize predictor
    predictor = ContractPredictor(checkpoint_path, config_path)
    
    # Example text
    sample_text = """
    CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
    Độc lập - Tự do - Hạnh phúc
    
    HỢP ĐỒNG MUA BÁN HÀNG HÓA
    Số: 123/2026/HĐMB
    
    Hôm nay, ngày 22 tháng 01 năm 2026, tại Hà Nội, chúng tôi gồm:
    Bên A: Công ty TNHH ABC
    Bên B: Công ty Cổ phần XYZ
    """
    
    # Predict
    result = predictor.predict(sample_text, return_probs=True)
    print("\nKết quả dự đoán:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
