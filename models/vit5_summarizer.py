"""
ViT5-based Vietnamese Contract Summarization

Model: VietAI/vit5-base
Task: Abstractive summarization for contracts
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from typing import Dict, List, Optional
import numpy as np


class ViT5ContractSummarizer:
    """
    ViT5-based Contract Summarization
    
    Features:
    - Generate concise summaries
    - Highlight key information
    - Vietnamese language optimization
    - Beam search for quality
    """
    
    def __init__(
        self,
        model_name: str = "VietAI/vit5-base",
        max_input_length: int = 1024,
        max_output_length: int = 256,
        min_output_length: int = 50,
        device: Optional[str] = None
    ):
        """
        Initialize ViT5 Summarizer
        
        Args:
            model_name: ViT5 model variant (vit5-base or vit5-large)
            max_input_length: Maximum input token length
            max_output_length: Maximum summary length
            min_output_length: Minimum summary length
            device: Device to run on (cuda/cpu)
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.min_output_length = min_output_length
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading ViT5 model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        print(f"✓ Model loaded successfully")
        print(f"  Parameters: {self.model.num_parameters() / 1e6:.1f}M")
    
    def prepare_input(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Prepare input text for model
        
        Args:
            text: Input contract text
            
        Returns:
            Tokenized inputs
        """
        # For T5, prefix with task description helps
        # But ViT5 works well without it for Vietnamese
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def generate_summary(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = False
    ) -> str:
        """
        Generate summary for contract text
        
        Args:
            text: Input contract text
            max_length: Max summary length (default: self.max_output_length)
            min_length: Min summary length (default: self.min_output_length)
            num_beams: Beam search width (higher = better quality, slower)
            length_penalty: > 1.0 encourages longer summaries
            no_repeat_ngram_size: Prevent n-gram repetition
            early_stopping: Stop when all beams finish
            temperature: Sampling temperature (lower = more conservative)
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Use sampling instead of beam search
            
        Returns:
            Generated summary text
        """
        self.model.eval()
        
        max_length = max_length or self.max_output_length
        min_length = min_length or self.min_output_length
        
        # Prepare input
        inputs = self.prepare_input(text)
        
        # Generate
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample
            )
        
        # Decode
        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return summary.strip()
    
    def generate_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """
        Generate summaries for multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            **kwargs: Arguments for generate_summary
            
        Returns:
            List of summaries
        """
        summaries = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_input_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            self.model.eval()
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=kwargs.get('max_length', self.max_output_length),
                    min_length=kwargs.get('min_length', self.min_output_length),
                    num_beams=kwargs.get('num_beams', 4),
                    length_penalty=kwargs.get('length_penalty', 2.0),
                    no_repeat_ngram_size=kwargs.get('no_repeat_ngram_size', 3),
                    early_stopping=kwargs.get('early_stopping', True)
                )
            
            # Decode batch
            batch_summaries = self.tokenizer.batch_decode(
                summary_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            summaries.extend([s.strip() for s in batch_summaries])
        
        return summaries
    
    def extract_key_info(self, text: str) -> Dict[str, str]:
        """
        Extract key information using guided generation
        
        Args:
            text: Contract text
            
        Returns:
            Dictionary with key information
        """
        # Generate different aspects
        prompts = {
            "parties": "Các bên tham gia hợp đồng:",
            "value": "Giá trị hợp đồng:",
            "duration": "Thời hạn hợp đồng:",
            "payment": "Phương thức thanh toán:"
        }
        
        key_info = {}
        
        for key, prompt in prompts.items():
            # Add prompt to guide generation
            prompted_text = f"{prompt}\n{text[:500]}..."  # Use first 500 chars
            summary = self.generate_summary(
                prompted_text,
                max_length=64,
                min_length=10,
                num_beams=2
            )
            key_info[key] = summary
        
        return key_info
    
    def save_model(self, output_dir: str):
        """Save fine-tuned model"""
        print(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("✓ Model saved")
    
    def load_model(self, model_dir: str):
        """Load fine-tuned model"""
        print(f"Loading model from {model_dir}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model.to(self.device)
        print("✓ Model loaded")
    
    @staticmethod
    def compute_metrics(eval_pred, tokenizer):
        """
        Compute ROUGE metrics for evaluation
        
        Args:
            eval_pred: Predictions from trainer
            tokenizer: Tokenizer for decoding
            
        Returns:
            Dictionary of metrics
        """
        try:
            from datasets import load_metric
            rouge = load_metric("rouge")
        except:
            print("Warning: rouge_score not available, using simple metrics")
            return {}
        
        predictions, labels = eval_pred
        
        # Decode predictions
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in labels (padding)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Add newline for rouge
        decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
        decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]
        
        # Compute ROUGE
        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=False
        )
        
        # Extract scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add length metrics
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}


def create_summarizer(
    model_name: str = "VietAI/vit5-base",
    **kwargs
) -> ViT5ContractSummarizer:
    """
    Factory function to create ViT5 summarizer
    
    Args:
        model_name: Model variant (vit5-base or vit5-large)
        **kwargs: Additional arguments
        
    Returns:
        ViT5ContractSummarizer instance
    """
    return ViT5ContractSummarizer(model_name=model_name, **kwargs)


# Example usage
if __name__ == "__main__":
    # Create summarizer
    summarizer = create_summarizer()
    
    # Sample text
    text = """
    HỢP ĐỒNG CUNG CẤP DỊCH VỤ
    
    Bên A: Công ty TNHH Công Nghệ ABC
    Bên B: Công ty CP Thương Mại XYZ
    
    Giá trị hợp đồng: 500.000.000 đồng
    Thời hạn: 12 tháng
    Thanh toán: 40% trước, 60% sau
    
    Bên A cam kết cung cấp dịch vụ tư vấn và phát triển phần mềm.
    Bên B có trách nhiệm thanh toán đúng hạn.
    """
    
    # Generate summary
    print("Generating summary...")
    summary = summarizer.generate_summary(text)
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(summary)
