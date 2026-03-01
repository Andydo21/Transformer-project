"""
Model definitions for Vietnamese legal contract processing using PhoBERT
Định nghĩa các model xử lý hợp đồng pháp lý tiếng Việt sử dụng PhoBERT
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    AutoConfig,
    RobertaModel,
    RobertaForSequenceClassification,
    RobertaForQuestionAnswering,
    RobertaForTokenClassification
)
from typing import Optional, Dict, Tuple
import os


class ContractClassifier(nn.Module):
    """
    Model phân loại hợp đồng sử dụng PhoBERT
    Phân loại các loại hợp đồng: mua bán, thuê, dịch vụ, v.v.
    """
    
    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_labels: int = 3,
        dropout: float = 0.1,
        hidden_size: int = 768
    ):
        """
        Args:
            model_name: Tên model PhoBERT pretrained
            num_labels: Số lượng nhãn phân loại
            dropout: Dropout rate
            hidden_size: Kích thước hidden layer
        """
        super(ContractClassifier, self).__init__()
        
        # Load PhoBERT config và model
        self.config = AutoConfig.from_pretrained(model_name)
        self.phobert = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Optional: Thêm layer trung gian để tăng capacity
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            Dictionary chứa logits và loss (nếu có labels)
        """
        # PhoBERT encoding
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Lấy [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Additional dense layer (optional)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        
        # Dropout và classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss nếu có labels
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }


class ContractNER(nn.Module):
    """
    Model Named Entity Recognition cho hợp đồng
    Trích xuất các thực thể: Bên A, Bên B, giá trị, thời hạn, v.v.
    """
    
    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_labels: int = 9,  # B-PARTY, I-PARTY, B-VALUE, I-VALUE, etc.
        dropout: float = 0.1
    ):
        """
        Args:
            model_name: Tên model PhoBERT pretrained
            num_labels: Số lượng nhãn NER (theo BIO tagging)
            dropout: Dropout rate
        """
        super(ContractNER, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.phobert = AutoModel.from_pretrained(model_name)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass cho NER"""
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits
        }


class ContractQA(nn.Module):
    """
    Model Question Answering cho hợp đồng
    Trả lời câu hỏi dựa trên nội dung hợp đồng
    """
    
    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        dropout: float = 0.1
    ):
        """
        Args:
            model_name: Tên model PhoBERT pretrained
            dropout: Dropout rate
        """
        super(ContractQA, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.phobert = AutoModel.from_pretrained(model_name)
        
        # QA heads cho start và end positions
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass cho QA"""
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        # Calculate loss
        loss = None
        if start_positions is not None and end_positions is not None:
            # Clamp positions
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
        
        return {
            'loss': loss,
            'start_logits': start_logits,
            'end_logits': end_logits
        }


def load_model(
    model_name: str = "vinai/phobert-base",
    task_type: str = 'classification',
    num_labels: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu',
    **kwargs
):
    """
    Factory function để load model phù hợp
    
    Args:
        model_name: Tên PhoBERT model
        task_type: Loại tác vụ ('classification', 'ner', 'qa')
        num_labels: Số lượng nhãn
        checkpoint_path: Đường dẫn checkpoint để load
        device: Device để load model
        **kwargs: Các tham số bổ sung
    
    Returns:
        Model instance
    """
    dropout = kwargs.get('dropout', 0.1)
    
    if task_type == 'classification':
        if num_labels is None:
            raise ValueError("num_labels is required for classification")
        
        # Option 1: Sử dụng custom model
        model = ContractClassifier(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout
        )
        
        # Option 2: Sử dụng HuggingFace built-in (uncomment nếu muốn dùng)
        # model = AutoModelForSequenceClassification.from_pretrained(
        #     model_name,
        #     num_labels=num_labels
        # )
    
    elif task_type == 'ner':
        if num_labels is None:
            raise ValueError("num_labels is required for NER")
        
        model = ContractNER(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout
        )
    
    elif task_type == 'qa':
        model = ContractQA(
            model_name=model_name,
            dropout=dropout
        )
    
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    # Load checkpoint nếu có
    if checkpoint_path and os.path.exists(checkpoint_path):
        if os.path.isdir(checkpoint_path):
            checkpoint_file = os.path.join(checkpoint_path, 'pytorch_model.bin')
        else:
            checkpoint_file = checkpoint_path
        
        if os.path.exists(checkpoint_file):
            state_dict = torch.load(checkpoint_file, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✓ Đã load checkpoint từ {checkpoint_file}")
    
    model.to(device)
    return model


def save_model(
    model,
    save_path: str,
    tokenizer=None
):
    """
    Lưu model và tokenizer
    
    Args:
        model: Model instance
        save_path: Đường dẫn lưu
        tokenizer: Tokenizer instance (optional)
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Lưu model weights
    if hasattr(model, 'save_pretrained'):
        # HuggingFace model
        model.save_pretrained(save_path)
    else:
        # Custom model
        model_path = os.path.join(save_path, 'pytorch_model.bin')
        torch.save(model.state_dict(), model_path)
    
    # Lưu tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
    
    print(f"✓ Model đã được lưu tại: {save_path}")


def count_parameters(model) -> Tuple[int, int]:
    """
    Đếm số lượng parameters trong model
    
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
