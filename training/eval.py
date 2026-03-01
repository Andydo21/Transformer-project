"""
Evaluation utilities for model assessment
Các công cụ đánh giá model
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


def evaluate(
    model,
    eval_loader: DataLoader,
    device: torch.device,
    loss_fn=None,
    task_type: str = 'classification'
) -> Tuple[float, Dict]:
    """
    Đánh giá model trên validation/test set
    
    Args:
        model: Model cần đánh giá
        eval_loader: DataLoader cho eval data
        device: Device để sử dụng
        loss_fn: Loss function (optional)
        task_type: Loại task ('classification', 'ner', 'qa')
    
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    if task_type == 'classification':
        return evaluate_classification(model, eval_loader, device, loss_fn)
    elif task_type == 'ner':
        return evaluate_ner(model, eval_loader, device, loss_fn)
    elif task_type == 'qa':
        return evaluate_qa(model, eval_loader, device)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def evaluate_classification(
    model,
    eval_loader: DataLoader,
    device: torch.device,
    loss_fn=None
) -> Tuple[float, Dict]:
    """Đánh giá model phân loại"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            if 'loss' in outputs and outputs['loss'] is not None:
                total_loss += outputs['loss'].item()
            
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(eval_loader) if total_loss > 0 else 0
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average='weighted',
        zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average=None,
        zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist()
    }
    
    return avg_loss, metrics


def evaluate_ner(
    model,
    eval_loader: DataLoader,
    device: torch.device,
    loss_fn=None
) -> Tuple[float, Dict]:
    """Đánh giá model NER"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Evaluating NER'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            if 'loss' in outputs and outputs['loss'] is not None:
                total_loss += outputs['loss'].item()
            
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=-1)
            
            # Flatten predictions và labels, bỏ qua padding (-100)
            for pred_seq, label_seq in zip(preds, labels):
                for pred, label in zip(pred_seq, label_seq):
                    if label != -100:  # Ignore padding
                        all_preds.append(pred.item())
                        all_labels.append(label.item())
    
    # Calculate metrics
    avg_loss = total_loss / len(eval_loader) if total_loss > 0 else 0
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average='weighted',
        zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return avg_loss, metrics


def evaluate_qa(
    model,
    eval_loader: DataLoader,
    device: torch.device
) -> Tuple[float, Dict]:
    """Đánh giá model QA"""
    model.eval()
    
    total_loss = 0
    exact_matches = 0
    f1_scores = []
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Evaluating QA'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            
            if 'loss' in outputs and outputs['loss'] is not None:
                total_loss += outputs['loss'].item()
            
            # Get predictions
            start_logits = outputs['start_logits']
            end_logits = outputs['end_logits']
            
            start_preds = torch.argmax(start_logits, dim=-1)
            end_preds = torch.argmax(end_logits, dim=-1)
            
            # Calculate exact match và F1
            for start_pred, end_pred, start_true, end_true in zip(
                start_preds, end_preds, start_positions, end_positions
            ):
                if start_pred == start_true and end_pred == end_true:
                    exact_matches += 1
                
                # Simple F1 calculation (overlap of predicted vs true span)
                pred_span = set(range(start_pred.item(), end_pred.item() + 1))
                true_span = set(range(start_true.item(), end_true.item() + 1))
                
                if len(pred_span) > 0 and len(true_span) > 0:
                    overlap = len(pred_span & true_span)
                    precision = overlap / len(pred_span)
                    recall = overlap / len(true_span)
                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = 0
                    f1_scores.append(f1)
                
                total_samples += 1
    
    avg_loss = total_loss / len(eval_loader) if total_loss > 0 else 0
    
    metrics = {
        'exact_match': exact_matches / total_samples if total_samples > 0 else 0,
        'f1': np.mean(f1_scores) if f1_scores else 0
    }
    
    return avg_loss, metrics


def evaluate_and_report(
    model,
    eval_loader: DataLoader,
    device: torch.device,
    task_type: str = 'classification',
    label_names: List[str] = None
) -> Dict:
    """
    Đánh giá chi tiết và tạo report
    
    Args:
        model: Model cần đánh giá
        eval_loader: DataLoader
        device: Device
        task_type: Loại task
        label_names: Tên các nhãn (cho classification)
    
    Returns:
        Dictionary chứa metrics và report
    """
    loss, metrics = evaluate(model, eval_loader, device, None, task_type)
    
    result = {
        'loss': loss,
        'metrics': metrics
    }
    
    # Generate detailed report for classification
    if task_type == 'classification' and label_names:
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Overall Precision: {metrics['precision']:.4f}")
        print(f"Overall Recall: {metrics['recall']:.4f}")
        print(f"Overall F1: {metrics['f1']:.4f}")
        print("\nPer-class metrics:")
        for idx, name in enumerate(label_names):
            if idx < len(metrics['precision_per_class']):
                print(f"  {name}:")
                print(f"    Precision: {metrics['precision_per_class'][idx]:.4f}")
                print(f"    Recall: {metrics['recall_per_class'][idx]:.4f}")
                print(f"    F1: {metrics['f1_per_class'][idx]:.4f}")
    
    return result
