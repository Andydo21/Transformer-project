"""
Metrics computation utilities
Module tính toán các metrics đánh giá
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from typing import Dict, List, Tuple
import torch


def compute_metrics(eval_pred, task_type: str = 'classification') -> Dict:
    """
    Compute metrics cho HuggingFace Trainer
    
    Args:
        eval_pred: EvalPrediction object từ Trainer
        task_type: Loại task ('classification', 'ner', 'qa')
    
    Returns:
        Dictionary chứa các metrics
    """
    if task_type == 'classification':
        return compute_classification_metrics(eval_pred)
    elif task_type == 'ner':
        return compute_ner_metrics(eval_pred)
    elif task_type == 'qa':
        return compute_qa_metrics(eval_pred)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def compute_classification_metrics(eval_pred) -> Dict:
    """
    Compute metrics cho classification task
    
    Args:
        eval_pred: EvalPrediction với predictions và label_ids
    
    Returns:
        Dictionary chứa accuracy, precision, recall, f1
    """
    predictions, labels = eval_pred
    
    # Get predicted labels
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    if len(predictions.shape) > 1:
        preds = np.argmax(predictions, axis=-1)
    else:
        preds = predictions
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average='weighted',
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_ner_metrics(eval_pred) -> Dict:
    """
    Compute metrics cho NER task
    
    Args:
        eval_pred: EvalPrediction với predictions và label_ids
    
    Returns:
        Dictionary chứa NER metrics
    """
    predictions, labels = eval_pred
    
    # Get predicted labels
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    preds = np.argmax(predictions, axis=-1)
    
    # Flatten và loại bỏ padding (-100)
    true_predictions = []
    true_labels = []
    
    for pred_seq, label_seq in zip(preds, labels):
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                true_predictions.append(pred)
                true_labels.append(label)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, true_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        true_predictions,
        average='weighted',
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_qa_metrics(eval_pred) -> Dict:
    """
    Compute metrics cho QA task
    
    Args:
        eval_pred: EvalPrediction với predictions và label_ids
    
    Returns:
        Dictionary chứa QA metrics (exact match, F1)
    """
    predictions, labels = eval_pred
    
    # predictions: (start_logits, end_logits)
    start_logits, end_logits = predictions
    start_preds = np.argmax(start_logits, axis=-1)
    end_preds = np.argmax(end_logits, axis=-1)
    
    # labels: (start_positions, end_positions)
    start_true, end_true = labels[:, 0], labels[:, 1]
    
    # Calculate exact match
    exact_matches = np.sum((start_preds == start_true) & (end_preds == end_true))
    exact_match_ratio = exact_matches / len(start_true)
    
    # Calculate F1 (simplified span overlap)
    f1_scores = []
    for start_p, end_p, start_t, end_t in zip(start_preds, end_preds, start_true, end_true):
        pred_span = set(range(start_p, end_p + 1))
        true_span = set(range(start_t, end_t + 1))
        
        if len(pred_span) == 0 or len(true_span) == 0:
            f1_scores.append(0.0)
            continue
        
        overlap = len(pred_span & true_span)
        precision = overlap / len(pred_span)
        recall = overlap / len(true_span)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        f1_scores.append(f1)
    
    return {
        'exact_match': exact_match_ratio,
        'f1': np.mean(f1_scores)
    }


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict:
    """
    Compute comprehensive multiclass classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted')
    
    Returns:
        Dictionary chứa tất cả metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support': support.tolist(),
        'confusion_matrix': cm.tolist()
    }


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray = None
) -> Dict:
    """
    Compute binary classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores/probabilities (optional, for AUC)
    
    Returns:
        Dictionary chứa binary metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # Add AUC if scores provided
    if y_scores is not None:
        try:
            auc = roc_auc_score(y_true, y_scores)
            metrics['auc'] = auc
        except:
            pass
    
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = None
):
    """
    In báo cáo classification chi tiết
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Tên các classes
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    print("=" * 60)


def print_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None
):
    """
    In confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Tên các labels
    """
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    
    if labels:
        print(f"{'':15s}", end='')
        for label in labels:
            print(f"{label:15s}", end='')
        print()
        
        for i, label in enumerate(labels):
            print(f"{label:15s}", end='')
            for j in range(len(labels)):
                print(f"{cm[i][j]:15d}", end='')
            print()
    else:
        print(cm)
    
    print("=" * 60)


def calculate_perplexity(loss: float) -> float:
    """
    Tính perplexity từ loss
    
    Args:
        loss: Cross-entropy loss
    
    Returns:
        Perplexity value
    """
    return np.exp(loss)


def calculate_bleu(references: List[str], predictions: List[str]) -> float:
    """
    Tính BLEU score (simplified)
    Cần nltk hoặc sacrebleu cho implementation đầy đủ
    
    Args:
        references: Reference texts
        predictions: Predicted texts
    
    Returns:
        BLEU score
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu
        scores = []
        for ref, pred in zip(references, predictions):
            score = sentence_bleu([ref.split()], pred.split())
            scores.append(score)
        return np.mean(scores)
    except ImportError:
        print("Warning: nltk not installed. Cannot calculate BLEU score.")
        return 0.0
