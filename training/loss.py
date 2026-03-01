"""
Loss functions for training
Các hàm loss cho training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss để xử lý class imbalance
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Weighting factor trong khoảng (0,1)
            gamma: Focusing parameter để modulate loss
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits của shape [batch_size, num_classes]
            targets: Ground truth labels của shape [batch_size]
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss để prevent overconfidence
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1
    ):
        """
        Args:
            num_classes: Số lượng classes
            smoothing: Smoothing parameter (typically 0.1)
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits shape [batch_size, num_classes]
            targets: Ground truth labels shape [batch_size]
        
        Returns:
            Label smoothing loss value
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smooth labels
        smooth_labels = torch.zeros_like(log_probs)
        smooth_labels.fill_(self.smoothing / (self.num_classes - 1))
        smooth_labels.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = (-smooth_labels * log_probs).sum(dim=-1).mean()
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss cho class imbalance
    """
    
    def __init__(self, weights: Optional[torch.Tensor] = None):
        """
        Args:
            weights: Class weights tensor
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits
            targets: Ground truth labels
        
        Returns:
            Weighted cross entropy loss
        """
        return F.cross_entropy(inputs, targets, weight=self.weights)


class DiceLoss(nn.Module):
    """
    Dice Loss thường dùng cho segmentation và NER
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Smoothing constant để tránh division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predicted probabilities [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        
        Returns:
            Dice loss value
        """
        inputs = F.softmax(inputs, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1))
        
        intersection = (inputs * targets_one_hot).sum()
        dice = (2. * intersection + self.smooth) / (
            inputs.sum() + targets_one_hot.sum() + self.smooth
        )
        
        return 1 - dice


def get_loss_function(
    loss_type: str = 'cross_entropy',
    **kwargs
):
    """
    Factory function để get loss function
    
    Args:
        loss_type: Loại loss function
            - 'cross_entropy': Standard cross entropy
            - 'focal': Focal loss cho imbalanced data
            - 'label_smoothing': Label smoothing loss
            - 'weighted_ce': Weighted cross entropy
            - 'dice': Dice loss
        **kwargs: Additional arguments cho loss function
    
    Returns:
        Loss function instance
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 0.25)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'label_smoothing':
        num_classes = kwargs.get('num_classes', 3)
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingLoss(
            num_classes=num_classes,
            smoothing=smoothing
        )
    
    elif loss_type == 'weighted_ce':
        weights = kwargs.get('weights', None)
        if weights is not None and not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)
        return WeightedCrossEntropyLoss(weights=weights)
    
    elif loss_type == 'dice':
        smooth = kwargs.get('smooth', 1.0)
        return DiceLoss(smooth=smooth)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    """
    Tính class weights từ distribution của labels
    Sử dụng inverse frequency weighting
    
    Args:
        labels: List các labels
        num_classes: Số lượng classes
    
    Returns:
        Class weights tensor
    """
    from collections import Counter
    
    label_counts = Counter(labels)
    total = len(labels)
    
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)  # Avoid division by zero
        weight = total / (num_classes * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)
