"""
Advanced training techniques for high accuracy
Các kỹ thuật training nâng cao để đạt độ chính xác cao
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR
)
from typing import Dict, List, Optional
import numpy as np
from copy import deepcopy


class FocalLoss(nn.Module):
    """
    Focal Loss - Tốt cho imbalanced datasets
    Focuses on hard examples
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing - Giảm overconfidence
    Improves generalization
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class MixupDataAugmentation:
    """
    Mixup - Mix training samples
    Tốt cho regularization và generalization
    """
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, x, y):
        """
        Args:
            x: Input batch [B, ...]
            y: Target labels [B]
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Mixup loss"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class StochasticWeightAveraging:
    """
    SWA (Stochastic Weight Averaging)
    Average model weights để improve generalization
    """
    def __init__(self, model, swa_start: int = 10, swa_lr: float = 0.05):
        self.model = model
        self.swa_model = deepcopy(model)
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.swa_n = 0
    
    def update(self, epoch: int, model):
        """Update SWA model"""
        if epoch >= self.swa_start:
            if self.swa_n == 0:
                # First SWA update
                self.swa_model.load_state_dict(model.state_dict())
            else:
                # Running average
                for swa_p, p in zip(self.swa_model.parameters(), model.parameters()):
                    swa_p.data = (swa_p.data * self.swa_n + p.data) / (self.swa_n + 1)
            self.swa_n += 1
    
    def get_swa_model(self):
        """Get averaged model"""
        return self.swa_model


class EMA:
    """
    Exponential Moving Average của model weights
    Smoother model, better generalization
    """
    def __init__(self, model, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA weights to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class GradientAccumulator:
    """
    Gradient Accumulation
    Simulate larger batch size with limited memory
    """
    def __init__(self, model, optimizer, accumulation_steps: int = 4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
    
    def step(self, loss):
        """Accumulate gradients"""
        # Scale loss
        loss = loss / self.accumulation_steps
        loss.backward()
        
        self.step_count += 1
        
        if self.step_count % self.accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step_count = 0
            return True
        return False


class WarmupScheduler:
    """
    Learning Rate Warmup + Cosine Decay
    Best practice cho transformer training
    """
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float = 2e-5,
        min_lr: float = 1e-7
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class AdversarialTraining:
    """
    Adversarial Training - Thêm nhiễu vào embeddings
    Improve robustness và generalization
    """
    def __init__(self, model, epsilon: float = 1e-5):
        self.model = model
        self.epsilon = epsilon
    
    def adversarial_step(self, embeddings, loss):
        """
        Generate adversarial embeddings
        
        Args:
            embeddings: Input embeddings
            loss: Current loss
        """
        # Get gradients wrt embeddings
        embeddings.retain_grad()
        loss.backward(retain_graph=True)
        
        # Calculate perturbation
        grad = embeddings.grad
        delta = self.epsilon * grad / (torch.norm(grad) + 1e-8)
        
        # Add perturbation
        adv_embeddings = embeddings + delta
        
        return adv_embeddings


class TestTimeAugmentation:
    """
    TTA (Test-Time Augmentation)
    Predict với multiple augmented versions, average results
    """
    def __init__(self, augmenter, num_aug: int = 5):
        self.augmenter = augmenter
        self.num_aug = num_aug
    
    def predict(self, model, text: str, tokenizer):
        """
        Predict with TTA
        
        Args:
            model: Model
            text: Input text
            tokenizer: Tokenizer
        """
        predictions = []
        
        # Original prediction
        encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = model(**encoded)
            predictions.append(torch.softmax(output.logits, dim=-1))
        
        # Augmented predictions
        aug_texts = self.augmenter.augment(text, num_aug=self.num_aug)
        for aug_text in aug_texts:
            encoded = tokenizer(aug_text, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                output = model(**encoded)
                predictions.append(torch.softmax(output.logits, dim=-1))
        
        # Average predictions
        avg_pred = torch.stack(predictions).mean(dim=0)
        
        return avg_pred


class RDropRegularization:
    """
    R-Drop Regularization
    Minimize KL divergence between two forward passes với same input
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def compute_loss(self, logits1, logits2, labels, ce_loss):
        """
        Compute R-Drop loss
        
        Args:
            logits1: First forward pass logits
            logits2: Second forward pass logits  
            labels: Ground truth labels
            ce_loss: Cross entropy loss
        """
        # Cross entropy loss
        loss_ce1 = ce_loss(logits1, labels)
        loss_ce2 = ce_loss(logits2, labels)
        ce = (loss_ce1 + loss_ce2) / 2
        
        # KL divergence
        p1 = F.log_softmax(logits1, dim=-1)
        p2 = F.log_softmax(logits2, dim=-1)
        
        kl1 = F.kl_div(p1, F.softmax(logits2, dim=-1), reduction='batchmean')
        kl2 = F.kl_div(p2, F.softmax(logits1, dim=-1), reduction='batchmean')
        kl = (kl1 + kl2) / 2
        
        # Total loss
        loss = ce + self.alpha * kl
        
        return loss


def get_optimizer_grouped_parameters(
    model,
    learning_rate: float,
    weight_decay: float,
    layerwise_lr_decay: float = 0.95
):
    """
    Layer-wise learning rate decay
    Lower layers get smaller learning rates
    """
    no_decay = ["bias", "LayerNorm.weight"]
    
    # Get layer names
    layers = [model.phobert.embeddings] + list(model.phobert.encoder.layer)
    layers.reverse()
    
    optimizer_grouped_parameters = []
    
    for i, layer in enumerate(layers):
        lr = learning_rate * (layerwise_lr_decay ** i)
        
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ])
    
    # Classifier head gets full learning rate
    optimizer_grouped_parameters.extend([
        {
            "params": [p for n, p in model.named_parameters() 
                      if "classifier" in n and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if "classifier" in n and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ])
    
    return optimizer_grouped_parameters


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout
    Sử dụng multiple dropout masks và average predictions
    """
    def __init__(self, hidden_size: int, num_classes: int, num_samples: int = 5, dropout: float = 0.5):
        super().__init__()
        self.num_samples = num_samples
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_samples)])
        self.classifiers = nn.ModuleList([nn.Linear(hidden_size, num_classes) for _ in range(num_samples)])
    
    def forward(self, x):
        """
        Forward with multiple dropout samples
        """
        logits_list = []
        for dropout, classifier in zip(self.dropouts, self.classifiers):
            logits = classifier(dropout(x))
            logits_list.append(logits)
        
        # Average logits
        logits = torch.stack(logits_list, dim=0).mean(dim=0)
        return logits
