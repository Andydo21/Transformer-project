"""
Training module
Module training cho models
"""
from .train import train_with_hf_trainer, train_custom_loop, load_config
from .eval import evaluate, evaluate_classification, evaluate_ner, evaluate_qa
from .loss import get_loss_function, FocalLoss, LabelSmoothingLoss
from .advanced_techniques import (
    FocalLoss,
    LabelSmoothingLoss,
    MixupDataAugmentation,
    StochasticWeightAveraging,
    EMA,
    WarmupScheduler,
    RDropRegularization,
    TestTimeAugmentation,
    get_optimizer_grouped_parameters,
    MultiSampleDropout
)

__all__ = [
    'train_with_hf_trainer',
    'train_custom_loop',
    'load_config',
    'evaluate',
    'evaluate_classification',
    'evaluate_ner',
    'evaluate_qa',
    'get_loss_function',
    'FocalLoss',
    'LabelSmoothingLoss',
    'MixupDataAugmentation',
    'StochasticWeightAveraging',
    'EMA',
    'WarmupScheduler',
    'RDropRegularization',
    'TestTimeAugmentation',
    'get_optimizer_grouped_parameters',
    'MultiSampleDropout'
]
