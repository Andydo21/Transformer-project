"""
Inference module
Module inference/dự đoán
"""
from .predict import ContractPredictor, predict_from_file

__all__ = [
    'ContractPredictor',
    'predict_from_file'
]
