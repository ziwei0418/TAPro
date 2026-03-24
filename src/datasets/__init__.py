"""
Dataset modules for WSSS
"""
from .bcss_dataset import BCSSTrainingDataset, BCSSTestDataset, BCSSWSSSDataset, get_cls_dataset

__all__ = ['BCSSTrainingDataset', 'BCSSTestDataset', 'BCSSWSSSDataset', 'get_cls_dataset']
