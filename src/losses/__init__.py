"""
Loss functions for WSSS
"""
from .contrast_loss import InfoNCELossFG, InfoNCELossBG, MultiResConsistencyLoss

__all__ = ['InfoNCELossFG', 'InfoNCELossBG', 'MultiResConsistencyLoss']
