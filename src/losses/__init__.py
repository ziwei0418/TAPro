"""
Loss functions for WSSS
"""
from .contrast_loss import InfoNCELossFG, InfoNCELossBG
from .background_prompt_loss import BackgroundPromptSuppressionLoss
from .asymmetric_loss import AsymmetricLoss

__all__ = [
    'InfoNCELossFG',
    'InfoNCELossBG',
    'BackgroundPromptSuppressionLoss',
    'AsymmetricLoss',
]
