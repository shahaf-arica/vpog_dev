"""
VPOG Training Dataloader Module
Handles data loading for Visual Patch-wise Object pose estimation with Groups of templates
"""

from .vpog_dataset import VPOGDataset, VPOGBatch
from .template_selector import TemplateSelector, extract_d_ref_from_pose
from .flow_computer import FlowComputer, compute_patch_flows

__all__ = [
    'VPOGDataset',
    'VPOGBatch',
    'TemplateSelector',
    'extract_d_ref_from_pose',
    'FlowComputer',
    'compute_patch_flows',
]
