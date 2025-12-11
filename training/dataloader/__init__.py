"""
VPOG Training Dataloader Module
Handles data loading for Visual Patch-wise Object pose estimation with Groups of templates
"""

from .vpog_dataset import VPOGTrainDataset
from .template_selector import TemplateSelector, extract_d_ref_from_pose
from .flow_computer import FlowComputer, compute_patch_flows
from .vis_utils import visualize_vpog_batch, save_visualization

__all__ = [
    'VPOGTrainDataset',
    'TemplateSelector',
    'extract_d_ref_from_pose',
    'FlowComputer',
    'compute_patch_flows',
    'visualize_vpog_batch',
    'save_visualization',
]
