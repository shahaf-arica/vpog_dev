# vpog/inference/__init__.py

from .correspondence import (
    CorrespondenceBuilder,
    build_coarse_correspondences,
    build_refined_correspondences,
    Correspondences,
)
from .pose_solver import PnPSolver
from .epropnp_solver import EProPnPSolver, EPROPNP_AVAILABLE
from .template_manager import TemplateManager, create_template_manager
from .pipeline import (
    InferencePipeline,
    create_inference_pipeline,
    PoseEstimate,
    InferenceResult,
)

__all__ = [
    'CorrespondenceBuilder',
    'build_coarse_correspondences',
    'build_refined_correspondences',
    'Correspondences',
    'PnPSolver',
    'EProPnPSolver',
    'EPROPNP_AVAILABLE',
    'TemplateManager',
    'create_template_manager',
    'InferencePipeline',
    'create_inference_pipeline',
    'PoseEstimate',
    'InferenceResult',
]
