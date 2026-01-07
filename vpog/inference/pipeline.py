"""
VPOG Inference Pipeline

End-to-end 6D pose estimation pipeline integrating:
- TemplateManager: Template loading and selection
- CorrespondenceBuilder: 2D-3D correspondence extraction
- PnPSolver: 6D pose estimation via RANSAC

Supports single object and multi-object inference with batch processing.

Author: VPOG Team
Date: December 2025
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from vpog.inference.template_manager import TemplateManager
from vpog.inference.correspondence import CorrespondenceBuilder, Correspondences
from vpog.inference.pose_solver import PnPSolver

logger = logging.getLogger(__name__)


@dataclass
class PoseEstimate:
    """Container for a single pose estimate."""
    object_id: str
    pose: np.ndarray          # [4, 4] transformation matrix
    score: float              # Confidence score (inlier ratio)
    num_inliers: int          # Number of RANSAC inliers
    num_correspondences: int  # Total correspondences
    template_id: Optional[int] = None  # Best matching template (if tracked)
    
    def __repr__(self) -> str:
        return (f"PoseEstimate(object={self.object_id}, score={self.score:.3f}, "
                f"inliers={self.num_inliers}/{self.num_correspondences})")


@dataclass
class InferenceResult:
    """Container for inference results."""
    estimates: List[PoseEstimate]
    query_image: np.ndarray
    processing_time: float  # seconds
    
    def __len__(self) -> int:
        return len(self.estimates)
    
    def best_estimate(self) -> Optional[PoseEstimate]:
        """Get estimate with highest score."""
        if not self.estimates:
            return None
        return max(self.estimates, key=lambda x: x.score)
    
    def get_estimate(self, object_id: str) -> Optional[PoseEstimate]:
        """Get estimate for specific object."""
        for est in self.estimates:
            if est.object_id == object_id:
                return est
        return None


class InferencePipeline:
    """
    Complete VPOG inference pipeline for 6D pose estimation.
    
    Features:
    - Single/multi-object inference
    - Template-based pose estimation
    - RANSAC-based outlier rejection
    - Batch processing support
    """
    
    def __init__(
        self,
        templates_dir: Union[str, Path],
        dataset_name: str = "gso",
        template_mode: str = "all",
        cache_size: int = 10,
        ransac_threshold: float = 8.0,
        ransac_iterations: int = 1000,
        ransac_confidence: float = 0.99,
        min_inliers: int = 4,
        device: str = "cpu",
    ):
        """
        Args:
            templates_dir: Path to template images directory
            dataset_name: Dataset name ("gso", "lmo", "ycbv", etc.)
            template_mode: "all" (all templates) or "subset" (selected)
            cache_size: Number of objects to cache
            ransac_threshold: RANSAC inlier threshold (pixels)
            ransac_iterations: RANSAC iterations
            ransac_confidence: RANSAC confidence level
            min_inliers: Minimum inliers for valid pose
            device: Torch device ("cpu" or "cuda")
        """
        self.device = device
        self.min_inliers = min_inliers
        
        # Initialize components
        self.template_manager = TemplateManager(
            templates_dir=templates_dir,
            dataset_name=dataset_name,
            mode=template_mode,
            cache_size=cache_size,
            device=device,
        )
        
        self.correspondence_builder = CorrespondenceBuilder()
        
        self.pose_solver = PnPSolver(
            ransac_threshold=ransac_threshold,
            ransac_iterations=ransac_iterations,
            ransac_confidence=ransac_confidence,
        )
        
        logger.info(f"InferencePipeline initialized:")
        logger.info(f"  Dataset: {dataset_name}")
        logger.info(f"  Template mode: {template_mode}")
        logger.info(f"  Device: {device}")
    
    def estimate_pose(
        self,
        query_image: np.ndarray,
        object_id: str,
        K: np.ndarray,
        query_pose_hint: Optional[np.ndarray] = None,
    ) -> PoseEstimate:
        """
        Estimate 6D pose for a single object.
        
        Args:
            query_image: Query RGB image [H, W, 3]
            object_id: Target object ID (e.g., "000733")
            K: Camera intrinsics [3, 3]
            query_pose_hint: Optional pose hint for subset mode [4, 4]
        
        Returns:
            PoseEstimate with pose matrix and confidence
        """
        import time
        start_time = time.time()
        
        # 1. Load templates
        templates = self.template_manager.load_object_templates(
            object_id=object_id,
            query_pose=query_pose_hint,
        )
        
        # 2. Build correspondences
        # Note: This is a placeholder - actual correspondence building requires
        # model predictions (patch classification + flow)
        # For now, we'll create a dummy method that uses template matching
        correspondences = self._build_correspondences_simple(
            query_image,
            templates,
        )
        
        if len(correspondences) < self.min_inliers:
            logger.warning(f"Too few correspondences: {len(correspondences)} < {self.min_inliers}")
            return PoseEstimate(
                object_id=object_id,
                pose=np.eye(4),
                score=0.0,
                num_inliers=0,
                num_correspondences=len(correspondences),
            )
        
        # 3. Solve for pose
        R, t, inliers, num_inliers = self.pose_solver.solve(
            pts_2d=correspondences.pts_2d,
            pts_3d=correspondences.pts_3d,
            K=K,
        )
        
        # Convert R, t to 4x4 matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = R
        pose_matrix[:3, 3] = t
        
        # Compute score (inlier ratio)
        score = num_inliers / len(correspondences) if len(correspondences) > 0 else 0.0
        
        elapsed = time.time() - start_time
        logger.debug(f"Pose estimation: {elapsed*1000:.1f}ms, score={score:.3f}")
        
        return PoseEstimate(
            object_id=object_id,
            pose=pose_matrix,
            score=score,
            num_inliers=num_inliers,
            num_correspondences=len(correspondences),
        )
    
    def estimate_poses(
        self,
        query_image: np.ndarray,
        object_ids: List[str],
        K: np.ndarray,
        query_pose_hints: Optional[List[np.ndarray]] = None,
    ) -> InferenceResult:
        """
        Estimate poses for multiple objects in a single image.
        
        Args:
            query_image: Query RGB image [H, W, 3]
            object_ids: List of target object IDs
            K: Camera intrinsics [3, 3]
            query_pose_hints: Optional pose hints for subset mode
        
        Returns:
            InferenceResult with all estimates
        """
        import time
        start_time = time.time()
        
        if query_pose_hints is None:
            query_pose_hints = [None] * len(object_ids)
        
        estimates = []
        for obj_id, pose_hint in zip(object_ids, query_pose_hints):
            try:
                estimate = self.estimate_pose(
                    query_image=query_image,
                    object_id=obj_id,
                    K=K,
                    query_pose_hint=pose_hint,
                )
                estimates.append(estimate)
            except Exception as e:
                logger.error(f"Failed to estimate pose for {obj_id}: {e}")
                # Add failed estimate
                estimates.append(PoseEstimate(
                    object_id=obj_id,
                    pose=np.eye(4),
                    score=0.0,
                    num_inliers=0,
                    num_correspondences=0,
                ))
        
        elapsed = time.time() - start_time
        
        return InferenceResult(
            estimates=estimates,
            query_image=query_image,
            processing_time=elapsed,
        )
    
    def _build_correspondences_simple(
        self,
        query_image: np.ndarray,
        templates: Dict,
    ) -> Correspondences:
        """
        Simple correspondence builder using template matching.
        
        Note: This is a placeholder for demonstration. In production,
        you would use model predictions (patch classification + flow).
        
        Args:
            query_image: Query RGB image
            templates: Template data from TemplateManager
        
        Returns:
            Correspondences object
        """
        # For now, create synthetic correspondences
        # In production, this would use:
        # 1. Model forward pass to get patch predictions
        # 2. CorrespondenceBuilder to convert predictions to 2D-3D
        
        num_correspondences = 100
        
        # Random 2D points in query image
        pts_2d = torch.rand(num_correspondences, 2) * 224
        
        # Random 3D points (placeholder)
        pts_3d = torch.randn(num_correspondences, 3) * 0.1
        pts_3d[:, 2] += 1.0  # Ensure positive Z
        
        # Uniform weights
        weights = torch.ones(num_correspondences)
        valid_mask = torch.ones(num_correspondences, dtype=torch.bool)
        
        return Correspondences(
            pts_2d=pts_2d,
            pts_3d=pts_3d,
            weights=weights,
            valid_mask=valid_mask,
        )
    
    def preload_objects(self, object_ids: List[str]):
        """
        Preload templates for multiple objects into cache.
        
        Args:
            object_ids: List of object IDs to preload
        """
        self.template_manager.preload_objects(object_ids)
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "template_manager": self.template_manager.get_cache_info(),
            "device": self.device,
            "min_inliers": self.min_inliers,
        }


def create_inference_pipeline(
    templates_dir: str,
    dataset_name: str = "gso",
    **kwargs
) -> InferencePipeline:
    """
    Factory function to create an InferencePipeline.
    
    Args:
        templates_dir: Path to templates directory
        dataset_name: Dataset name
        **kwargs: Additional arguments for InferencePipeline
    
    Returns:
        Initialized InferencePipeline
    """
    return InferencePipeline(
        templates_dir=templates_dir,
        dataset_name=dataset_name,
        **kwargs
    )


if __name__ == "__main__":
    """
    Basic test of InferencePipeline.
    Run with: PYTHONPATH=/data/home/ssaricha/gigapose:$PYTHONPATH python vpog/inference/pipeline.py
    """
    import sys
    from scipy.spatial.transform import Rotation
    
    print("=" * 60)
    print("InferencePipeline Basic Test")
    print("=" * 60)
    
    # Test configuration
    templates_dir = Path(__file__).parent.parent.parent / "datasets" / "templates"
    
    # Create pipeline
    print("\n=== Test 1: Pipeline Initialization ===")
    try:
        pipeline = InferencePipeline(
            templates_dir=templates_dir,
            dataset_name="gso",
            template_mode="all",
            cache_size=5,
        )
        print("✓ Pipeline initialized")
        print(f"  Stats: {pipeline.get_stats()}")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Test single object pose estimation
    print("\n=== Test 2: Single Object Pose Estimation ===")
    
    # Create synthetic query image and camera
    query_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    K = np.array([
        [280.0, 0.0, 112.0],
        [0.0, 280.0, 112.0],
        [0.0, 0.0, 1.0]
    ])
    
    try:
        estimate = pipeline.estimate_pose(
            query_image=query_image,
            object_id="000733",
            K=K,
        )
        print(f"✓ Pose estimated: {estimate}")
        print(f"  Pose matrix:\n{estimate.pose}")
        print(f"  Score: {estimate.score:.3f}")
        print(f"  Inliers: {estimate.num_inliers}/{estimate.num_correspondences}")
    except Exception as e:
        print(f"✗ Pose estimation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test multi-object estimation
    print("\n=== Test 3: Multi-Object Pose Estimation ===")
    
    try:
        result = pipeline.estimate_poses(
            query_image=query_image,
            object_ids=["000733", "000001"],
            K=K,
        )
        print(f"✓ Estimated poses for {len(result)} objects")
        print(f"  Processing time: {result.processing_time*1000:.1f}ms")
        for est in result.estimates:
            print(f"  {est}")
        
        best = result.best_estimate()
        print(f"  Best estimate: {best}")
    except Exception as e:
        print(f"✗ Multi-object estimation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test preloading
    print("\n=== Test 4: Preloading ===")
    try:
        objects = pipeline.template_manager.get_available_objects()[:3]
        pipeline.preload_objects(objects)
        print(f"✓ Preloaded {len(objects)} objects")
        stats = pipeline.get_stats()
        print(f"  Cache: {stats['template_manager']}")
    except Exception as e:
        print(f"✗ Preloading failed: {e}")
    
    print("\n" + "=" * 60)
    print("✓ Basic tests completed")
    print("=" * 60)
