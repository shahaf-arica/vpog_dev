"""
BOP Evaluation Utilities

Compute BOP Challenge metrics (MSPD, MSSD) and Average Recall (AR).
Designed for efficient validation: compute pose once, evaluate against multiple thresholds.

Metrics:
- MSPD: Maximum Symmetry-Aware Projection Distance
- MSSD: Maximum Symmetry-Aware Surface Distance  
- AR: Average Recall = (2/3) * AR_MSSD + (1/3) * AR_MSPD

References:
- BOP Challenge: http://bop.felk.cvut.cz/challenges/
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from utils.eval_errors import mspd, mssd
from utils.bop_model import BOPModels


@dataclass
class PoseEvaluation:
    """Single pose evaluation result."""
    scene_id: int
    im_id: int
    obj_id: int
    mspd_error: float  # mm for projection, mm for surface
    mssd_error: float  # mm
    score: float = 0.0  # Confidence score (e.g., inlier ratio)
    
    def is_correct(self, metric: str, threshold: float) -> bool:
        """Check if pose is correct under given metric and threshold."""
        if metric == 'mspd':
            return self.mspd_error < threshold
        elif metric == 'mssd':
            return self.mssd_error < threshold
        else:
            raise ValueError(f"Unknown metric: {metric}")


@dataclass
class BOPEvaluationResults:
    """Aggregated BOP evaluation results."""
    evaluations: List[PoseEvaluation] = field(default_factory=list)
    
    # Standard BOP thresholds (as fractions of object diameter or pixels)
    mspd_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.4, 0.3, 0.2, 0.1])
    mssd_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.4, 0.3, 0.2, 0.1])
    
    def add_evaluation(self, eval_result: PoseEvaluation):
        """Add a single pose evaluation."""
        self.evaluations.append(eval_result)
    
    def compute_recall_at_threshold(self, metric: str, threshold: float) -> float:
        """
        Compute recall (pass rate) at a given threshold.
        
        Args:
            metric: 'mspd' or 'mssd'
            threshold: Error threshold (pixels for mspd, mm for mssd)
            
        Returns:
            Recall in [0, 1]
        """
        if len(self.evaluations) == 0:
            return 0.0
        
        num_correct = sum(1 for e in self.evaluations if e.is_correct(metric, threshold))
        return num_correct / len(self.evaluations)
    
    def compute_average_recall(self, metric: str) -> float:
        """
        Compute Average Recall (AR) for a metric across all thresholds.
        
        AR = mean(recall across all thresholds)
        
        Args:
            metric: 'mspd' or 'mssd'
            
        Returns:
            Average Recall in [0, 1]
        """
        thresholds = self.mspd_thresholds if metric == 'mspd' else self.mssd_thresholds
        
        if len(self.evaluations) == 0:
            return 0.0
        
        recalls = [self.compute_recall_at_threshold(metric, thresh) for thresh in thresholds]
        return np.mean(recalls)
    
    def compute_combined_ar(self) -> float:
        """
        Compute combined Average Recall without VSD.
        
        AR = (2/3) * AR_MSSD + (1/3) * AR_MSPD
        
        This is a rough approximation of the full BOP AR which includes VSD.
        """
        ar_mspd = self.compute_average_recall('mspd')
        ar_mssd = self.compute_average_recall('mssd')
        return (2/3) * ar_mssd + (1/3) * ar_mspd
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        if len(self.evaluations) == 0:
            return {
                'num_samples': 0,
                'ar_mspd': 0.0,
                'ar_mssd': 0.0,
                'ar_combined': 0.0,
                'mean_mspd_error': 0.0,
                'mean_mssd_error': 0.0,
            }
        
        return {
            'num_samples': len(self.evaluations),
            'ar_mspd': self.compute_average_recall('mspd'),
            'ar_mssd': self.compute_average_recall('mssd'),
            'ar_combined': self.compute_combined_ar(),
            'mean_mspd_error': np.mean([e.mspd_error for e in self.evaluations]),
            'mean_mssd_error': np.mean([e.mssd_error for e in self.evaluations]),
        }
    
    def get_per_threshold_recalls(self) -> Dict[str, Dict[float, float]]:
        """Get recall for each threshold (useful for detailed logging)."""
        return {
            'mspd': {
                thresh: self.compute_recall_at_threshold('mspd', thresh)
                for thresh in self.mspd_thresholds
            },
            'mssd': {
                thresh: self.compute_recall_at_threshold('mssd', thresh)
                for thresh in self.mssd_thresholds
            }
        }


class BOPEvaluator:
    """
    Evaluator for BOP metrics using model symmetries.
    
    This class handles:
    - Loading BOP models and symmetries
    - Computing MSPD and MSSD errors
    - Tracking errors across validation samples
    """
    
    def __init__(
        self,
        dataset_name: str,
        datasets_path: Optional[str] = None,
        mspd_thresholds: Optional[List[float]] = None,
        mssd_thresholds: Optional[List[float]] = None,
    ):
        """
        Initialize BOP evaluator.
        
        Args:
            dataset_name: BOP dataset name (e.g., 'ycbv', 'tless', 'lmo')
            datasets_path: Path to BOP datasets root (optional)
            mspd_thresholds: Custom MSPD thresholds in pixels (default: [0.5, 0.4, 0.3, 0.2, 0.1])
            mssd_thresholds: Custom MSSD thresholds as fraction of diameter (default: [0.5, 0.4, 0.3, 0.2, 0.1])
        """
        self.dataset_name = dataset_name
        self.bop_models = BOPModels(dataset_name, datasets_path)
        
        # Initialize results tracker
        self.results = BOPEvaluationResults()
        if mspd_thresholds is not None:
            self.results.mspd_thresholds = mspd_thresholds
        if mssd_thresholds is not None:
            self.results.mssd_thresholds = mssd_thresholds
    
    def evaluate_pose(
        self,
        R_est: np.ndarray,
        t_est: np.ndarray,
        R_gt: np.ndarray,
        t_gt: np.ndarray,
        K: np.ndarray,
        obj_id: int,
        scene_id: int = 0,
        im_id: int = 0,
        score: float = 0.0,
    ) -> PoseEvaluation:
        """
        Evaluate a single estimated pose against ground truth.
        
        Args:
            R_est: 3x3 estimated rotation matrix
            t_est: 3x1 estimated translation vector (mm)
            R_gt: 3x3 ground truth rotation matrix
            t_gt: 3x1 ground truth translation vector (mm)
            K: 3x3 camera intrinsic matrix
            obj_id: BOP object ID
            scene_id: Scene ID (for tracking)
            im_id: Image ID (for tracking)
            score: Confidence score (optional, e.g., inlier ratio)
            
        Returns:
            PoseEvaluation with MSPD and MSSD errors
        """
        # Get model points and symmetries
        pts = self.bop_models.vertices(obj_id)  # [N, 3]
        syms = self.bop_models.get_object_syms(obj_id)
        diameter = self.bop_models.model_diameters[obj_id]
        
        # Compute MSPD (pixels)
        mspd_error, _ = mspd(R_est, t_est, R_gt, t_gt, K, pts, syms)
        
        # Compute MSSD (mm) - convert to fraction of diameter for thresholding
        mssd_error_mm, _ = mssd(R_est, t_est, R_gt, t_gt, pts, syms)
        mssd_error = mssd_error_mm / diameter  # Normalize by diameter
        
        return PoseEvaluation(
            scene_id=scene_id,
            im_id=im_id,
            obj_id=obj_id,
            mspd_error=mspd_error,
            mssd_error=mssd_error,
            score=score,
        )
    
    def add_evaluation(
        self,
        R_est: np.ndarray,
        t_est: np.ndarray,
        R_gt: np.ndarray,
        t_gt: np.ndarray,
        K: np.ndarray,
        obj_id: int,
        scene_id: int = 0,
        im_id: int = 0,
        score: float = 0.0,
    ):
        """
        Evaluate and add a pose to the results tracker.
        
        This is the main method to use during validation.
        """
        eval_result = self.evaluate_pose(
            R_est, t_est, R_gt, t_gt, K, obj_id,
            scene_id, im_id, score
        )
        self.results.add_evaluation(eval_result)
    
    def reset(self):
        """Reset evaluation results."""
        self.results = BOPEvaluationResults(
            mspd_thresholds=self.results.mspd_thresholds,
            mssd_thresholds=self.results.mssd_thresholds,
        )
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics summary."""
        return self.results.get_summary()
    
    def get_detailed_metrics(self) -> Tuple[Dict[str, float], Dict[str, Dict[float, float]]]:
        """
        Get detailed metrics including per-threshold recalls.
        
        Returns:
            (summary, per_threshold_recalls)
        """
        return self.results.get_summary(), self.results.get_per_threshold_recalls()


# Convenience function for quick evaluation
def evaluate_pose_bop(
    R_est: np.ndarray,
    t_est: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    K: np.ndarray,
    obj_id: int,
    dataset_name: str,
    datasets_path: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Quick evaluation of a single pose (useful for testing/debugging).
    
    Returns:
        (mspd_error, mssd_error) in pixels and fraction of diameter
    """
    evaluator = BOPEvaluator(dataset_name, datasets_path)
    eval_result = evaluator.evaluate_pose(
        R_est, t_est, R_gt, t_gt, K, obj_id
    )
    return eval_result.mspd_error, eval_result.mssd_error
