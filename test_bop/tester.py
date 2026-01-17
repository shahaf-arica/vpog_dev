"""
VPOG BOP Tester

Main testing orchestrator that:
1. Loads VPOG checkpoint
2. Iterates through BOP test dataset
3. Calls vpog.inference for pose estimation
4. Collects results (poses + timing)
5. Saves intermediate results

Author: VPOG Team
Date: January 2026
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import numpy as np
import torch
from tqdm import tqdm

from src.utils.logging import get_logger
from test_bop.dataloader import BOPTestDataset, BOPDetection
from test_bop.bop_formatter import BOPFormatter
from vpog.inference.pipeline import InferencePipeline, PoseEstimate

logger = get_logger(__name__)


@dataclass
class TestResult:
    """Single test result with pose and timing."""
    scene_id: int
    im_id: int
    obj_id: int
    score: float
    R: np.ndarray  # [3, 3] rotation matrix
    t: np.ndarray  # [3] translation vector (mm)
    time: float  # Total inference time (seconds)
    detection_time: float  # Detection time (seconds)
    num_inliers: int = 0
    num_correspondences: int = 0
    template_id: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dict for saving."""
        return {
            'scene_id': int(self.scene_id),
            'im_id': int(self.im_id),
            'obj_id': int(self.obj_id),
            'score': float(self.score),
            'R': self.R.reshape(-1).tolist(),  # Flatten for CSV
            't': self.t.tolist(),
            'time': float(self.time),
            'detection_time': float(self.detection_time),
            'num_inliers': int(self.num_inliers),
            'num_correspondences': int(self.num_correspondences),
        }


class VPOGBOPTester:
    """
    VPOG BOP Tester - Main testing orchestrator.
    
    Handles end-to-end testing:
    - Load VPOG model and checkpoint
    - Process BOP test images
    - Run VPOG inference for pose estimation
    - Save results in BOP format
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        templates_dir: str,
        root_dir: str,
        dataset_name: str,
        output_dir: str,
        test_setting: str = "localization",
        device: str = "cuda",
        batch_size: int = 1,
        **inference_kwargs,
    ):
        """
        Args:
            checkpoint_path: Path to VPOG checkpoint (.ckpt file)
            templates_dir: Directory with rendered templates
            root_dir: BOP datasets root directory
            dataset_name: BOP dataset name (ycbv, tless, lmo, etc.)
            output_dir: Output directory for results
            test_setting: "localization" or "detection"
            device: Device for inference ("cuda" or "cpu")
            batch_size: Batch size (usually 1 for testing)
            **inference_kwargs: Additional args for InferencePipeline
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.templates_dir = Path(templates_dir)
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.test_setting = test_setting
        self.device = device
        self.batch_size = batch_size
        
        # Create output directories
        self.predictions_dir = self.output_dir / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing VPOG BOP Tester for {dataset_name}")
        logger.info(f"  Checkpoint: {self.checkpoint_path}")
        logger.info(f"  Templates: {self.templates_dir}")
        logger.info(f"  Output: {self.output_dir}")
        
        # Load VPOG model
        self._load_model(**inference_kwargs)
        
        # Load test dataset
        self._load_dataset()
        
        # BOP formatter for output conversion
        self.formatter = BOPFormatter(
            dataset_name=dataset_name,
            output_dir=self.output_dir,
        )
        
        logger.info("VPOG BOP Tester initialized successfully")
    
    def _load_model(self, **inference_kwargs):
        """Load VPOG model from checkpoint."""
        logger.info("Loading VPOG model...")
        
        # Load Lightning module from checkpoint
        from training.lightning_module import VPOGLightningModule
        
        lightning_module = VPOGLightningModule.load_from_checkpoint(
            self.checkpoint_path,
            map_location=self.device,
        )
        lightning_module.eval()
        lightning_module.to(self.device)
        
        # Extract the model
        self.model = lightning_module.model
        
        # Initialize InferencePipeline with the loaded model
        self.inference_pipeline = InferencePipeline(
            templates_dir=str(self.templates_dir),
            dataset_name=self.dataset_name,
            device=self.device,
            **inference_kwargs
        )
        
        # Store the model in the pipeline (if needed)
        if hasattr(self.inference_pipeline, 'model'):
            self.inference_pipeline.model = self.model
        
        logger.info(f"Model loaded successfully from {self.checkpoint_path}")
    
    def _load_dataset(self):
        """Load BOP test dataset."""
        logger.info("Loading BOP test dataset...")
        self.dataset = BOPTestDataset(
            root_dir=str(self.root_dir),
            dataset_name=self.dataset_name,
            templates_dir=str(self.templates_dir),
            test_setting=self.test_setting,
        )
        logger.info(f"Loaded {len(self.dataset)} test detections")
    
    def run_test(self) -> List[TestResult]:
        """
        Run full BOP test.
        
        Returns:
            List of TestResult objects
        """
        logger.info(f"Starting BOP test on {self.dataset_name}")
        logger.info(f"Total detections: {len(self.dataset)}")
        
        all_results = []
        batch_results = []
        
        # Process each detection
        for idx in tqdm(range(len(self.dataset)), desc="Testing"):
            # Load data
            data = self.dataset[idx]
            detection = data["detection"]
            
            # Run inference
            try:
                result = self._process_detection(data, detection)
                all_results.append(result)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process detection {idx}: {e}")
                logger.exception(e)
                continue
            
            # Save batch periodically (every 100 detections)
            if len(batch_results) >= 100:
                self._save_batch(batch_results, len(all_results) // 100)
                batch_results = []
        
        # Save remaining batch
        if batch_results:
            self._save_batch(batch_results, (len(all_results) // 100) + 1)
        
        logger.info(f"Testing complete: {len(all_results)} results")
        
        # Convert to BOP format
        self._finalize_results(all_results)
        
        return all_results
    
    def _process_detection(self, data: Dict[str, Any], detection: BOPDetection) -> TestResult:
        """
        Process single detection.
        
        Args:
            data: Detection data from dataset
            detection: BOPDetection metadata
            
        Returns:
            TestResult with pose and timing
        """
        query_image = data["query_image"]
        templates = data["templates"]
        
        # Run VPOG inference
        start_time = time.time()
        
        # Convert query image to numpy [H, W, 3] uint8
        if isinstance(query_image, torch.Tensor):
            query_image = query_image.numpy().transpose(1, 2, 0)
        if query_image.max() <= 1.0:
            query_image = (query_image * 255).astype(np.uint8)
        
        # Convert K to numpy if needed
        query_K = data["query_K"]
        if isinstance(query_K, torch.Tensor):
            query_K = query_K.numpy()
        
        object_id_str = f"{detection.obj_id:06d}"
        
        try:
            pose_estimate = self.inference_pipeline.estimate_pose(
                query_image=query_image,
                object_id=object_id_str,
                K=query_K,
                query_pose_hint=None,
            )
            inference_time = time.time() - start_time
            R = pose_estimate.pose[:3, :3]
            t = pose_estimate.pose[:3, 3] * 1000.0  # Convert to mm for BOP
            score = pose_estimate.score
            num_inliers = pose_estimate.num_inliers
            num_correspondences = pose_estimate.num_correspondences
            template_id = pose_estimate.template_id
        except Exception as e:
            logger.warning(f"Inference failed for detection {detection}: {e}")
            inference_time = time.time() - start_time
            R = np.eye(3, dtype=np.float32)
            t = np.zeros(3, dtype=np.float32)
            score = 0.0
            num_inliers = 0
            num_correspondences = 0
            template_id = None
        
        return TestResult(
            scene_id=detection.scene_id,
            im_id=detection.im_id,
            obj_id=detection.obj_id,
            score=score,
            R=R,
            t=t,
            time=inference_time,
            detection_time=detection.time,
            num_inliers=num_inliers,
            num_correspondences=num_correspondences,
            template_id=template_id,
        )
    
    def _save_batch(self, results: List[TestResult], batch_id: int):
        """Save batch of results to .npz file."""
        batch_path = self.predictions_dir / f"{batch_id:06d}.npz"
        
        # Convert to arrays
        scene_ids = np.array([r.scene_id for r in results], dtype=np.int32)
        im_ids = np.array([r.im_id for r in results], dtype=np.int32)
        obj_ids = np.array([r.obj_id for r in results], dtype=np.int32)
        scores = np.array([r.score for r in results], dtype=np.float32)
        times = np.array([r.time for r in results], dtype=np.float32)
        detection_times = np.array([r.detection_time for r in results], dtype=np.float32)
        
        # Stack poses
        poses = np.zeros((len(results), 4, 4), dtype=np.float32)
        for i, r in enumerate(results):
            poses[i, :3, :3] = r.R
            poses[i, :3, 3] = r.t
            poses[i, 3, 3] = 1.0
        
        # Save
        np.savez(
            batch_path,
            scene_id=scene_ids,
            im_id=im_ids,
            object_id=obj_ids,
            scores=scores,
            poses=poses,
            time=times,
            detection_time=detection_times,
        )
        
        logger.debug(f"Saved batch {batch_id} with {len(results)} results")
    
    def _finalize_results(self, results: List[TestResult]):
        """Convert results to BOP CSV format."""
        logger.info("Converting results to BOP format...")
        
        # Use BOP formatter
        csv_path = self.formatter.convert_to_csv(
            results=results,
            model_name="vpog",
            run_id="test",
        )
        
        logger.info(f"BOP CSV saved to: {csv_path}")


if __name__ == "__main__":
    # Example usage
    from src.utils.logging import setup_logger
    setup_logger()
    
    tester = VPOGBOPTester(
        checkpoint_path="/path/to/checkpoint.ckpt",
        templates_dir="/path/to/templates",
        root_dir="/path/to/bop/datasets",
        dataset_name="ycbv",
        output_dir="./test_results",
        test_setting="localization",
    )
    
    results = tester.run_test()
    print(f"Completed: {len(results)} results")
