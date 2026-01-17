"""
BOP Evaluation Runner

Runs official BOP toolkit evaluation and aggregates results.
Calls bop_toolkit/scripts/eval_bop19_pose.py for official metrics.

Author: VPOG Team
Date: January 2026
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from bop_toolkit_lib import inout
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BOPEvaluator:
    """
    BOP Evaluation Runner.
    
    Calls official BOP toolkit scripts to compute:
    - VSD (Visible Surface Discrepancy)
    - MSSD (Maximum Symmetry-Aware Surface Distance)
    - MSPD (Maximum Symmetry-Aware Projection Distance)
    - AR (Average Recall)
    """
    
    def __init__(
        self,
        results_dir: str,
        bop_toolkit_path: Optional[str] = None,
    ):
        """
        Args:
            results_dir: Directory to save evaluation results
            bop_toolkit_path: Path to bop_toolkit (default: assume in PATH)
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # BOP toolkit path
        if bop_toolkit_path is None:
            # Assume bop_toolkit is in the gigapose root
            self.bop_toolkit_path = Path(__file__).parent.parent / "bop_toolkit"
        else:
            self.bop_toolkit_path = Path(bop_toolkit_path)
        
        if not self.bop_toolkit_path.exists():
            logger.warning(f"BOP toolkit not found at {self.bop_toolkit_path}")
    
    def evaluate_csv(
        self,
        csv_path: str,
        dataset_name: str,
        renderer_type: str = "vispy",
    ) -> Optional[Dict]:
        """
        Evaluate a single CSV file using BOP toolkit.
        
        Args:
            csv_path: Path to CSV file with predictions
            dataset_name: BOP dataset name
            renderer_type: Renderer for VSD ("vispy", "python", or "cpp")
            
        Returns:
            Dictionary with BOP scores or None if evaluation failed
        """
        csv_path = Path(csv_path)
        
        # Check if dataset has GT (some don't: hb, itodd)
        if dataset_name in ["hb", "itodd"]:
            logger.info(f"Dataset {dataset_name} has no GT, skipping evaluation")
            # Just copy the CSV to results directory
            import shutil
            dest = self.results_dir / csv_path.name
            shutil.copy(csv_path, dest)
            logger.info(f"Copied predictions to {dest}")
            return None
        
        input_dir = csv_path.parent
        file_name = csv_path.name
        
        # Run BOP toolkit evaluation
        eval_script = self.bop_toolkit_path / "scripts" / "eval_bop19_pose.py"
        
        if not eval_script.exists():
            logger.error(f"Evaluation script not found: {eval_script}")
            return None
        
        command = [
            "python",
            str(eval_script),
            f"--renderer_type={renderer_type}",
            f"--results_path={input_dir}",
            f"--eval_path={input_dir}",
            f"--result_filenames={file_name}",
        ]
        
        logger.info(f"Running BOP evaluation: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Evaluation failed with code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return None
            
            logger.info("Evaluation completed successfully")
            
            # Load results
            result_path = input_dir / file_name.replace(".csv", "") / "scores_bop19.json"
            
            if not result_path.exists():
                logger.error(f"Results file not found: {result_path}")
                return None
            
            scores = inout.load_json(result_path)
            
            # Copy results to main results directory
            dest_path = self.results_dir / f"{dataset_name}_scores.json"
            import shutil
            shutil.copy(result_path, dest_path)
            logger.info(f"Saved scores to {dest_path}")
            
            return scores
            
        except subprocess.TimeoutExpired:
            logger.error("Evaluation timed out after 1 hour")
            return None
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None
    
    def evaluate_multiple(
        self,
        csv_paths: List[str],
        dataset_names: List[str],
        renderer_type: str = "vispy",
    ) -> Dict[str, Dict]:
        """
        Evaluate multiple CSV files.
        
        Args:
            csv_paths: List of CSV file paths
            dataset_names: Corresponding dataset names
            renderer_type: Renderer for VSD
            
        Returns:
            Dictionary mapping dataset names to scores
        """
        all_scores = {}
        
        for csv_path, dataset_name in zip(csv_paths, dataset_names):
            logger.info(f"Evaluating {dataset_name}")
            scores = self.evaluate_csv(csv_path, dataset_name, renderer_type)
            
            if scores is not None:
                all_scores[dataset_name] = scores
                
                # Log key metrics
                ar = scores.get("bop19_average_recall", 0.0)
                logger.info(f"{dataset_name} AR: {ar:.4f}")
        
        # Save aggregated results
        if all_scores:
            summary_path = self.results_dir / "all_scores.json"
            inout.save_json(summary_path, all_scores)
            logger.info(f"Saved aggregated scores to {summary_path}")
        
        return all_scores
    
    def print_summary(self, scores: Dict[str, Dict]):
        """
        Print summary of evaluation results.
        
        Args:
            scores: Dictionary mapping dataset names to score dicts
        """
        logger.info("=" * 80)
        logger.info("BOP Evaluation Summary")
        logger.info("=" * 80)
        
        for dataset_name, dataset_scores in scores.items():
            ar = dataset_scores.get("bop19_average_recall", 0.0)
            vsd = dataset_scores.get("bop19_average_recall_vsd", 0.0)
            mssd = dataset_scores.get("bop19_average_recall_mssd", 0.0)
            mspd = dataset_scores.get("bop19_average_recall_mspd", 0.0)
            
            logger.info(f"{dataset_name:10s}: AR={ar:.4f}, VSD={vsd:.4f}, MSSD={mssd:.4f}, MSPD={mspd:.4f}")
        
        # Compute average across datasets
        if scores:
            avg_ar = sum(s.get("bop19_average_recall", 0.0) for s in scores.values()) / len(scores)
            logger.info(f"{'Average':10s}: AR={avg_ar:.4f}")
        
        logger.info("=" * 80)


if __name__ == "__main__":
    # Test evaluator
    from src.utils.logging import setup_logger
    setup_logger()
    
    evaluator = BOPEvaluator(results_dir="./test_results/eval")
    
    # Evaluate single dataset
    scores = evaluator.evaluate_csv(
        csv_path="./test_results/vpog-pbrreal-rgb-mmodel_ycbv-test_test.csv",
        dataset_name="ycbv",
    )
    
    if scores:
        evaluator.print_summary({"ycbv": scores})
