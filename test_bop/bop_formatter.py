"""
BOP Format Converter

Converts VPOG predictions to BOP Challenge CSV format:
- scene_id, im_id, obj_id, score, R (9 values), t (3 values), time

Author: VPOG Team
Date: January 2026
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BOPFormatter:
    """
    Converts VPOG test results to BOP Challenge format.
    
    Output format (CSV):
        scene_id, im_id, obj_id, score, R (9 values), t (3 values), time
        
    This matches the format used by GigaPose and expected by BOP toolkit.
    """
    
    def __init__(
        self,
        dataset_name: str,
        output_dir: str,
    ):
        """
        Args:
            dataset_name: BOP dataset name
            output_dir: Output directory for formatted results
        """
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_to_csv(
        self,
        results: List[Any],  # List of TestResult objects
        model_name: str = "vpog",
        run_id: str = "test",
    ) -> Path:
        """
        Convert test results to BOP CSV format.
        
        Args:
            results: List of TestResult objects from tester
            model_name: Model name for output filename
            run_id: Run ID for output filename
            
        Returns:
            Path to saved CSV file
        """
        logger.info(f"Converting {len(results)} results to BOP CSV format")
        
        # Prepare data for DataFrame
        rows = []
        for result in results:
            # Flatten rotation matrix (row-major order)
            R_flat = result.R.flatten()
            
            row = {
                'scene_id': int(result.scene_id),
                'im_id': int(result.im_id),
                'obj_id': int(result.obj_id),
                'score': float(result.score),
                'R': ' '.join([f"{v:.6f}" for v in R_flat]),
                't': ' '.join([f"{v:.6f}" for v in result.t]),
                'time': float(result.time),
            }
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Sort by scene_id, im_id, obj_id for consistency
        df = df.sort_values(['scene_id', 'im_id', 'obj_id']).reset_index(drop=True)
        
        # Save to CSV
        # Format: {model_name}-pbrreal-rgb-mmodel_{dataset_name}-test_{run_id}.csv
        csv_filename = f"{model_name}-pbrreal-rgb-mmodel_{self.dataset_name}-test_{run_id}.csv"
        csv_path = self.output_dir / csv_filename
        
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved BOP CSV to: {csv_path}")
        
        return csv_path
    
    def load_from_npz_batches(
        self,
        predictions_dir: str,
    ) -> List[Dict]:
        """
        Load results from .npz batch files.
        
        Args:
            predictions_dir: Directory containing .npz batch files
            
        Returns:
            List of result dictionaries
        """
        predictions_dir = Path(predictions_dir)
        npz_files = sorted(predictions_dir.glob("*.npz"))
        
        logger.info(f"Loading from {len(npz_files)} .npz files")
        
        all_results = []
        
        for npz_file in tqdm(npz_files, desc="Loading batches"):
            data = np.load(npz_file)
            
            batch_size = len(data["scene_id"])
            
            for i in range(batch_size):
                result = {
                    'scene_id': int(data["scene_id"][i]),
                    'im_id': int(data["im_id"][i]),
                    'obj_id': int(data["object_id"][i]),
                    'score': float(data["scores"][i]),
                    'R': data["poses"][i][:3, :3],
                    't': data["poses"][i][:3, 3],
                    'time': float(data["time"][i]),
                }
                all_results.append(result)
        
        logger.info(f"Loaded {len(all_results)} results from batches")
        return all_results
    
    def convert_npz_to_csv(
        self,
        predictions_dir: str,
        model_name: str = "vpog",
        run_id: str = "test",
    ) -> Path:
        """
        Convenience method: Load from .npz batches and convert to CSV.
        
        Args:
            predictions_dir: Directory containing .npz files
            model_name: Model name for output
            run_id: Run ID for output
            
        Returns:
            Path to saved CSV file
        """
        # Load results
        results = self.load_from_npz_batches(predictions_dir)
        
        # Convert to CSV-friendly format
        class ResultWrapper:
            """Simple wrapper to match TestResult interface."""
            def __init__(self, data):
                self.scene_id = data['scene_id']
                self.im_id = data['im_id']
                self.obj_id = data['obj_id']
                self.score = data['score']
                self.R = data['R']
                self.t = data['t']
                self.time = data['time']
        
        wrapped_results = [ResultWrapper(r) for r in results]
        
        # Convert
        return self.convert_to_csv(wrapped_results, model_name, run_id)
    
    def validate_csv(self, csv_path: str) -> bool:
        """
        Validate BOP CSV format.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Check required columns
            required_cols = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns in {csv_path}")
                return False
            
            # Check data types
            for col in ['scene_id', 'im_id', 'obj_id']:
                if not pd.api.types.is_integer_dtype(df[col]):
                    logger.error(f"Column {col} should be integer")
                    return False
            
            for col in ['score', 'time']:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    logger.error(f"Column {col} should be numeric")
                    return False
            
            # Check R and t format (space-separated values)
            for col in ['R', 't']:
                sample = df[col].iloc[0]
                if not isinstance(sample, str):
                    logger.error(f"Column {col} should be string of space-separated values")
                    return False
            
            logger.info(f"CSV validation passed: {csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"CSV validation failed: {e}")
            return False


if __name__ == "__main__":
    # Test formatter
    from src.utils.logging import setup_logger
    setup_logger()
    
    formatter = BOPFormatter(
        dataset_name="ycbv",
        output_dir="./test_results",
    )
    
    # Example: Convert .npz batches to CSV
    csv_path = formatter.convert_npz_to_csv(
        predictions_dir="./test_results/predictions",
        model_name="vpog",
        run_id="test",
    )
    
    # Validate
    formatter.validate_csv(csv_path)
