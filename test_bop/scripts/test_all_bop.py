#!/usr/bin/env python
"""
Test VPOG on all BOP core datasets.

Runs testing on all 7 BOP core datasets in sequence:
- lmo, tudl, icbin, tless, ycbv (with GT)
- itodd, hb (no GT, predictions only)

Usage:
    python test_bop/scripts/test_all_bop.py --config test_bop/config/test.yaml
    
    # Or with overrides:
    python test_bop/scripts/test_all_bop.py \
        --checkpoint /path/to/ckpt.ckpt \
        --output ./results_all
"""

import argparse
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import yaml
from src.utils.logging import get_logger, setup_logger
from test_bop.tester import VPOGBOPTester
from test_bop.evaluator import BOPEvaluator

logger = get_logger(__name__)


# BOP core datasets
BOP_CORE_DATASETS = [
    "lmo",
    "tudl",
    "icbin",
    "tless",
    "ycbv",
    "itodd",  # No GT
    "hb",  # No GT
]


def parse_args():
    parser = argparse.ArgumentParser(description="Test VPOG on all BOP datasets")
    
    parser.add_argument(
        "--config",
        type=str,
        default="test_bop/config/test.yaml",
        help="Path to config file",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--output", type=str, help="Output base directory")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to test (default: all BOP core)",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    
    return parser.parse_args()


def load_config(config_path: str, args) -> dict:
    """Load config and apply command-line overrides."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if args.checkpoint:
        config['checkpoint_path'] = args.checkpoint
    if args.output:
        config['output_dir'] = args.output
    if args.device:
        config['inference']['device'] = args.device
    if args.skip_eval:
        config['evaluate'] = False
    
    return config


def test_single_dataset(
    config: dict,
    dataset_name: str,
    output_base: Path,
) -> tuple:
    """
    Test on single dataset.
    
    Returns:
        (csv_path, scores) tuple
    """
    logger.info("=" * 80)
    logger.info(f"Testing on {dataset_name}")
    logger.info("=" * 80)
    
    # Create dataset-specific output directory
    output_dir = output_base / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # Create tester
    tester = VPOGBOPTester(
        checkpoint_path=config['checkpoint_path'],
        templates_dir=config['templates_dir'],
        root_dir=config['root_dir'],
        dataset_name=dataset_name,
        output_dir=str(output_dir),
        test_setting=config['test_setting'],
        device=config['inference']['device'],
        batch_size=config['batch_size'],
        **config['inference'],
    )
    
    # Run test
    results = tester.run_test()
    
    elapsed = time.time() - start_time
    logger.info(f"{dataset_name}: {len(results)} predictions in {elapsed:.1f}s")
    
    # Find CSV file
    csv_files = list(output_dir.glob("*.csv"))
    csv_path = csv_files[0] if csv_files else None
    
    return csv_path, None


def main():
    # Setup logging
    setup_logger()
    
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config, args)
    
    # Determine datasets to test
    datasets = args.datasets if args.datasets else BOP_CORE_DATASETS
    
    logger.info("=" * 80)
    logger.info("VPOG BOP Testing - All Datasets")
    logger.info("=" * 80)
    logger.info(f"Datasets: {', '.join(datasets)}")
    logger.info(f"Checkpoint: {config['checkpoint_path']}")
    logger.info(f"Output: {config['output_dir']}")
    logger.info("=" * 80)
    
    output_base = Path(config['output_dir'])
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Test each dataset
    csv_paths = []
    dataset_names = []
    
    for dataset_name in datasets:
        try:
            csv_path, _ = test_single_dataset(config, dataset_name, output_base)
            if csv_path:
                csv_paths.append(str(csv_path))
                dataset_names.append(dataset_name)
        except Exception as e:
            logger.error(f"Failed to test {dataset_name}: {e}")
            continue
    
    logger.info("=" * 80)
    logger.info(f"Completed testing on {len(dataset_names)} datasets")
    logger.info("=" * 80)
    
    # Run evaluation if requested
    if config.get('evaluate', True) and csv_paths:
        logger.info("Starting BOP evaluation on all datasets...")
        
        evaluator = BOPEvaluator(
            results_dir=output_base / "eval",
        )
        
        all_scores = evaluator.evaluate_multiple(
            csv_paths=csv_paths,
            dataset_names=dataset_names,
            renderer_type=config.get('renderer_type', 'vispy'),
        )
        
        if all_scores:
            evaluator.print_summary(all_scores)
    
    logger.info("All done!")


if __name__ == "__main__":
    main()
