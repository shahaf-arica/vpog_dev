#!/usr/bin/env python
"""
Test VPOG on a single BOP dataset.

Usage:
    python test_bop/scripts/test_single.py --config test_bop/config/test.yaml --dataset ycbv
    
    # Or override via command line:
    python test_bop/scripts/test_single.py \
        --checkpoint /path/to/ckpt.ckpt \
        --dataset ycbv \
        --output ./results
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import yaml
from src.utils.logging import get_logger, setup_logger
from test_bop.tester import VPOGBOPTester
from test_bop.evaluator import BOPEvaluator

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Test VPOG on BOP dataset")
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="test_bop/config/test.yaml",
        help="Path to config file",
    )
    
    # Override options
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--dataset", type=str, help="Dataset name (ycbv, tless, etc.)")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    
    return parser.parse_args()


def load_config(config_path: str, args) -> dict:
    """Load config and apply command-line overrides."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if args.checkpoint:
        config['checkpoint_path'] = args.checkpoint
    if args.dataset:
        config['dataset_name'] = args.dataset
    if args.output:
        config['output_dir'] = args.output
    if args.device:
        config['inference']['device'] = args.device
    if args.no_eval:
        config['evaluate'] = False
    
    return config


def main():
    # Setup logging
    setup_logger()
    
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config, args)
    
    logger.info("=" * 80)
    logger.info("VPOG BOP Testing - Single Dataset")
    logger.info("=" * 80)
    logger.info(f"Dataset: {config['dataset_name']}")
    logger.info(f"Checkpoint: {config['checkpoint_path']}")
    logger.info(f"Output: {config['output_dir']}")
    logger.info("=" * 80)
    
    # Create tester
    tester = VPOGBOPTester(
        checkpoint_path=config['checkpoint_path'],
        templates_dir=config['templates_dir'],
        root_dir=config['root_dir'],
        dataset_name=config['dataset_name'],
        output_dir=config['output_dir'],
        test_setting=config['test_setting'],
        device=config['inference']['device'],
        batch_size=config['batch_size'],
        # Pass inference kwargs
        **config['inference'],
    )
    
    # Run test
    logger.info("Starting test...")
    results = tester.run_test()
    logger.info(f"Test complete: {len(results)} predictions")
    
    # Run evaluation if requested
    if config.get('evaluate', True):
        logger.info("Starting BOP evaluation...")
        
        evaluator = BOPEvaluator(
            results_dir=Path(config['output_dir']) / "eval",
        )
        
        # Find CSV file
        csv_files = list(Path(config['output_dir']).glob("*.csv"))
        if not csv_files:
            logger.error("No CSV file found for evaluation")
            return
        
        csv_path = csv_files[0]
        logger.info(f"Evaluating: {csv_path}")
        
        scores = evaluator.evaluate_csv(
            csv_path=str(csv_path),
            dataset_name=config['dataset_name'],
            renderer_type=config.get('renderer_type', 'vispy'),
        )
        
        if scores:
            evaluator.print_summary({config['dataset_name']: scores})
        else:
            logger.warning("Evaluation failed or skipped (no GT available)")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
