"""
BOP Testing Module for VPOG

This module handles BOP Challenge testing with:
- BOP test dataloader with default detections
- VPOG inference orchestration
- BOP format conversion
- Official BOP evaluation

Directory structure:
    dataloader.py - BOP test data loading
    tester.py - Main test orchestration
    bop_formatter.py - Format conversion to BOP CSV/JSON
    evaluator.py - BOP toolkit evaluation
    config/ - Test configurations
    scripts/ - Test execution scripts
"""

__version__ = "1.0.0"
