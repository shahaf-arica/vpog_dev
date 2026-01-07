#!/usr/bin/env python3
"""
Run all VPOG inference pipeline tests.

Usage:
    python run_all_tests.py                  # Run all tests
    python run_all_tests.py --stage 1        # Run Stage 1 only
    python run_all_tests.py --stage 1 2 3    # Run Stages 1, 2, 3
    python run_all_tests.py --verbose        # Verbose output
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time


# Test configurations
TESTS = {
    1: {
        "name": "Stage 1: Correspondence Builder",
        "file": "vpog/inference/test_correspondence.py",
        "expected": 6,
    },
    2: {
        "name": "Stage 2: Pose Solvers",
        "file": "vpog/inference/test_pose_solvers.py",
        "expected": 7,  # 5 core + 2 EPro-PnP (optional)
    },
    3: {
        "name": "Stage 3: Template Manager",
        "file": "vpog/inference/test_template_manager.py",
        "expected": 8,
    },
    4: {
        "name": "Stage 4: Full Pipeline",
        "file": "vpog/inference/test_pipeline.py",
        "expected": 8,
    },
}


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*60)
    print(text)
    print("="*60)


def print_subheader(text):
    """Print formatted subheader."""
    print("\n" + "-"*60)
    print(text)
    print("-"*60)


def run_test(stage_num, verbose=False):
    """Run a single stage test."""
    config = TESTS[stage_num]
    
    print_subheader(f"Running {config['name']}")
    print(f"File: {config['file']}")
    print(f"Expected tests: {config['expected']}")
    
    # Build command
    project_root = Path(__file__).parent
    test_file = project_root / config['file']
    
    cmd = [
        sys.executable,
        str(test_file),
    ]
    
    # Set PYTHONPATH
    env = {
        **subprocess.os.environ,
        'PYTHONPATH': str(project_root),
    }
    
    # Run test
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        # Parse output
        output = result.stdout + result.stderr
        
        if verbose:
            print("\nTest Output:")
            print(output)
        
        # Check for success indicators
        passed = result.returncode == 0
        
        # Count passed tests
        passed_count = output.count("PASS") if "PASS" in output else 0
        failed_count = output.count("FAIL") if "FAIL" in output else 0
        
        # Print summary
        print(f"\nResult: {'✓ PASSED' if passed else '✗ FAILED'}")
        print(f"Tests: {passed_count} passed, {failed_count} failed")
        print(f"Time: {elapsed:.2f}s")
        
        if not passed and not verbose:
            print("\nError output:")
            print(result.stderr[:500])  # First 500 chars
            print("\n(Use --verbose for full output)")
        
        return passed, passed_count, failed_count
        
    except subprocess.TimeoutExpired:
        print(f"\n✗ TIMEOUT after 5 minutes")
        return False, 0, 0
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return False, 0, 0


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Run VPOG inference pipeline tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4],
        help="Specific stage(s) to test (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed test output",
    )
    
    args = parser.parse_args()
    
    # Determine which stages to run
    stages = args.stage if args.stage else [1, 2, 3, 4]
    
    # Print header
    print_header("VPOG INFERENCE PIPELINE - TEST SUITE")
    print(f"\nRunning {len(stages)} stage(s): {', '.join(map(str, stages))}")
    print(f"Python: {sys.executable}")
    print(f"Verbose: {args.verbose}")
    
    # Run tests
    results = {}
    total_passed = 0
    total_failed = 0
    
    for stage_num in stages:
        passed, num_passed, num_failed = run_test(stage_num, args.verbose)
        results[stage_num] = {
            "passed": passed,
            "num_passed": num_passed,
            "num_failed": num_failed,
        }
        total_passed += num_passed
        total_failed += num_failed
    
    # Print final summary
    print_header("FINAL SUMMARY")
    
    print("\nPer-Stage Results:")
    for stage_num in stages:
        config = TESTS[stage_num]
        result = results[stage_num]
        status = "✓" if result["passed"] else "✗"
        print(f"  {status} Stage {stage_num}: {result['num_passed']}/{config['expected']} tests passed")
    
    print(f"\nOverall:")
    print(f"  Total passed: {total_passed}")
    print(f"  Total failed: {total_failed}")
    
    # Calculate total expected
    total_expected = sum(TESTS[s]["expected"] for s in stages)
    success_rate = (total_passed / total_expected * 100) if total_expected > 0 else 0
    
    print(f"  Success rate: {success_rate:.1f}% ({total_passed}/{total_expected})")
    
    # Overall status
    all_passed = all(r["passed"] for r in results.values())
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nFailed stages:")
        for stage_num, result in results.items():
            if not result["passed"]:
                print(f"  - Stage {stage_num}: {TESTS[stage_num]['name']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
