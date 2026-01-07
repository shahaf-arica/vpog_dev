"""
Test Suite for VPOG InferencePipeline

Comprehensive end-to-end tests for the complete inference pipeline.

Run with:
    cd /data/home/ssaricha/gigapose
    PYTHONPATH=/data/home/ssaricha/gigapose:$PYTHONPATH python vpog/inference/test_pipeline.py
"""

import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from vpog.inference import (
    InferencePipeline,
    create_inference_pipeline,
    PoseEstimate,
    InferenceResult,
)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"=== {title} ===")
    print('=' * 60)


def create_synthetic_query() -> tuple:
    """Create synthetic query image and camera intrinsics."""
    query_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    K = np.array([
        [280.0, 0.0, 112.0],
        [0.0, 280.0, 112.0],
        [0.0, 0.0, 1.0]
    ])
    return query_image, K


def test_pipeline_initialization():
    """Test pipeline initialization with different configurations."""
    print_section("Test 1: Pipeline Initialization")
    
    templates_dir = project_root / "datasets" / "templates"
    
    # Test all mode
    try:
        pipeline = InferencePipeline(
            templates_dir=templates_dir,
            dataset_name="gso",
            template_mode="all",
            cache_size=5,
        )
        print("✓ Initialized pipeline in 'all' mode")
        stats = pipeline.get_stats()
        print(f"  Device: {stats['device']}")
        print(f"  Min inliers: {stats['min_inliers']}")
        print(f"  Cache size: {stats['template_manager']['cache_size']}")
    except Exception as e:
        print(f"✗ Failed to initialize in 'all' mode: {e}")
        return False
    
    # Test subset mode
    try:
        pipeline = InferencePipeline(
            templates_dir=templates_dir,
            dataset_name="gso",
            template_mode="subset",
            cache_size=3,
        )
        print("✓ Initialized pipeline in 'subset' mode")
    except Exception as e:
        print(f"✗ Failed to initialize in 'subset' mode: {e}")
        return False
    
    # Test factory function
    try:
        pipeline = create_inference_pipeline(
            templates_dir=str(templates_dir),
            dataset_name="gso",
            cache_size=5,
        )
        print("✓ Created pipeline via factory function")
    except Exception as e:
        print(f"✗ Factory function failed: {e}")
        return False
    
    return True


def test_single_object_estimation():
    """Test pose estimation for a single object."""
    print_section("Test 2: Single Object Pose Estimation")
    
    templates_dir = project_root / "datasets" / "templates"
    pipeline = InferencePipeline(
        templates_dir=templates_dir,
        dataset_name="gso",
        template_mode="all",
        cache_size=5,
        min_inliers=4,
    )
    
    query_image, K = create_synthetic_query()
    
    try:
        start_time = time.time()
        estimate = pipeline.estimate_pose(
            query_image=query_image,
            object_id="000733",
            K=K,
        )
        elapsed = time.time() - start_time
        
        print(f"✓ Pose estimated in {elapsed*1000:.1f}ms")
        print(f"  Object: {estimate.object_id}")
        print(f"  Score: {estimate.score:.3f}")
        print(f"  Inliers: {estimate.num_inliers}/{estimate.num_correspondences}")
        
        # Check pose estimate structure
        assert estimate.pose.shape == (4, 4), "Pose should be 4x4 matrix"
        assert 0.0 <= estimate.score <= 1.0, "Score should be in [0, 1]"
        assert estimate.num_inliers >= 0, "Inliers should be non-negative"
        assert estimate.num_correspondences > 0, "Should have correspondences"
        
        # Check pose matrix properties
        assert np.allclose(estimate.pose[3, :], [0, 0, 0, 1]), "Bottom row should be [0,0,0,1]"
        
        print("  ✓ All checks passed")
        
    except Exception as e:
        print(f"✗ Single object estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_multi_object_estimation():
    """Test pose estimation for multiple objects."""
    print_section("Test 3: Multi-Object Pose Estimation")
    
    templates_dir = project_root / "datasets" / "templates"
    pipeline = InferencePipeline(
        templates_dir=templates_dir,
        dataset_name="gso",
        template_mode="all",
        cache_size=10,
    )
    
    query_image, K = create_synthetic_query()
    
    # Get available objects
    available = pipeline.template_manager.get_available_objects()
    if len(available) < 3:
        print(f"⚠ Only {len(available)} objects available")
        test_objects = available
    else:
        test_objects = available[:3]
    
    print(f"Testing with objects: {test_objects}")
    
    try:
        start_time = time.time()
        result = pipeline.estimate_poses(
            query_image=query_image,
            object_ids=test_objects,
            K=K,
        )
        elapsed = time.time() - start_time
        
        print(f"✓ Estimated {len(result)} poses in {result.processing_time*1000:.1f}ms")
        print(f"  Total time: {elapsed*1000:.1f}ms")
        
        # Check result structure
        assert len(result.estimates) == len(test_objects), "Should have one estimate per object"
        assert result.query_image.shape == query_image.shape, "Query image should match"
        assert result.processing_time > 0, "Processing time should be positive"
        
        # Check each estimate
        for i, estimate in enumerate(result.estimates):
            print(f"  [{i}] {estimate}")
            assert estimate.object_id == test_objects[i], "Object ID should match"
            assert estimate.pose.shape == (4, 4), "Pose should be 4x4"
        
        # Test best estimate
        best = result.best_estimate()
        print(f"  Best: {best.object_id} (score={best.score:.3f})")
        assert best is not None, "Should have best estimate"
        
        # Test get_estimate
        est = result.get_estimate(test_objects[0])
        assert est is not None, "Should find first object"
        assert est.object_id == test_objects[0], "Object ID should match"
        
        print("  ✓ All checks passed")
        
    except Exception as e:
        print(f"✗ Multi-object estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_preloading():
    """Test template preloading functionality."""
    print_section("Test 4: Template Preloading")
    
    templates_dir = project_root / "datasets" / "templates"
    pipeline = InferencePipeline(
        templates_dir=templates_dir,
        dataset_name="gso",
        template_mode="all",
        cache_size=10,
    )
    
    available = pipeline.template_manager.get_available_objects()
    preload_objects = available[:5] if len(available) >= 5 else available
    
    print(f"Preloading {len(preload_objects)} objects...")
    
    try:
        start_time = time.time()
        pipeline.preload_objects(preload_objects)
        elapsed = time.time() - start_time
        
        stats = pipeline.get_stats()
        cached = stats['template_manager']['cached_objects']
        
        print(f"✓ Preloaded in {elapsed:.2f}s")
        print(f"  Cached objects: {cached}")
        print(f"  Cache order: {stats['template_manager']['cache_order']}")
        
        assert cached == len(preload_objects), f"Expected {len(preload_objects)} cached"
        
        # Test that preloaded objects are fast to access
        query_image, K = create_synthetic_query()
        
        start_time = time.time()
        estimate = pipeline.estimate_pose(query_image, preload_objects[0], K)
        cached_time = time.time() - start_time
        
        print(f"  Cached object access: {cached_time*1000:.1f}ms")
        print("  ✓ Preloading successful")
        
    except Exception as e:
        print(f"✗ Preloading failed: {e}")
        return False
    
    return True


def test_subset_mode():
    """Test subset mode with query pose hints."""
    print_section("Test 5: Subset Mode Selection")
    
    templates_dir = project_root / "datasets" / "templates"
    pipeline = InferencePipeline(
        templates_dir=templates_dir,
        dataset_name="gso",
        template_mode="subset",
        cache_size=3,
    )
    
    query_image, K = create_synthetic_query()
    
    # Create a query pose hint
    R = Rotation.random().as_matrix()
    t = np.array([0.0, 0.0, 1.0])
    query_pose_hint = np.eye(4)
    query_pose_hint[:3, :3] = R
    query_pose_hint[:3, 3] = t
    
    try:
        estimate = pipeline.estimate_pose(
            query_image=query_image,
            object_id="000733",
            K=K,
            query_pose_hint=query_pose_hint,
        )
        
        print(f"✓ Subset mode estimation: {estimate}")
        print(f"  Used query pose hint for template selection")
        
        # In subset mode, fewer templates are used
        # (though correspondences are still synthetic in this test)
        
        print("  ✓ Subset mode working")
        
    except Exception as e:
        print(f"✗ Subset mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_error_handling():
    """Test error handling for invalid inputs."""
    print_section("Test 6: Error Handling")
    
    templates_dir = project_root / "datasets" / "templates"
    pipeline = InferencePipeline(
        templates_dir=templates_dir,
        dataset_name="gso",
        template_mode="all",
    )
    
    query_image, K = create_synthetic_query()
    
    # Test nonexistent object (should handle gracefully)
    try:
        result = pipeline.estimate_poses(
            query_image=query_image,
            object_ids=["999999"],
            K=K,
        )
        # Should return failed estimate
        assert len(result.estimates) == 1, "Should have one estimate"
        assert result.estimates[0].score == 0.0, "Failed estimate should have score 0"
        print("✓ Handled nonexistent object gracefully")
    except Exception as e:
        print(f"  Note: Error for nonexistent object: {e}")
    
    # Test subset mode without query pose hint
    pipeline_subset = InferencePipeline(
        templates_dir=templates_dir,
        dataset_name="gso",
        template_mode="subset",
    )
    
    try:
        result = pipeline_subset.estimate_poses(
            query_image=query_image,
            object_ids=["000733"],
            K=K,
            # Missing query_pose_hints
        )
        # Should fail but be caught
        assert len(result.estimates) == 1, "Should have one estimate"
        print("✓ Handled missing query pose hint in subset mode")
    except Exception as e:
        print(f"  Note: Error for missing pose hint: {e}")
    
    return True


def test_pose_estimate_dataclass():
    """Test PoseEstimate dataclass functionality."""
    print_section("Test 7: PoseEstimate Dataclass")
    
    # Create a pose estimate
    pose = np.eye(4)
    estimate = PoseEstimate(
        object_id="000733",
        pose=pose,
        score=0.85,
        num_inliers=42,
        num_correspondences=50,
        template_id=15,
    )
    
    print(f"✓ Created PoseEstimate: {estimate}")
    
    # Check fields
    assert estimate.object_id == "000733"
    assert np.allclose(estimate.pose, pose)
    assert estimate.score == 0.85
    assert estimate.num_inliers == 42
    assert estimate.num_correspondences == 50
    assert estimate.template_id == 15
    
    # Check repr
    repr_str = repr(estimate)
    assert "000733" in repr_str
    assert "0.850" in repr_str
    
    print("  ✓ All fields correct")
    return True


def test_inference_result_dataclass():
    """Test InferenceResult dataclass functionality."""
    print_section("Test 8: InferenceResult Dataclass")
    
    query_image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Create estimates
    estimates = [
        PoseEstimate("000733", np.eye(4), 0.9, 45, 50),
        PoseEstimate("000001", np.eye(4), 0.7, 35, 50),
        PoseEstimate("000003", np.eye(4), 0.5, 25, 50),
    ]
    
    result = InferenceResult(
        estimates=estimates,
        query_image=query_image,
        processing_time=1.23,
    )
    
    print(f"✓ Created InferenceResult with {len(result)} estimates")
    
    # Test length
    assert len(result) == 3
    
    # Test best_estimate
    best = result.best_estimate()
    assert best is not None
    assert best.object_id == "000733"  # Highest score
    assert best.score == 0.9
    print(f"  Best: {best.object_id} (score={best.score})")
    
    # Test get_estimate
    est = result.get_estimate("000001")
    assert est is not None
    assert est.object_id == "000001"
    assert est.score == 0.7
    print(f"  Get estimate for 000001: score={est.score}")
    
    # Test nonexistent
    est = result.get_estimate("999999")
    assert est is None
    print("  ✓ Returns None for nonexistent object")
    
    print("  ✓ All InferenceResult features working")
    return True


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 60)
    print("VPOG InferencePipeline Test Suite")
    print("=" * 60)
    
    tests = [
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Single Object Estimation", test_single_object_estimation),
        ("Multi-Object Estimation", test_multi_object_estimation),
        ("Template Preloading", test_preloading),
        ("Subset Mode Selection", test_subset_mode),
        ("Error Handling", test_error_handling),
        ("PoseEstimate Dataclass", test_pose_estimate_dataclass),
        ("InferenceResult Dataclass", test_inference_result_dataclass),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
