"""
Test Suite for VPOG TemplateManager

Comprehensive tests for template loading, caching, and selection.

Run with:
    cd /data/home/ssaricha/gigapose
    PYTHONPATH=/data/home/ssaricha/gigapose:$PYTHONPATH python vpog/inference/test_template_manager.py
"""

import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from vpog.inference import TemplateManager, create_template_manager


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"=== {title} ===")
    print('=' * 60)


def create_random_pose() -> np.ndarray:
    """Create a random 6D pose."""
    R = Rotation.random().as_matrix()
    t = np.random.randn(3) * 0.5
    t[2] += 1.0  # Ensure positive Z
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def test_initialization():
    """Test TemplateManager initialization."""
    print_section("Test 1: Initialization")
    
    templates_dir = project_root / "datasets" / "templates"
    
    # Test all mode
    try:
        manager = TemplateManager(
            templates_dir=templates_dir,
            dataset_name="gso",
            mode="all",
            level_templates=1,
        )
        print(f"✓ Initialized in 'all' mode")
        print(f"  Templates per object: {manager.num_templates}")
        assert manager.num_templates == 162, f"Expected 162 templates, got {manager.num_templates}"
    except Exception as e:
        print(f"✗ Failed to initialize in 'all' mode: {e}")
        return False
    
    # Test subset mode
    try:
        manager = TemplateManager(
            templates_dir=templates_dir,
            dataset_name="gso",
            mode="subset",
            num_positive=4,
            num_negative=2,
        )
        print(f"✓ Initialized in 'subset' mode")
        print(f"  Templates per query: {manager.num_templates}")
        assert manager.num_templates == 6, f"Expected 6 templates (4+2), got {manager.num_templates}"
    except Exception as e:
        print(f"✗ Failed to initialize in 'subset' mode: {e}")
        return False
    
    # Test invalid directory
    try:
        manager = TemplateManager(
            templates_dir="/nonexistent/path",
            dataset_name="gso",
            mode="all",
        )
        print(f"✗ Should have raised error for invalid directory")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised error for invalid directory")
    
    return True


def test_all_mode_loading():
    """Test loading all templates for an object."""
    print_section("Test 2: All Mode Loading")
    
    templates_dir = project_root / "datasets" / "templates"
    manager = TemplateManager(
        templates_dir=templates_dir,
        dataset_name="gso",
        mode="all",
        cache_size=3,
    )
    
    # Load templates for object 733
    try:
        data = manager.load_object_templates("000733")
        print(f"✓ Loaded templates for object 733")
        
        # Check shapes
        assert data["images"].shape[0] == 162, f"Expected 162 images"
        assert data["images"].shape[3] == 3, f"Expected RGB images"
        assert data["masks"].shape[0] == 162, f"Expected 162 masks"
        assert data["poses"].shape == (162, 4, 4), f"Expected (162, 4, 4) poses"
        assert len(data["template_indices"]) == 162, f"Expected 162 indices"
        
        print(f"  Images: {data['images'].shape} {data['images'].dtype}")
        print(f"  Masks: {data['masks'].shape} {data['masks'].dtype}")
        print(f"  Poses: {data['poses'].shape} {data['poses'].dtype}")
        print(f"  Indices: {len(data['template_indices'])}")
        
        # Check data types
        assert isinstance(data["images"], torch.Tensor), "Images should be tensor"
        assert isinstance(data["masks"], torch.Tensor), "Masks should be tensor"
        assert isinstance(data["poses"], torch.Tensor), "Poses should be tensor"
        
        # Check value ranges
        assert data["images"].min() >= 0 and data["images"].max() <= 255, "RGB values out of range"
        assert data["masks"].min() >= 0 and data["masks"].max() <= 1, "Mask values out of range"
        
        print(f"  ✓ All shapes and types correct")
        
    except Exception as e:
        print(f"✗ Failed to load templates: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_subset_mode_selection():
    """Test template selection in subset mode."""
    print_section("Test 3: Subset Mode Selection")
    
    templates_dir = project_root / "datasets" / "templates"
    manager = TemplateManager(
        templates_dir=templates_dir,
        dataset_name="gso",
        mode="subset",
        num_positive=4,
        num_negative=2,
    )
    
    # Create query pose
    query_pose = create_random_pose()
    
    try:
        data = manager.load_object_templates("000733", query_pose=query_pose)
        print(f"✓ Selected templates for query pose")
        
        # Check counts
        assert data["images"].shape[0] == 6, f"Expected 6 templates (4+2)"
        assert len(data["template_indices"]) == 6, f"Expected 6 indices"
        assert len(data["template_types"]) == 6, f"Expected 6 type labels"
        
        # Check types (4 positives, 2 negatives)
        num_pos = (data["template_types"] == 0).sum().item()
        num_neg = (data["template_types"] == 1).sum().item()
        assert num_pos == 4, f"Expected 4 positives, got {num_pos}"
        assert num_neg == 2, f"Expected 2 negatives, got {num_neg}"
        
        print(f"  Selected indices: {data['template_indices']}")
        print(f"  Template types: {data['template_types']} (0=pos, 1=neg)")
        print(f"  d_ref: {data['d_ref']}")
        
        # Check d_ref
        assert "d_ref" in data, "d_ref should be returned"
        assert data["d_ref"].shape == (3,), "d_ref should be 3D vector"
        d_ref_norm = torch.norm(data["d_ref"]).item()
        assert abs(d_ref_norm - 1.0) < 1e-5, f"d_ref should be unit vector, norm={d_ref_norm}"
        
        print(f"  ✓ Selection correct (4 pos + 2 neg, d_ref unit vector)")
        
    except Exception as e:
        print(f"✗ Failed template selection: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_cache_functionality():
    """Test template caching mechanism."""
    print_section("Test 4: Cache Functionality")
    
    templates_dir = project_root / "datasets" / "templates"
    manager = TemplateManager(
        templates_dir=templates_dir,
        dataset_name="gso",
        mode="all",
        cache_size=2,  # Small cache for testing
    )
    
    # Get available objects
    objects = manager.get_available_objects()
    if len(objects) < 3:
        print(f"✗ Need at least 3 objects for cache test, found {len(objects)}")
        return False
    
    test_objects = objects[:3]
    print(f"Testing with objects: {test_objects}")
    
    # Load first object
    try:
        data1 = manager.load_object_templates(test_objects[0])
        cache_info = manager.get_cache_info()
        assert cache_info["cached_objects"] == 1, "Should have 1 cached"
        print(f"✓ Loaded {test_objects[0]}, cache: {cache_info['cached_objects']}")
        
        # Load second object
        data2 = manager.load_object_templates(test_objects[1])
        cache_info = manager.get_cache_info()
        assert cache_info["cached_objects"] == 2, "Should have 2 cached"
        print(f"✓ Loaded {test_objects[1]}, cache: {cache_info['cached_objects']}")
        
        # Load third object (should evict first)
        data3 = manager.load_object_templates(test_objects[2])
        cache_info = manager.get_cache_info()
        assert cache_info["cached_objects"] == 2, "Should still have 2 cached (LRU)"
        assert test_objects[0] not in cache_info["cache_order"], f"{test_objects[0]} should be evicted"
        print(f"✓ Loaded {test_objects[2]}, cache: {cache_info['cached_objects']}")
        print(f"  Cache order (LRU): {cache_info['cache_order']}")
        
        # Re-load first object (should be from disk, not cache)
        data1_reload = manager.load_object_templates(test_objects[0])
        cache_info = manager.get_cache_info()
        assert cache_info["cached_objects"] == 2, "Should still have 2 cached"
        assert test_objects[0] in cache_info["cache_order"], f"{test_objects[0]} should be in cache"
        print(f"✓ Re-loaded {test_objects[0]}, cache: {cache_info['cache_order']}")
        
        # Clear cache
        manager.clear_cache()
        cache_info = manager.get_cache_info()
        assert cache_info["cached_objects"] == 0, "Cache should be empty"
        print(f"✓ Cache cleared: {cache_info['cached_objects']} objects")
        
    except Exception as e:
        print(f"✗ Cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_preloading():
    """Test preloading multiple objects."""
    print_section("Test 5: Preloading")
    
    templates_dir = project_root / "datasets" / "templates"
    manager = TemplateManager(
        templates_dir=templates_dir,
        dataset_name="gso",
        mode="all",
        cache_size=5,
    )
    
    objects = manager.get_available_objects()
    if len(objects) < 3:
        print(f"✗ Need at least 3 objects, found {len(objects)}")
        return False
    
    preload_objects = objects[:3]
    print(f"Preloading: {preload_objects}")
    
    try:
        manager.preload_objects(preload_objects)
        cache_info = manager.get_cache_info()
        
        assert cache_info["cached_objects"] == 3, f"Expected 3 cached, got {cache_info['cached_objects']}"
        print(f"✓ Preloaded {cache_info['cached_objects']} objects")
        print(f"  Cached: {cache_info['cache_order']}")
        
    except Exception as e:
        print(f"✗ Preloading failed: {e}")
        return False
    
    return True


def test_batch_loading():
    """Test loading templates for multiple queries."""
    print_section("Test 6: Batch Loading (Subset Mode)")
    
    templates_dir = project_root / "datasets" / "templates"
    manager = TemplateManager(
        templates_dir=templates_dir,
        dataset_name="gso",
        mode="subset",
        num_positive=4,
        num_negative=2,
    )
    
    # Create multiple query poses
    num_queries = 3
    query_poses = [create_random_pose() for _ in range(num_queries)]
    
    try:
        # Load for each query
        all_data = []
        for i, query_pose in enumerate(query_poses):
            data = manager.load_object_templates("000733", query_pose=query_pose)
            all_data.append(data)
            print(f"  Query {i}: indices={data['template_indices']}")
        
        # Verify different selections for different poses
        indices_sets = [set(data["template_indices"].cpu().numpy()) for data in all_data]
        
        # At least some should be different
        all_same = all(s == indices_sets[0] for s in indices_sets)
        if all_same:
            print(f"  ⚠ Warning: All queries selected same templates (could be coincidence)")
        else:
            print(f"✓ Different queries selected different templates")
        
        print(f"✓ Batch loading successful for {num_queries} queries")
        
    except Exception as e:
        print(f"✗ Batch loading failed: {e}")
        return False
    
    return True


def test_error_handling():
    """Test error handling for invalid inputs."""
    print_section("Test 7: Error Handling")
    
    templates_dir = project_root / "datasets" / "templates"
    manager = TemplateManager(
        templates_dir=templates_dir,
        dataset_name="gso",
        mode="all",
    )
    
    # Test nonexistent object
    try:
        data = manager.load_object_templates("999999")
        print(f"✗ Should have raised error for nonexistent object")
        return False
    except ValueError:
        print(f"✓ Correctly raised error for nonexistent object")
    except Exception as e:
        print(f"  Note: Different error raised: {type(e).__name__}")
    
    # Test subset mode without query pose
    manager_subset = TemplateManager(
        templates_dir=templates_dir,
        dataset_name="gso",
        mode="subset",
        num_positive=4,
        num_negative=2,
    )
    
    try:
        data = manager_subset.load_object_templates("000733")  # Missing query_pose
        print(f"✗ Should have raised error for missing query_pose")
        return False
    except ValueError:
        print(f"✓ Correctly raised error for missing query_pose in subset mode")
    
    return True


def test_factory_function():
    """Test create_template_manager factory function."""
    print_section("Test 8: Factory Function")
    
    templates_dir = project_root / "datasets" / "templates"
    
    try:
        manager = create_template_manager(
            templates_dir=str(templates_dir),
            dataset_name="gso",
            mode="all",
            cache_size=5,
        )
        print(f"✓ Factory function created manager")
        print(f"  Mode: {manager.mode}")
        print(f"  Dataset: {manager.dataset_name}")
        print(f"  Templates: {manager.num_templates}")
        
    except Exception as e:
        print(f"✗ Factory function failed: {e}")
        return False
    
    return True


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 60)
    print("VPOG TemplateManager Test Suite")
    print("=" * 60)
    
    tests = [
        ("Initialization", test_initialization),
        ("All Mode Loading", test_all_mode_loading),
        ("Subset Mode Selection", test_subset_mode_selection),
        ("Cache Functionality", test_cache_functionality),
        ("Preloading", test_preloading),
        ("Batch Loading", test_batch_loading),
        ("Error Handling", test_error_handling),
        ("Factory Function", test_factory_function),
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
