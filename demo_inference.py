"""
VPOG Inference Demo

Demonstrates the complete VPOG inference pipeline with realistic usage scenarios.

Usage:
    # Basic demo
    python demo_inference.py

    # With specific object
    python demo_inference.py --object 000733

    # Multi-object scene
    python demo_inference.py --objects 000733 000001 000003

    # Fast subset mode
    python demo_inference.py --mode subset --object 000733
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import time
from scipy.spatial.transform import Rotation
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vpog.inference import InferencePipeline, create_inference_pipeline


def create_synthetic_query(size=224):
    """Create a synthetic query image for demo."""
    # Create a colorful synthetic image
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Add some patterns
    for i in range(0, size, 32):
        color = np.random.randint(0, 255, 3)
        cv2.rectangle(img, (i, i), (i+32, i+32), color.tolist(), -1)
    
    # Add some circles
    for _ in range(10):
        center = (np.random.randint(0, size), np.random.randint(0, size))
        radius = np.random.randint(5, 30)
        color = np.random.randint(0, 255, 3)
        cv2.circle(img, center, radius, color.tolist(), -1)
    
    return img


def get_default_camera_intrinsics():
    """Return default camera intrinsics matrix."""
    return np.array([
        [280.0, 0.0, 112.0],
        [0.0, 280.0, 112.0],
        [0.0, 0.0, 1.0]
    ])


def print_pose_info(estimate):
    """Pretty print pose estimate information."""
    print(f"\n{'='*60}")
    print(f"Object: {estimate.object_id}")
    print(f"{'='*60}")
    print(f"Score: {estimate.score:.3f} ({estimate.num_inliers}/{estimate.num_correspondences} inliers)")
    
    # Extract rotation and translation
    R = estimate.pose[:3, :3]
    t = estimate.pose[:3, 3]
    
    # Convert rotation to Euler angles
    euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    
    print(f"\nRotation (Euler XYZ):")
    print(f"  Roll:  {euler[0]:7.2f}°")
    print(f"  Pitch: {euler[1]:7.2f}°")
    print(f"  Yaw:   {euler[2]:7.2f}°")
    
    print(f"\nTranslation:")
    print(f"  X: {t[0]:7.4f} m")
    print(f"  Y: {t[1]:7.4f} m")
    print(f"  Z: {t[2]:7.4f} m")
    
    print(f"\nFull 4×4 Pose Matrix:")
    print(estimate.pose)
    
    if estimate.score > 0.7:
        print(f"\n✓ HIGH CONFIDENCE - Reliable pose estimate")
    elif estimate.score > 0.4:
        print(f"\n⚠ MEDIUM CONFIDENCE - Pose may be approximate")
    else:
        print(f"\n✗ LOW CONFIDENCE - Pose estimate unreliable")


def demo_single_object(pipeline, object_id, K):
    """Demo single object pose estimation."""
    print("\n" + "="*60)
    print("DEMO 1: Single Object Pose Estimation")
    print("="*60)
    
    # Create query image
    print("\n1. Creating synthetic query image...")
    query_img = create_synthetic_query()
    print(f"   ✓ Query image: {query_img.shape}")
    
    # Estimate pose
    print(f"\n2. Estimating pose for object {object_id}...")
    start_time = time.time()
    estimate = pipeline.estimate_pose(query_img, object_id, K)
    elapsed = time.time() - start_time
    
    print(f"   ✓ Completed in {elapsed*1000:.1f}ms")
    
    # Print results
    print_pose_info(estimate)
    
    return estimate


def demo_multi_object(pipeline, object_ids, K):
    """Demo multi-object pose estimation."""
    print("\n" + "="*60)
    print("DEMO 2: Multi-Object Pose Estimation")
    print("="*60)
    
    # Create query image
    print("\n1. Creating synthetic query image...")
    query_img = create_synthetic_query()
    
    # Estimate poses
    print(f"\n2. Estimating poses for {len(object_ids)} objects...")
    print(f"   Objects: {', '.join(object_ids)}")
    
    start_time = time.time()
    result = pipeline.estimate_poses(query_img, object_ids, K)
    elapsed = time.time() - start_time
    
    print(f"   ✓ Completed in {result.processing_time*1000:.1f}ms (wall time: {elapsed*1000:.1f}ms)")
    
    # Print results
    print(f"\n3. Results for {len(result.estimates)} objects:")
    for i, est in enumerate(result.estimates):
        print(f"\n   [{i+1}] {est.object_id}")
        print(f"       Score: {est.score:.3f} ({est.num_inliers} inliers)")
        confidence = "HIGH" if est.score > 0.7 else "MED" if est.score > 0.4 else "LOW"
        print(f"       Confidence: {confidence}")
    
    # Best estimate
    best = result.best_estimate()
    print(f"\n4. Best Estimate: {best.object_id} (score={best.score:.3f})")
    
    return result


def demo_preloading(pipeline):
    """Demo template preloading for performance."""
    print("\n" + "="*60)
    print("DEMO 3: Template Preloading (Performance)")
    print("="*60)
    
    # Get available objects
    available = pipeline.template_manager.get_available_objects()
    test_objects = available[:5] if len(available) >= 5 else available
    
    print(f"\n1. Testing with {len(test_objects)} objects: {', '.join(test_objects)}")
    
    # Create query
    query_img = create_synthetic_query()
    K = get_default_camera_intrinsics()
    
    # First call (cold - templates not cached)
    print("\n2. First inference (cold - templates not cached)...")
    start_time = time.time()
    est1 = pipeline.estimate_pose(query_img, test_objects[0], K)
    cold_time = time.time() - start_time
    print(f"   ✓ Cold inference: {cold_time*1000:.1f}ms")
    
    # Second call (warm - template cached)
    print("\n3. Second inference (warm - template cached)...")
    start_time = time.time()
    est2 = pipeline.estimate_pose(query_img, test_objects[0], K)
    warm_time = time.time() - start_time
    print(f"   ✓ Warm inference: {warm_time*1000:.1f}ms")
    
    speedup = cold_time / warm_time if warm_time > 0 else 0
    print(f"   ✓ Speedup: {speedup:.1f}x")
    
    # Preload remaining objects
    print(f"\n4. Preloading remaining {len(test_objects)-1} objects...")
    start_time = time.time()
    pipeline.preload_objects(test_objects[1:])
    preload_time = time.time() - start_time
    print(f"   ✓ Preloaded in {preload_time:.2f}s")
    
    # Show cache stats
    stats = pipeline.get_stats()
    cache_info = stats['template_manager']
    print(f"\n5. Cache Statistics:")
    print(f"   Cached objects: {cache_info['cached_objects']}/{cache_info['cache_size']}")
    print(f"   Objects: {', '.join(cache_info['cache_order'])}")


def demo_subset_mode():
    """Demo fast inference with subset mode."""
    print("\n" + "="*60)
    print("DEMO 4: Subset Mode (Fast Inference)")
    print("="*60)
    
    templates_dir = project_root / "datasets" / "templates"
    
    # Create subset mode pipeline
    print("\n1. Creating pipeline in subset mode (4 positive + 2 negative templates)...")
    pipeline = InferencePipeline(
        templates_dir=templates_dir,
        dataset_name="gso",
        template_mode="subset",
        cache_size=5,
    )
    print("   ✓ Pipeline created")
    
    # Create query and pose hint
    query_img = create_synthetic_query()
    K = get_default_camera_intrinsics()
    
    # Create pose hint
    R = Rotation.from_euler('xyz', [30, 45, 15], degrees=True).as_matrix()
    t = np.array([0.0, 0.0, 1.0])
    pose_hint = np.eye(4)
    pose_hint[:3, :3] = R
    pose_hint[:3, 3] = t
    
    print("\n2. Estimating pose with query pose hint...")
    print(f"   Hint rotation: [30°, 45°, 15°]")
    print(f"   Hint translation: [0, 0, 1] m")
    
    start_time = time.time()
    estimate = pipeline.estimate_pose(
        query_img,
        "000733",
        K,
        query_pose_hint=pose_hint,
    )
    elapsed = time.time() - start_time
    
    print(f"   ✓ Completed in {elapsed*1000:.1f}ms")
    print(f"   Only 6 templates used (vs 162 in all mode)")
    print(f"   Score: {estimate.score:.3f}")


def demo_batch_processing(pipeline):
    """Demo batch processing multiple queries."""
    print("\n" + "="*60)
    print("DEMO 5: Batch Processing Multiple Queries")
    print("="*60)
    
    # Create multiple queries
    num_queries = 3
    object_ids = ["000733", "000001", "000003"]
    
    print(f"\n1. Processing {num_queries} queries...")
    print(f"   Objects: {', '.join(object_ids)}")
    
    K = get_default_camera_intrinsics()
    
    # Process each query
    results = []
    total_time = 0
    
    for i in range(num_queries):
        query_img = create_synthetic_query()
        
        start_time = time.time()
        result = pipeline.estimate_poses(query_img, object_ids, K)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        results.append(result)
        print(f"\n   Query {i+1}: {elapsed*1000:.1f}ms")
        best = result.best_estimate()
        print(f"     Best: {best.object_id} (score={best.score:.3f})")
    
    avg_time = total_time / num_queries
    print(f"\n2. Batch Statistics:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average per query: {avg_time*1000:.1f}ms")
    print(f"   Throughput: {num_queries/total_time:.2f} queries/sec")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="VPOG Inference Pipeline Demo")
    parser.add_argument("--templates", type=str, default="datasets/templates",
                       help="Path to templates directory")
    parser.add_argument("--dataset", type=str, default="gso",
                       help="Dataset name (gso, lmo, ycbv, etc.)")
    parser.add_argument("--object", type=str, default="000733",
                       help="Single object ID for demo")
    parser.add_argument("--objects", type=str, nargs="+",
                       default=["000733", "000001", "000003"],
                       help="Multiple object IDs for multi-object demo")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["all", "subset"],
                       help="Template mode: all (162) or subset (4+2)")
    parser.add_argument("--cache-size", type=int, default=10,
                       help="Template cache size")
    parser.add_argument("--demos", type=str, nargs="+",
                       default=["all"],
                       choices=["all", "single", "multi", "preload", "subset", "batch"],
                       help="Which demos to run")
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("VPOG INFERENCE PIPELINE DEMO")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Templates: {args.templates}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Mode: {args.mode}")
    print(f"  Cache size: {args.cache_size}")
    
    # Check templates directory
    templates_dir = Path(args.templates)
    if not templates_dir.exists():
        print(f"\n✗ ERROR: Templates directory not found: {templates_dir}")
        print(f"  Please ensure templates are available at the specified path.")
        return 1
    
    # Initialize pipeline
    print("\n" + "="*60)
    print("Initializing Pipeline")
    print("="*60)
    
    try:
        pipeline = create_inference_pipeline(
            templates_dir=str(templates_dir),
            dataset_name=args.dataset,
            template_mode=args.mode,
            cache_size=args.cache_size,
        )
        print("✓ Pipeline initialized successfully")
        
        stats = pipeline.get_stats()
        print(f"  Device: {stats['device']}")
        print(f"  Min inliers: {stats['min_inliers']}")
        
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Camera intrinsics
    K = get_default_camera_intrinsics()
    
    # Run demos
    demos_to_run = args.demos
    if "all" in demos_to_run:
        demos_to_run = ["single", "multi", "preload", "subset", "batch"]
    
    try:
        if "single" in demos_to_run:
            demo_single_object(pipeline, args.object, K)
        
        if "multi" in demos_to_run:
            demo_multi_object(pipeline, args.objects, K)
        
        if "preload" in demos_to_run:
            demo_preloading(pipeline)
        
        if "subset" in demos_to_run:
            demo_subset_mode()
        
        if "batch" in demos_to_run:
            demo_batch_processing(pipeline)
        
    except KeyboardInterrupt:
        print("\n\n✗ Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Final summary
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\n✓ All demos completed successfully!")
    print("\nNext steps:")
    print("  1. Integrate trained VPOG model for real correspondences")
    print("  2. Test on real query images")
    print("  3. Evaluate on BOP benchmark datasets")
    print("  4. Add visualization tools")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
