"""
Template Selector for VPOG
Handles selection of S_p nearest out-of-plane templates and S_n random negative templates
Also extracts d_ref (reference direction) from template poses
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist

# Import from GigaPose
from src.lib3d.numpy import opencv2opengl
from src.lib3d.template_transform import get_obj_poses_from_template_level


def extract_d_ref_from_pose(pose: np.ndarray) -> np.ndarray:
    """
    Extract reference direction (d_ref) from a template pose.
    
    d_ref is the unit vector pointing from the template origin to the camera.
    In the template coordinate system (OpenGL), this is the camera's viewing direction.
    
    Args:
        pose: 4x4 transformation matrix (camera to object in OpenCV convention)
    
    Returns:
        d_ref: 3D unit vector representing camera direction
    """
    # Convert to OpenGL coordinate system (GigaPose uses this for template matching)
    pose_opengl = opencv2opengl(pose)
    
    # In OpenGL, the camera looks along -Z axis
    # The third column of rotation matrix gives the Z-axis direction in object frame
    # This represents the direction from object to camera
    d_ref = pose_opengl[:3, 2]
    
    # Normalize to unit vector
    d_ref = d_ref / np.linalg.norm(d_ref)
    
    return d_ref


class TemplateSelector:
    """
    Selects S = S_p + S_n templates for each query:
    - S_p: nearest templates by out-of-plane rotation
    - S_n: random negative templates with >= min_angle difference
    """
    
    def __init__(
        self,
        level_templates: int = 1,
        pose_distribution: str = "all",
        num_positive: int = 4,
        num_negative: int = 2,
        min_negative_angle_deg: float = 90.0,
        d_ref_random_ratio: float = 0.0,
        positive_sampling_mode: str = "nearest",
        positive_sampling_deg_range: float = 30.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            level_templates: Icosphere level for templates (1=162 views, 2=642 views)
            pose_distribution: 'all' or 'upper' hemisphere
            num_positive: S_p - number of nearest templates to select
            num_negative: S_n - number of random negative templates
            min_negative_angle_deg: Minimum angular difference for negative templates
            d_ref_random_ratio: Probability of using random template as d_ref instead of nearest
            positive_sampling_mode: 'nearest' or 'random_within_range'
                - 'nearest': Take S_p-1 strictly nearest templates to the closest one
                - 'random_within_range': Randomly sample S_p-1 from templates within deg_range
            positive_sampling_deg_range: Angular range (degrees) for random sampling mode
            seed: Random seed for reproducibility (None = no seed)
        """
        self.level_templates = level_templates
        self.pose_distribution = pose_distribution
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.min_negative_angle_deg = min_negative_angle_deg
        self.d_ref_random_ratio = d_ref_random_ratio
        self.positive_sampling_mode = positive_sampling_mode
        self.positive_sampling_deg_range = positive_sampling_deg_range
        
        # Set random seed if provided
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        
        # Load all available template poses
        self.avail_index, self.template_poses = get_obj_poses_from_template_level(
            level_templates,
            pose_distribution,
            return_cam=False,
            return_index=True,
        )
        
        # Convert to OpenGL for direction computation
        self.template_poses_opengl = opencv2opengl(self.template_poses)
        
        # Extract directions (camera to object direction, which is -Z axis in camera frame)
        # For out-of-plane rotation, we care about viewing direction
        self.template_directions = self.template_poses_opengl[:, :3, 2]  # Nx3
        
        # Normalize directions
        self.template_directions = self.template_directions / np.linalg.norm(
            self.template_directions, axis=1, keepdims=True
        )
    
    def select_templates(
        self,
        query_pose: np.ndarray,
        return_d_ref: bool = True,
    ) -> Dict:
        """
        Select S_p + S_n templates for a given query pose.
        
        CORRECTED LOGIC:
        1. Find the single nearest out-of-plane template (like GigaPose)
        2. Find S_p-1 additional templates that are close to this nearest template on S²
        3. Select S_n random negative templates with ≥ min_angle difference
        
        Args:
            query_pose: 4x4 transformation matrix of query object
            return_d_ref: Whether to return d_ref
        
        Returns:
            Dictionary containing:
                - positive_indices: List of S_p template indices (nearest)
                - negative_indices: List of S_n template indices (random with constraint)
                - d_ref: Reference direction (3D unit vector) if return_d_ref=True
                - d_ref_source: 'nearest' or 'random' indicating how d_ref was chosen
        """
        # Convert query pose to OpenGL
        query_pose_opengl = opencv2opengl(query_pose)
        query_direction = query_pose_opengl[:3, 2]
        query_direction = query_direction / np.linalg.norm(query_direction)
        
        # STEP 1: Find the single nearest template using full SO(3) distance
        # Compute SO(3) distances from query to ALL templates
        angles_to_query = np.zeros(len(self.template_poses))
        R_query = query_pose[:3, :3]
        
        for i, template_pose in enumerate(self.template_poses):
            R_template = template_pose[:3, :3]
            R_rel = R_query.T @ R_template
            trace = np.trace(R_rel)
            cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
            angles_to_query[i] = np.rad2deg(np.arccos(cos_angle))
        
        # Get the nearest template index (minimum angle)
        nearest_template_idx = int(np.argmin(angles_to_query))
        nearest_template_pose = self.template_poses[nearest_template_idx]
        
        # STEP 2: Find S_p-1 templates close to the nearest template in SO(3)
        # Compute proper SO(3) angular distances from ALL templates to the nearest template
        angles_to_nearest = np.zeros(len(self.template_poses))
        R_nearest = nearest_template_pose[:3, :3]
        
        for i, template_pose in enumerate(self.template_poses):
            R_template = template_pose[:3, :3]
            R_rel = R_nearest.T @ R_template
            trace = np.trace(R_rel)
            cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
            angles_to_nearest[i] = np.rad2deg(np.arccos(cos_angle))
        
        # Sort templates by their angular distance to the nearest template
        sorted_by_distance_to_nearest = np.argsort(angles_to_nearest)
        
        if self.positive_sampling_mode == "nearest":
            # Take the S_p-1 strictly nearest templates (excluding the nearest itself at index 0)
            positive_indices = [nearest_template_idx]
            for idx in sorted_by_distance_to_nearest[1:]:  # Skip index 0 which is nearest_template_idx
                if len(positive_indices) >= self.num_positive:
                    break
                positive_indices.append(int(idx))
        
        elif self.positive_sampling_mode == "random_within_range":
            # Find all templates within deg_range of the nearest template
            max_angle = self.positive_sampling_deg_range
            candidates_mask = (angles_to_nearest <= max_angle) & (np.arange(len(angles_to_nearest)) != nearest_template_idx)
            candidates = np.where(candidates_mask)[0]
            
            if len(candidates) < self.num_positive - 1:
                # Not enough candidates, fall back to nearest
                print(f"Warning: Only {len(candidates)} templates within {max_angle}° of nearest, need {self.num_positive-1}. Using nearest.")
                positive_indices = sorted_by_distance_to_nearest[:self.num_positive].tolist()
            else:
                # Randomly sample S_p-1 from candidates
                sampled = self.rng.choice(candidates, size=self.num_positive - 1, replace=False)
                positive_indices = [nearest_template_idx] + sampled.tolist()
        else:
            raise ValueError(f"Unknown positive_sampling_mode: {self.positive_sampling_mode}")
        
        # STEP 3: Select negative templates (large angular difference from query)
        # Find all templates with angle >= min_negative_angle_deg
        negative_candidates = np.where(angles_to_query >= self.min_negative_angle_deg)[0]
        
        if len(negative_candidates) < self.num_negative:
            # If not enough candidates, take the farthest available ones
            sorted_by_distance_to_query = np.argsort(-angles_to_query)  # Descending (farthest first)
            negative_indices = sorted_by_distance_to_query[:self.num_negative].tolist()
        else:
            # Randomly sample from candidates
            sampled = self.rng.choice(negative_candidates, size=self.num_negative, replace=False)
            negative_indices = sampled.tolist()
        
        result = {
            'positive_indices': positive_indices,
            'negative_indices': negative_indices,
            'all_indices': positive_indices + negative_indices,
            'nearest_template_idx': nearest_template_idx,
        }
        
        if return_d_ref:
            # Determine d_ref: use nearest template or random based on ratio
            if self.rng.random() < self.d_ref_random_ratio:
                # Use random template from positives as reference
                d_ref_idx = self.rng.choice(positive_indices)
                d_ref_source = 'random'
            else:
                # Use the nearest template as reference
                d_ref_idx = nearest_template_idx
                d_ref_source = 'nearest'
            
            # Extract d_ref from the selected template
            d_ref_pose = self.template_poses[d_ref_idx]
            d_ref = extract_d_ref_from_pose(d_ref_pose)
            
            result['d_ref'] = d_ref
            result['d_ref_source'] = d_ref_source
            result['d_ref_template_idx'] = d_ref_idx
        
        return result
    
    def get_template_view_id(self, template_idx: int) -> int:
        """
        Get the actual view ID for rendering from template index.
        
        Args:
            template_idx: Index in the template list
        
        Returns:
            view_id: Actual view ID for loading template image
        """
        return int(self.avail_index[template_idx])
    
    def compute_angular_distance(
        self,
        pose1: np.ndarray,
        pose2: np.ndarray,
    ) -> float:
        """
        Compute angular distance between two poses in SO(3) (in degrees).
        
        Uses the geodesic distance on SO(3): angle = arccos((trace(R1^T @ R2) - 1) / 2)
        This properly measures the 3D rotation difference, not just viewing direction.
        
        Args:
            pose1, pose2: 4x4 transformation matrices
        
        Returns:
            Angular distance in degrees
        """
        # Extract rotation matrices
        R1 = pose1[:3, :3]
        R2 = pose2[:3, :3]
        
        # Compute relative rotation: R1^T @ R2
        R_rel = R1.T @ R2
        
        # Geodesic distance on SO(3): angle = arccos((trace(R) - 1) / 2)
        trace = np.trace(R_rel)
        # Clamp to valid range for numerical stability
        cos_angle = (trace - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.rad2deg(angle_rad)
        
        return angle_deg


if __name__ == "__main__":
    """
    REAL TEST with GSO data and visualizations
    Tests template selection with actual query poses and saves visualizations
    """
    import os
    import sys
    from pathlib import Path
    import cv2
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from bop_toolkit_lib import inout
    from src.custom_megapose.template_dataset import TemplateDataset
    from PIL import Image
    
    print("=" * 80)
    print("REAL TEST: TemplateSelector with GSO Data")
    print("=" * 80)
    
    # Setup paths
    dataset_dir = project_root / "datasets" / "gso"
    templates_dir = project_root / "datasets" / "templates" / "gso"
    save_dir = project_root / "tmp" / "template_selector_test"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if not dataset_dir.exists():
        print(f"\n✗ GSO dataset not found at {dataset_dir}")
        print("Please ensure GSO data is available")
        sys.exit(1)
    
    if not templates_dir.exists():
        print(f"\n✗ Templates not found at {templates_dir}")
        print("Please run: python -m src.scripts.render_gso_templates")
        sys.exit(1)
    
    print(f"✓ Found GSO dataset at {dataset_dir}")
    print(f"✓ Found templates at {templates_dir}")
    
    # Load model info and template dataset
    model_infos = inout.load_json(dataset_dir / "models_info.json")
    print(f"✓ Loaded {len(model_infos)} objects")
    
    # Initialize template dataset
    template_config = type('Config', (), {
        'dir': str(templates_dir),
        'level_templates': 1,
        'pose_distribution': 'all',
        'scale_factor': 10.0,
        'num_templates': 162,
        'pose_name': 'object_poses/OBJECT_ID.npy',
    })()
    
    template_dataset = TemplateDataset.from_config(
        model_infos[:5],  # Test first 5 objects
        template_config
    )
    print(f"✓ Initialized TemplateDataset with {len(template_dataset)} objects")
    
    # Initialize selector with seed for reproducibility
    selector = TemplateSelector(
        level_templates=1,
        pose_distribution="all",
        num_positive=4,
        num_negative=2,
        min_negative_angle_deg=90.0,
        d_ref_random_ratio=0.0,
        positive_sampling_mode="nearest",  # or "random_within_range"
        positive_sampling_deg_range=30.0,
        seed=42,  # Fixed seed for reproducibility
    )
    print(f"✓ Initialized TemplateSelector (S_p={selector.num_positive}, S_n={selector.num_negative})")
    print(f"  Sampling mode: {selector.positive_sampling_mode}")
    print(f"  Random seed: {selector.seed}")
    print(f"  Available templates: {len(selector.template_poses)}")
    
    # Test with real template poses
    num_test_samples = 3
    print(f"\n" + "=" * 80)
    print(f"Testing with {num_test_samples} real objects from GSO")
    print("=" * 80)
    rng = np.random.default_rng(42)
    for sample_idx in range(num_test_samples):
        template_obj = template_dataset[sample_idx]
        obj_label = template_obj.label
        
        print(f"\n{'='*80}")
        print(f"Sample {sample_idx + 1}: Object {obj_label}")
        print(f"{'='*80}")
        
        # Load a random query pose from this object's templates
        poses_file = templates_dir / f"object_poses/{int(obj_label):06d}.npy"
        if not poses_file.exists():
            print(f"✗ Poses file not found: {poses_file}")
            continue
        
        all_poses = np.load(poses_file)
        
        # Pick a random template index and use its EXACT pose from selector
        query_template_idx = int(rng.integers(0, len(selector.template_poses)))
        query_view_id = int(selector.avail_index[query_template_idx])
        
        # CRITICAL: Use selector's pose directly to guarantee exact match
        query_pose = selector.template_poses[query_template_idx].copy()
        
        print(f"Query: selector_idx={query_template_idx}, view_id={query_view_id}")
        
        # Verify: Distance from query to its own template should be 0
        dist_to_self = selector.compute_angular_distance(query_pose, selector.template_poses[query_template_idx])
        print(f"  Self-distance check: {dist_to_self:.10f}° (should be ~0)")
        
        # Select templates
        result = selector.select_templates(query_pose, return_d_ref=True)
        
        nearest_idx = result['nearest_template_idx']
        nearest_view_id = selector.get_template_view_id(nearest_idx)
        
        print(f"\nSelected Templates:")
        print(f"  Nearest: selector_idx={nearest_idx}, view_id={nearest_view_id}")
        
        # CRITICAL CHECK: Nearest should be query_template_idx!
        if nearest_idx != query_template_idx:
            print(f"  ✗✗✗ BUG: Nearest ({nearest_idx}) != Query ({query_template_idx})!")
            dist = selector.compute_angular_distance(query_pose, selector.template_poses[nearest_idx])
            print(f"       Distance to nearest: {dist:.6f}°")
        else:
            print(f"  ✓ Nearest matches query!")
        
        print(f"  Positive indices: {result['positive_indices']}")
        print(f"  Positive view_ids: {[selector.get_template_view_id(i) for i in result['positive_indices']]}")
        print(f"  Negative indices: {result['negative_indices']}")
        print(f"  Negative view_ids: {[selector.get_template_view_id(i) for i in result['negative_indices']]}")
        
        # Verify angular distances
        nearest_pose = selector.template_poses[nearest_idx]
        
        print(f"\nAngular Distances from Query and Nearest:")
        for i, idx in enumerate(result['positive_indices']):
            view_id = selector.get_template_view_id(idx)
            angle_to_query = selector.compute_angular_distance(query_pose, selector.template_poses[idx])
            angle_to_nearest = selector.compute_angular_distance(nearest_pose, selector.template_poses[idx])
            marker = "*" if idx == nearest_idx else " "
            marker += "Q" if idx == query_template_idx else " "
            print(f"  {marker}Positive {i+1} (template {view_id:3d}): {angle_to_query:6.2f}° from query, {angle_to_nearest:6.2f}° from nearest")
        
        for i, idx in enumerate(result['negative_indices']):
            view_id = selector.get_template_view_id(idx)
            angle_to_query = selector.compute_angular_distance(query_pose, selector.template_poses[idx])
            print(f"   Negative {i+1} (template {view_id:3d}): {angle_to_query:6.2f}° from query")
        
        # VISUALIZATION: Load and display selected templates
        print(f"\nGenerating visualization...")
        
        # Layout: 2 rows x 4 columns
        # Row 1: Query + 3 positives
        # Row 2: 1 positive + 2 negatives + empty
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Load query template image
        query_img_path = templates_dir / f"{int(obj_label):06d}/{query_view_id:06d}.png"
        if query_img_path.exists():
            query_img = np.array(Image.open(query_img_path))
            axes[0].imshow(query_img)
            nearest_view_id = selector.get_template_view_id(result['nearest_template_idx'])
            axes[0].set_title(f'Query (view {query_view_id})\nNearest: view {nearest_view_id}', 
                            fontsize=10, fontweight='bold', color='blue')
            axes[0].axis('off')
        
        # Load all 4 positive templates
        for i, idx in enumerate(result['positive_indices']):
            if i < 3:
                ax_idx = i + 1  # First row: positions 1, 2, 3
            else:
                ax_idx = 4  # Second row: position 0
            
            view_id = selector.get_template_view_id(idx)
            img_path = templates_dir / f"{int(obj_label):06d}/{view_id:06d}.png"
            if img_path.exists():
                img = np.array(Image.open(img_path))
                axes[ax_idx].imshow(img)
                angle_to_query = selector.compute_angular_distance(query_pose, selector.template_poses[idx])
                angle_to_nearest = selector.compute_angular_distance(nearest_pose, selector.template_poses[idx])
                marker = "★" if idx == nearest_idx else ""
                axes[ax_idx].set_title(f'{marker}Positive {i+1} (t={view_id})\nQuery: {angle_to_query:.1f}° | Nearest: {angle_to_nearest:.1f}°', 
                                   fontsize=9, color='green', fontweight='bold')
                axes[ax_idx].axis('off')
        
        # Load negative templates
        for i, idx in enumerate(result['negative_indices']):
            ax_idx = 5 + i  # Second row: positions 1, 2
            view_id = selector.get_template_view_id(idx)
            img_path = templates_dir / f"{int(obj_label):06d}/{view_id:06d}.png"
            if img_path.exists():
                img = np.array(Image.open(img_path))
                axes[ax_idx].imshow(img)
                angle = selector.compute_angular_distance(query_pose, selector.template_poses[idx])
                axes[ax_idx].set_title(f'Negative {i+1} (t={view_id})\nQuery: {angle:.1f}°', 
                                   fontsize=9, color='red', fontweight='bold')
                axes[ax_idx].axis('off')
        
        # Hide unused axis
        axes[7].axis('off')
        
        # Add d_ref visualization as text
        fig.text(0.5, 0.02, 
                f'Object {obj_label} | d_ref = [{result["d_ref"][0]:.3f}, {result["d_ref"][1]:.3f}, {result["d_ref"][2]:.3f}]',
                ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        save_path = save_dir / f"sample_{sample_idx:02d}_obj{obj_label}_selection.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved visualization to {save_path}")
    
    # TEST RANDOM SAMPLING MODE
    print(f"\n" + "=" * 80)
    print(f"Testing RANDOM_WITHIN_RANGE mode (30° neighborhood)")
    print("=" * 80)
    
    # Create selector with random sampling
    selector_random = TemplateSelector(
        level_templates=1,
        pose_distribution="all",
        num_positive=4,
        num_negative=2,
        min_negative_angle_deg=90.0,
        d_ref_random_ratio=0.0,
        positive_sampling_mode="random_within_range",
        positive_sampling_deg_range=30.0,
        seed=42,
    )
    print(f"✓ Initialized selector with random_within_range mode")
    
    # Test with first object
    template_obj = template_dataset[0]
    obj_label = template_obj.label
    poses_file = templates_dir / f"object_poses/{int(obj_label):06d}.npy"
    all_poses = np.load(poses_file)
    
    # Use a consistent query view_id that exists in selector
    query_view_id = int(selector_random.avail_index[14])  # Use index 14 in the selector's template list
    query_pose = all_poses[query_view_id].copy()
    query_template_idx = 14
    
    result_random = selector_random.select_templates(query_pose, return_d_ref=True)
    nearest_idx_random = result_random['nearest_template_idx']
    nearest_pose_random = selector_random.template_poses[nearest_idx_random]
    
    print(f"\nObject {obj_label}, Query view_id {query_view_id} (selector index {query_template_idx})")
    print(f"  Nearest template: {nearest_idx_random}")
    print(f"  Positive indices: {result_random['positive_indices']}")
    
    print(f"\nAngular Distances (Random Sampling):")
    for i, idx in enumerate(result_random['positive_indices']):
        view_id = selector_random.get_template_view_id(idx)
        angle_to_query = selector_random.compute_angular_distance(query_pose, selector_random.template_poses[idx])
        angle_to_nearest = selector_random.compute_angular_distance(nearest_pose_random, selector_random.template_poses[idx])
        marker = "*" if idx == nearest_idx_random else " "
        print(f"  {marker}Positive {i+1} (template {view_id:3d}): {angle_to_query:6.2f}° from query, {angle_to_nearest:6.2f}° from nearest")
    
    # Visualization for random mode
    print(f"\nGenerating random mode visualization...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Query
    query_img_path = templates_dir / f"{int(obj_label):06d}/{query_view_id:06d}.png"
    if query_img_path.exists():
        query_img = np.array(Image.open(query_img_path))
        axes[0].imshow(query_img)
        axes[0].set_title(f'Query (view {query_view_id})', 
                        fontsize=10, fontweight='bold', color='blue')
        axes[0].axis('off')
    
    # All 4 positives
    for i, idx in enumerate(result_random['positive_indices']):
        if i < 3:
            ax_idx = i + 1
        else:
            ax_idx = 4
        
        view_id = selector_random.get_template_view_id(idx)
        img_path = templates_dir / f"{int(obj_label):06d}/{view_id:06d}.png"
        if img_path.exists():
            img = np.array(Image.open(img_path))
            axes[ax_idx].imshow(img)
            angle_to_query = selector_random.compute_angular_distance(query_pose, selector_random.template_poses[idx])
            angle_to_nearest = selector_random.compute_angular_distance(nearest_pose_random, selector_random.template_poses[idx])
            marker = "★" if idx == nearest_idx_random else "~"
            axes[ax_idx].set_title(f'{marker}Positive {i+1} (t={view_id})\nQuery: {angle_to_query:.1f}° | Nearest: {angle_to_nearest:.1f}°', 
                               fontsize=9, color='darkgreen', fontweight='bold')
            axes[ax_idx].axis('off')
    
    # Negatives
    for i, idx in enumerate(result_random['negative_indices']):
        ax_idx = 5 + i
        view_id = selector_random.get_template_view_id(idx)
        img_path = templates_dir / f"{int(obj_label):06d}/{view_id:06d}.png"
        if img_path.exists():
            img = np.array(Image.open(img_path))
            axes[ax_idx].imshow(img)
            angle = selector_random.compute_angular_distance(query_pose, selector_random.template_poses[idx])
            axes[ax_idx].set_title(f'Negative {i+1} (t={view_id})\nQuery: {angle:.1f}°', 
                               fontsize=9, color='red', fontweight='bold')
            axes[ax_idx].axis('off')
    
    axes[7].axis('off')
    
    fig.text(0.5, 0.02, 
            f'RANDOM SAMPLING MODE | Object {obj_label} | Sampled from 30° neighborhood',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    save_path_random = save_dir / f"sample_random_mode_obj{obj_label}_selection.png"
    plt.savefig(save_path_random, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved random mode visualization to {save_path_random}")
    
    # Summary statistics across all templates
    print(f"\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    # Test angle distribution
    angles_positive = []
    angles_negative = []
    
    for _ in range(100):
        query_pose = np.eye(4)
        query_pose[:3, :3] = Rotation.random().as_matrix()
        result = selector.select_templates(query_pose, return_d_ref=True)
        
        for idx in result['positive_indices']:
            angle = selector.compute_angular_distance(query_pose, selector.template_poses[idx])
            angles_positive.append(angle)
        
        for idx in result['negative_indices']:
            angle = selector.compute_angular_distance(query_pose, selector.template_poses[idx])
            angles_negative.append(angle)
    
    print(f"\nPositive Templates (100 samples):")
    print(f"  Mean angle: {np.mean(angles_positive):.2f}°")
    print(f"  Min angle:  {np.min(angles_positive):.2f}°")
    print(f"  Max angle:  {np.max(angles_positive):.2f}°")
    
    print(f"\nNegative Templates (100 samples):")
    print(f"  Mean angle: {np.mean(angles_negative):.2f}°")
    print(f"  Min angle:  {np.min(angles_negative):.2f}°")
    print(f"  Max angle:  {np.max(angles_negative):.2f}°")
    print(f"  % > 90°:    {(np.array(angles_negative) >= 90).mean() * 100:.1f}%")
    
    print("\n" + "=" * 80)
    print(f"✓ All tests passed!")
    print(f"✓ Visualizations saved to {save_dir}")
    print("=" * 80)
