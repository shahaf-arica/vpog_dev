"""
Quick test to visualize the 3 ORIGINAL depth maps
"""
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.dataloader.vpog_dataset import VPOGTrainDataset
from src.utils.dataloader import NoneFilteringDataLoader

print("Initializing dataset...")
template_config = {
    'dir': str(project_root / "datasets" / "templates"),
    'level_templates': 1,
    'pose_distribution': 'all',
    'scale_factor': 10.0,
    'num_templates': 162,
    'pose_name': 'object_poses/OBJECT_ID.npy',
}

dataset = VPOGTrainDataset(
    root_dir=str(project_root / "datasets"),
    dataset_name='gso',
    template_config=template_config,
    num_positive_templates=1,  # Just 1 for simplicity
    num_negative_templates=0,
    patch_size=16,
    image_size=224,
    batch_size=1,
    depth_scale=10.0,
    seed=2,
)

dataloader = NoneFilteringDataLoader(
    dataset.web_dataloader.datapipeline,
    batch_size=1,
    num_workers=0,
    collate_fn=dataset.collate_fn,
)

print("Loading batch and computing depth maps...")
batch = next(iter(dataloader))

# Extract data for first sample, first template
query_data = dataset.process_query(batch)
template_data, _ = dataset.process_templates(query_data)

b = 0
s = 0

device = query_data.K.device

# ORIGINAL query data
q_K_orig = query_data.K_original[b]
q_depth_orig = query_data.full_depth[b]
q_mask_orig = query_data.mask_original[b]
q_pose = query_data.pose[b]

# ORIGINAL template data
t_K_orig = template_data.K_original
t_pose = template_data.pose[b, s]
t_depth_orig = template_data.depth_original[b, s]

H_orig, W_orig = q_depth_orig.shape
H_t_orig, W_t_orig = t_depth_orig.shape

print(f"\nQuery depth: {H_orig}x{W_orig}, range [{q_depth_orig.min():.0f}, {q_depth_orig.max():.0f}] mm")
print(f"Template depth: {H_t_orig}x{W_t_orig}, range [{t_depth_orig.min():.0f}, {t_depth_orig.max():.0f}] mm")

# Create pixel grid for query
y_grid, x_grid = torch.meshgrid(
    torch.arange(H_orig, device=device, dtype=torch.float32),
    torch.arange(W_orig, device=device, dtype=torch.float32),
    indexing='ij'
)

# Unproject query
fx_q, fy_q, cx_q, cy_q = q_K_orig[0,0], q_K_orig[1,1], q_K_orig[0,2], q_K_orig[1,2]
z_q = q_depth_orig
x_q_3d = (x_grid - cx_q) * z_q / fx_q
y_q_3d = (y_grid - cy_q) * z_q / fy_q
points_q = torch.stack([x_q_3d, y_q_3d, z_q], dim=-1)

# Transform to template frame
T_q2w = torch.inverse(q_pose)
T_w2t = t_pose
T_q2t = T_w2t @ T_q2w
R = T_q2t[:3, :3]
t_vec = T_q2t[:3, 3]

points_t = torch.matmul(points_q, R.T) + t_vec
x_t_3d, y_t_3d, z_t = points_t[..., 0], points_t[..., 1], points_t[..., 2]

# Project to template
fx_t, fy_t, cx_t, cy_t = t_K_orig[0,0], t_K_orig[1,1], t_K_orig[0,2], t_K_orig[1,2]
u_t = (x_t_3d / z_t) * fx_t + cx_t
v_t = (y_t_3d / z_t) * fy_t + cy_t

# Valid projection
valid_q = (q_depth_orig > 0) & (q_mask_orig > 0.5)
valid_proj = (
    (z_t > 0) & 
    (u_t >= 0) & (u_t < W_t_orig) &
    (v_t >= 0) & (v_t < H_t_orig) &
    valid_q
)

print(f"\nValid query pixels: {valid_q.sum()} / {H_orig * W_orig}")
print(f"Valid projections: {valid_proj.sum()} / {valid_q.sum()}")

# Create projected depth map
projected_depth = torch.zeros(H_t_orig, W_t_orig, device=device)
valid_mask = valid_proj.flatten()
if valid_mask.any():
    u_int = u_t.long().clamp(0, W_t_orig-1)
    v_int = v_t.long().clamp(0, H_t_orig-1)
    u_valid = u_int.flatten()[valid_mask]
    v_valid = v_int.flatten()[valid_mask]
    z_valid = z_t.flatten()[valid_mask]
    indices = v_valid * W_t_orig + u_valid
    projected_depth.view(-1).scatter_(0, indices, z_valid)

print(f"Projected depth: non-zero pixels = {(projected_depth > 0).sum()}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Query depth (masked)
q_depth_viz = (q_depth_orig * q_mask_orig).cpu().numpy()
im1 = axes[0].imshow(q_depth_viz, cmap='jet', vmin=0, vmax=q_depth_viz[q_depth_viz > 0].max() if (q_depth_viz > 0).any() else 1)
axes[0].set_title(f'Query Depth (ORIGINAL {H_orig}x{W_orig})\nMasked, in mm')
axes[0].axis('off')
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# 2. Template depth
t_depth_viz = t_depth_orig.cpu().numpy()
im2 = axes[1].imshow(t_depth_viz, cmap='jet', vmin=0, vmax=t_depth_viz.max())
axes[1].set_title(f'Template Depth (ORIGINAL {H_t_orig}x{W_t_orig})\nin mm')
axes[1].axis('off')
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

# 3. Projected depth
proj_viz = projected_depth.cpu().numpy()
im3 = axes[2].imshow(proj_viz, cmap='jet', vmin=0, vmax=proj_viz.max() if proj_viz.max() > 0 else 1)
axes[2].set_title(f'Query→Template Projected\n{(proj_viz > 0).sum()} non-zero pixels')
axes[2].axis('off')
plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle('3 ORIGINAL Depth Maps (NO CROPPING)', fontsize=14, fontweight='bold')
plt.tight_layout()

save_path = project_root / "tmp" / "vpog_dataset_test" / "depth_maps_original.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved to {save_path}")
plt.close()

print("\nDone!")
