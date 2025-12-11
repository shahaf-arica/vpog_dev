"""
Debug script to check template selection for a specific object and seed
"""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from training.dataloader.vpog_dataset import VPOGTrainDataset
from src.utils.dataloader import NoneFilteringDataLoader
import matplotlib.pyplot as plt
from PIL import Image

SEED = 2  # Change this to test different seeds
TARGET_OBJECT = "733"  # The object to debug

project_root = Path.cwd()
template_config = {
    'dir': str(project_root / 'datasets' / 'templates'),
    'level_templates': 1,
    'pose_distribution': 'all',
    'scale_factor': 10.0,
    'num_templates': 162,
    'pose_name': 'object_poses/OBJECT_ID.npy',
}

print(f"="*80)
print(f"DEBUG: Template Selection for Object {TARGET_OBJECT} with SEED={SEED}")
print(f"="*80)

dataset = VPOGTrainDataset(
    root_dir=str(project_root / 'datasets'),
    dataset_name='gso',
    template_config=template_config,
    num_positive_templates=3,
    num_negative_templates=2,
    seed=SEED,
    batch_size=8,  # Larger batch to find our object
)

dataloader = NoneFilteringDataLoader(
    dataset.web_dataloader.datapipeline,
    batch_size=8,
    num_workers=0,
    collate_fn=dataset.collate_fn,
)

# Find batch containing target object
print(f"\nSearching for object {TARGET_OBJECT}...")
batch = None
sample_idx = None

for batch_data in dataloader:
    if batch_data is None:
        continue
    labels = batch_data.infos['label'].astype(str).values
    if TARGET_OBJECT in labels:
        batch = batch_data
        sample_idx = np.where(labels == TARGET_OBJECT)[0][0]
        print(f"✓ Found object {TARGET_OBJECT} at batch index {sample_idx}")
        break
    
if batch is None:
    print(f"✗ Object {TARGET_OBJECT} not found in first few batches")
    sys.exit(1)

# Get query pose
query_pose = batch.poses[sample_idx, 0].cpu().numpy()

# Load ALL template poses for this object
template_pose_path = project_root / 'datasets' / 'templates' / 'gso' / 'object_poses' / f'{int(TARGET_OBJECT):06d}.npy'
template_poses = np.load(template_pose_path)

print(f"\nComputing angles from query to ALL {len(template_poses)} templates...")

# Compute SO(3) distance to all templates
angles = []
for i, template_pose in enumerate(template_poses):
    R_query = query_pose[:3, :3]
    R_template = template_pose[:3, :3]
    R_rel = R_query.T @ R_template
    trace = np.trace(R_rel)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle_deg = np.rad2deg(np.arccos(cos_angle))
    angles.append(angle_deg)

angles = np.array(angles)

# Find top 20 nearest
nearest_20 = np.argsort(angles)[:20]
print(f"\nTop 20 nearest templates:")
for rank, idx in enumerate(nearest_20):
    print(f"  Rank {rank+1:2d}: idx={idx:3d} (view_id={idx:06d}.png), angle={angles[idx]:7.2f}°")

print(f"\nSelected templates (from batch):")
selected_indices = batch.template_indices[sample_idx].cpu().numpy()
selected_types = batch.template_types[sample_idx].cpu().numpy()

for i, (idx, typ) in enumerate(zip(selected_indices, selected_types)):
    type_str = "POS" if typ == 0 else "NEG"
    rank = np.where(np.argsort(angles) == idx)[0][0] + 1
    print(f"  T{i} ({type_str}): idx={idx:3d}, angle={angles[idx]:7.2f}°, rank={rank:3d}")

# Check specific template 70
if 70 < len(angles):
    rank_70 = np.where(np.argsort(angles) == 70)[0][0] + 1
    print(f"\nTemplate at index 70 (the one you mentioned):")
    print(f"  Angle={angles[70]:7.2f}°, rank={rank_70}")
    print(f"  File: datasets/templates/gso/{int(TARGET_OBJECT):06d}/{70:06d}.png")

# Visualize
print(f"\nCreating visualization...")
template_dir = project_root / 'datasets' / 'templates' / 'gso' / f'{int(TARGET_OBJECT):06d}'

fig, axes = plt.subplots(3, 7, figsize=(21, 10))

# Row 1: Query + top 6 nearest templates
def denorm_img(tensor):
    img = tensor.cpu().numpy()
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    img = img * std + mean
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0, 1)

query_img = denorm_img(batch.images[sample_idx, 0])
axes[0, 0].imshow(query_img)
axes[0, 0].set_title(f'QUERY\nObj {TARGET_OBJECT}', fontsize=10, fontweight='bold', color='blue')
axes[0, 0].axis('off')

for i in range(6):
    idx = nearest_20[i]
    template_img_path = template_dir / f'{idx:06d}.png'
    if template_img_path.exists():
        img = np.array(Image.open(template_img_path))
        axes[0, i+1].imshow(img)
        axes[0, i+1].set_title(f'Rank {i+1}\nidx={idx}, {angles[idx]:.1f}°', fontsize=9)
        axes[0, i+1].axis('off')

# Row 2: Selected positive templates
for i in range(3):
    idx = selected_indices[i]
    template_img = denorm_img(batch.images[sample_idx, i+1])
    axes[1, i].imshow(template_img)
    rank = np.where(np.argsort(angles) == idx)[0][0] + 1
    axes[1, i].set_title(f'SELECTED POS {i}\nidx={idx}, {angles[idx]:.1f}°\nRank={rank}', 
                          fontsize=9, color='green', fontweight='bold')
    axes[1, i].axis('off')

# Show template 70 if mentioned
if 70 < len(angles):
    template_img_path = template_dir / f'{70:06d}.png'
    if template_img_path.exists():
        img = np.array(Image.open(template_img_path))
        axes[1, 3].imshow(img)
        rank_70 = np.where(np.argsort(angles) == 70)[0][0] + 1
        axes[1, 3].set_title(f'Template 70\n(you mentioned)\n{angles[70]:.1f}°, Rank={rank_70}', 
                              fontsize=9, color='purple', fontweight='bold')
        axes[1, 3].axis('off')

# Row 3: Show rankings 7-10
for i in range(7):
    if i + 6 < len(nearest_20):
        idx = nearest_20[i + 6]
        template_img_path = template_dir / f'{idx:06d}.png'
        if template_img_path.exists():
            img = np.array(Image.open(template_img_path))
            axes[2, i].imshow(img)
            axes[2, i].set_title(f'Rank {i+7}\nidx={idx}, {angles[idx]:.1f}°', fontsize=9)
            axes[2, i].axis('off')

plt.suptitle(f'Object {TARGET_OBJECT} Template Selection Debug (SEED={SEED})\n' +
             'Row 1: Query + Top 6 nearest | Row 2: Selected POS + Template 70 | Row 3: Ranks 7-13',
             fontsize=12, fontweight='bold')
plt.tight_layout()

save_path = project_root / 'tmp' / 'vpog_dataset_test' / f'debug_obj{TARGET_OBJECT}_seed{SEED}.png'
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization to {save_path}")
plt.close()

print(f"\n" + "="*80)
print(f"ANALYSIS:")
print(f"  - If selected POS templates are NOT in top 3, there's a bug")
print(f"  - Check the visualization to see if selection makes sense")
print(f"="*80)
