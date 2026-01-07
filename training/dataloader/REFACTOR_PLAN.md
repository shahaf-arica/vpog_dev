# VPOG Dataset Visualization Refactoring Plan

## Summary
Separate visualization logic from data loading logic by moving all visualization methods from `VPOGDataset` class to the new `vpog_visualizations.py` module.

## Current State
- ✅ Created `/training/dataloader/vpog_visualizations.py` with standalone functions
- ✅ Added import: `from training.dataloader import vpog_visualizations as viz`
- ⚠️ Need to remove 4 visualization methods from VPOGDataset class (~800 lines)
- ⚠️ Need to update 3 call sites in compute_flow_labels_for_train()
- ⚠️ Need to update test visualization in `__main__` block

## Methods to Remove from VPOGDataset (lines 845-1703)

### 1. `_visualize_correspondences` (lines 846-985)
   - Not currently used in the code
   - Can be removed entirely

### 2. `_visualize_dense_patch_flow` (lines 988-1233)
   - Called at line ~2123
   - Replace with: `viz.visualize_dense_patch_flow(...)`

### 3. `_visualize_train_correspondences` (lines 1235-1447)
   - Called at line ~2094
   - Replace with: `viz.visualize_train_correspondences(...)`

### 4. `_visualize_patch_correspondences` (lines 1465-1703)
   - Called at line ~1447 (from within _visualize_train_correspondences)
   - Replace with: `viz.visualize_patch_correspondences(...)`

### 5. `_project_pcd_helper` (lines 819-844)
   - Only used by _visualize_correspondences
   - Can be removed entirely

## Call Site Updates

### Site 1: Line ~1447 (within _visualize_train_correspondences)
```python
# OLD:
self._visualize_patch_correspondences(
    query_data=query_data,
    template_data=template_data,
    ...
)

# NEW:
viz.visualize_patch_correspondences(
    query_rgb=query_data.centered_rgb[batch_idx],
    template_rgb=template_data.rgb[batch_idx, template_idx],
    patch_has_object=patch_has_object,
    patch_is_visible=patch_is_visible,
    patch_flow_x=patch_flow_x,
    patch_flow_y=patch_flow_y,
    patch_buddy_i=patch_buddy_i,
    patch_buddy_j=patch_buddy_j,
    batch_idx=batch_idx,
    template_idx=template_idx,
    obj_label=obj_label,
    patch_size=self.patch_size,
    num_patches_per_side=self.num_patches_per_side,
    vis_dir=Path(self.vis_dir),
)
```

### Site 2: Line ~2094 (in compute_flow_labels_for_train, debug mode)
```python
# OLD:
self._visualize_train_correspondences(
    query_data=query_data,
    template_data=template_data,
    ...
)

# NEW:
viz.visualize_train_correspondences(
    query_rgb=query_data.centered_rgb[b],
    template_rgb=template_data.rgb[b, s],
    q_depth=q_depth,
    t_depth=t_depth,
    q_mask=q_mask,
    query_seen_map=query_seen_map,
    template_seen_map=template_seen_map,
    template_not_seen_map=template_not_seen_map,
    q_pose=q_pose,
    t_pose=t_pose,
    batch_idx=b,
    template_idx=s,
    obj_label=query_data.infos.label[b],
    vis_dir=Path(self.vis_dir),
)

# Also call patch correspondences separately now
viz.visualize_patch_correspondences(
    query_rgb=query_data.centered_rgb[b],
    template_rgb=template_data.rgb[b, s],
    patch_has_object=patch_has_object,
    patch_is_visible=patch_is_visible,
    patch_flow_x=flow_x_norm,
    patch_flow_y=flow_y_norm,
    patch_buddy_i=buddy_i,
    patch_buddy_j=buddy_j,
    batch_idx=b,
    template_idx=s,
    obj_label=query_data.infos.label[b],
    patch_size=self.patch_size,
    num_patches_per_side=self.num_patches_per_side,
    vis_dir=Path(self.vis_dir),
)
```

### Site 3: Line ~2123 (in compute_flow_labels_for_train, debug mode)
```python
# OLD:
self._visualize_dense_patch_flow(
    query_rgb=query_data.centered_rgb[b],
    template_rgb=template_data.rgb[b, s],
    ...
)

# NEW:
viz.visualize_dense_patch_flow(
    query_rgb=query_data.centered_rgb[b],
    template_rgb=template_data.rgb[b, s],
    q_mask=q_mask,
    t_depth=t_depth,
    flow_grid=dense_flow[b, s],
    weight_grid=dense_visibility[b, s],
    patch_has_object=patch_has_object,
    patch_is_visible=patch_is_visible,
    patch_buddy_i=buddy_i,
    patch_buddy_j=buddy_j,
    batch_idx=b,
    template_idx=s,
    obj_label=query_data.infos.label[b],
    patch_size=self.patch_size,
    num_patches_per_side=self.num_patches_per_side,
    vis_dir=Path(self.vis_dir),
)
```

## Test Block Update (lines ~2500-2600)

### Remove helper functions (defined in __main__)
- `denorm_img()` → Use `viz.denormalize_image()`
- `compute_pose_angle()` → Use `viz.compute_pose_angle()`

### Replace visualization loop (lines ~2509-2590)
```python
# OLD: Large inline visualization code

# NEW: Simple call to viz module
print(f"\n✓ Generating per-sample visualizations...")
saved_paths = viz.visualize_batch_all_samples(
    batch=batch,
    dataset_name=dataset_name,
    save_dir=save_dir,
    seed=SEED,
    max_templates=None,  # Show all templates
)
print(f"✓ All visualizations saved to {save_dir / 'per_sample_visualizations'}")
```

## Benefits of This Refactor

1. **Separation of Concerns**: Data loading and visualization are now separate
2. **Reusability**: Visualization functions can be used from anywhere (training, testing, notebooks)
3. **Flexibility**: Easy to support any number of positive/negative templates
4. **Maintainability**: Visualization logic is centralized in one module
5. **Testing**: Easier to test visualization independently from data loading

## Implementation Steps

1. Update the 3 call sites in `compute_flow_labels_for_train()`
2. Remove the visualization methods from VPOGDataset class
3. Update the `__main__` test block
4. Test with: `python training/dataloader/vpog_dataset.py`

## Notes

- The new viz functions are already flexible with template numbers
- No changes needed to VPOGBatch structure
- All visualization parameters are explicitly passed (no hidden class state)
