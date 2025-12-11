# VPOG Architecture: Added Tokens with rope_mask System

## Critical Design Change

**Original Design (INCORRECT):**
- Unseen tokens were added AFTER AA module in classification head
- This meant unseen tokens never went through attention aggregation
- Unseen tokens had no positional encoding (correct part)

**New Design (CORRECT):**
- Added tokens (e.g., unseen) are added BEFORE AA module
- They go through attention aggregation WITH all other tokens
- BUT they skip positional encoding using `rope_mask` system
- After AA, they can be separated again or kept concatenated

## Key Components

### 1. TokenManager (`vpog/models/token_manager.py`)

Manages configurable added tokens that bypass RoPE:

```python
token_manager = TokenManager(
    num_query_added_tokens=0,      # Currently 0 for query
    num_template_added_tokens=1,   # 1 unseen token per template
    embed_dim=768,
)

# Add tokens before AA module
features_with_tokens, pos_with_tokens, rope_mask = \
    token_manager.add_query_tokens(features, pos2d)
# rope_mask: [B, N+added] - True for patches, False for added tokens

# Remove tokens after AA module (if needed)
features_only, added_features = \
    token_manager.remove_added_tokens(features_after_aa, is_query=True)
```

### 2. rope_mask System in AA Module

Every attention layer in AA module now accepts `rope_mask`:

```python
# In GlobalAttention and LocalAttention
def forward(self, x, pos2d, rope_mask=None, ...):
    # Apply RoPE only where rope_mask==True
    if rope_mask is not None:
        # For templates: [B, N] -> [B, N, 1, 1]
        mask = rope_mask[:, :, None, None]
        x_rope = apply_rope(x, pos2d)
        x = torch.where(mask, x_rope, x)  # Skip RoPE where mask==False
    else:
        x = apply_rope(x, pos2d)
```

### 3. Updated VPOG Model Flow

```python
# 1. Encode images
query_features, query_pos2d = encoder(query_images)
template_features, template_pos2d = encoder(template_images)

# 2. Add special tokens (NEW STEP)
query_with_tokens, query_pos_with_tokens, query_rope_mask = \
    token_manager.add_query_tokens(query_features, query_pos2d)

templates_with_tokens, templates_pos_with_tokens, templates_rope_mask = \
    token_manager.add_template_tokens(template_features, template_pos2d)

# 3. AA module with rope_mask
query_aa = aa_module(
    query_with_tokens,
    pos2d=query_pos_with_tokens,
    rope_mask=query_rope_mask,  # NEW
    is_template=False,
)

template_aa = aa_module(
    templates_with_tokens,
    pos2d=templates_pos_with_tokens,
    rope_mask=templates_rope_mask,  # NEW
    is_template=True,
)

# 4. Remove added tokens after AA
query_patches, query_added = token_manager.remove_added_tokens(query_aa, is_query=True)
template_patches, template_added = token_manager.remove_added_tokens(template_aa, is_query=False)

# 5. Classification head (added tokens concatenated back)
templates_for_classification = torch.cat([template_patches, template_added], dim=2)
logits = classification_head(query_patches, templates_for_classification)
# Output: [B, S, Nq, Nt+num_added]

# 6. Flow head (patches only, no added tokens)
flow = flow_head(query_patches, template_patches)
# Output: [B, S, Nq, Nt, 16, 16, 2]
```

## Configuration

In `configs/model/vpog.yaml`:

```yaml
model:
  encoder:
    # Added tokens configuration
    num_query_added_tokens: 0      # No added tokens for query
    num_template_added_tokens: 1   # 1 unseen token per template
```

## Benefits of New Architecture

1. **Attention Context**: Added tokens participate in attention with all patches
   - Unseen token can aggregate information from all patches
   - Better representation learning for "no match" case

2. **Configurable**: Easy to add more special tokens
   - Could add tokens for occluded regions
   - Could add tokens for uncertainty estimation
   - Just change `num_query_added_tokens` / `num_template_added_tokens`

3. **Flexible Positional Encoding**:
   - Patches get spatial encoding (RoPE2D or S²RoPE)
   - Added tokens skip encoding (rope_mask=False)
   - No manual token position management needed

4. **Clean Separation**: 
   - TokenManager handles all token addition/removal
   - AA module handles attention (doesn't care about token types)
   - Classification head handles matching (receives all tokens)
   - Flow head handles flow (receives only patches)

## Implementation Details

### rope_mask Broadcasting

Different shapes for different attention types:

**Global Attention (Templates with S²RoPE):**
```python
# Input: rope_mask [B, N]
# Reshape: [B, N] -> [B, N, 1, 1]
# Applies to: [B, N, num_heads, head_dim]
mask = rope_mask[:, :, None, None]
x_encoded = torch.where(mask, x_rope, x)
```

**Global Attention (Query with RoPE2D):**
```python
# Input: rope_mask [B, N]
# Reshape: [B, N] -> [B, 1, N, 1]
# Applies to: [B, num_heads, N, head_dim]
mask = rope_mask[:, None, :, None]
x_encoded = torch.where(mask, x_rope, x)
```

**Local Attention (Windowed):**
```python
# Input: rope_mask [B, N]
# Window partition: [B, N] -> [B, H, W] -> [B*nW, ws*ws]
rope_mask_2d = rope_mask.view(B, H, W)
rope_mask_windows = window_partition(rope_mask_2d, window_size)
# Apply within each window: [B*nW, ws*ws]
```

## Testing

See `vpog/models/test_vpog_integration.py` for comprehensive tests:

1. **TokenManager Integration**: Verifies token addition/removal
2. **Forward Pass**: Tests complete VPOG model with added tokens
3. **rope_mask Effectiveness**: Confirms added tokens skip RoPE

## Migration Notes

Files modified for this architecture:

1. **vpog/models/aa_module.py**: Added `rope_mask` parameter throughout
   - `GlobalAttention.forward()`: Conditional RoPE application
   - `LocalAttention.forward()`: Conditional RoPE with window partitioning
   - `AABlock.forward()`: Propagate rope_mask
   - `AAModule.forward()`: Pass rope_mask to all blocks

2. **vpog/models/token_manager.py**: NEW - Manages added tokens
   - `add_query_tokens()`: Adds tokens to query
   - `add_template_tokens()`: Adds tokens to templates
   - `remove_added_tokens()`: Separates patches from added tokens

3. **vpog/models/classification_head.py**: Updated to expect added tokens
   - Removed internal unseen token addition
   - Changed `Nt+1` to `Nt_with_added` (flexible)
   - Expects templates with added tokens already concatenated

4. **vpog/models/vpog_model.py**: Integrated TokenManager
   - Added TokenManager instantiation in `__init__`
   - Added token addition before AA module
   - Added token removal after AA module
   - Updated classification head to receive added tokens
   - Flow head receives patches only (no added tokens)

5. **configs/model/vpog.yaml**: NEW - Configuration file
   - Added `num_query_added_tokens` and `num_template_added_tokens`
   - Currently: 0 query tokens, 1 template token (unseen)

## Future Extensions

Easy to extend with more token types:

```yaml
# Example: Multiple special tokens
num_query_added_tokens: 2      # e.g., [occluded, uncertain]
num_template_added_tokens: 3   # e.g., [unseen, partial, truncated]
```

TokenManager automatically handles any number of added tokens with correct rope_mask generation.
