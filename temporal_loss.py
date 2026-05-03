"""Temporal consistency loss for video style transfer training.

Provides loss functions that penalize texture inconsistency across adjacent
video frames. The key idea: warp the stylized output of frame t to align
with frame t+1 using optical flow, then penalize differences in static
(non-occluded) regions.

This encourages the model to produce temporally stable stylizations at the
weight level, complementing the inference-time EMA smoothing in
temporal_smoother.py.

Requires a pre-computed optical flow between adjacent frames. The flow can
be obtained from any optical flow estimator (RAFT, FlowNet, PWC-Net, etc.).

Usage (conceptual, integrate into training loop):
    temporal_loss_fn = TemporalConsistencyLoss()
    ...
    for content_pair, flow in dataloader:  # content_pair: [B, 2, 3, H, W]
        content_t = content_pair[:, 0]      # frame t
        content_t1 = content_pair[:, 1]     # frame t+1
        flow_t_to_t1 = flow                 # optical flow from t to t+1

        output_t = model(content_t, style)
        output_t1 = model(content_t1, style)

        loss_temp = temporal_loss_fn(output_t, output_t1, flow_t_to_t1)
        total_loss = loss_content + loss_style + lambda_temp * loss_temp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _backward_warp(image, flow, mode='bilinear'):
    """Warp an image using backward optical flow.

    Moves pixels from source locations to target locations:
        output(x, y) = image(x + flow_x, y + flow_y)

    Args:
        image:  [B, C, H, W] source image
        flow:   [B, 2, H, W] optical flow (dx, dy) from target to source
        mode:   interpolation mode ('bilinear' or 'nearest')

    Returns:
        Warped image [B, C, H, W]
    """
    B, C, H, W = image.shape
    # Build normalized sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=image.device, dtype=image.dtype),
        torch.arange(W, device=image.device, dtype=image.dtype),
        indexing='ij'
    )
    # Normalize to [-1, 1]
    grid_x = 2.0 * grid_x / (W - 1) - 1.0
    grid_y = 2.0 * grid_y / (H - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

    # Add normalized flow offsets
    flow_norm = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
    flow_norm[..., 0] = flow_norm[..., 0] * 2.0 / (W - 1)
    flow_norm[..., 1] = flow_norm[..., 1] * 2.0 / (H - 1)
    grid = grid.unsqueeze(0) + flow_norm  # [B, H, W, 2]

    return F.grid_sample(image, grid, mode=mode, padding_mode='border',
                         align_corners=True)


def _flow_consistency_mask(flow_fwd, flow_bwd, threshold=1.0):
    """Compute occlusion mask by checking forward-backward flow consistency.

    Pixels where the forward and backward flows disagree are likely
    occluded and should be excluded from the temporal loss.

    Args:
        flow_fwd: [B, 2, H, W] flow from t to t+1
        flow_bwd: [B, 2, H, W] flow from t+1 to t
        threshold: maximum allowed discrepancy in pixels

    Returns:
        mask: [B, 1, H, W] — 1.0 for consistent regions, 0.0 for occluded
    """
    # Warp forward flow by backward flow
    flow_fwd_warped = _backward_warp(flow_fwd, flow_bwd)
    # Check consistency
    diff = torch.norm(flow_fwd_warped + flow_bwd, dim=1, keepdim=True)
    mask = (diff < threshold).float()
    return mask


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for video style transfer training.

    Penalizes the model for producing different textures in static regions
    of adjacent frames. The loss works by:
    1. Warping output_t to align with output_t+1 using optical flow
    2. Computing per-pixel differences between warped output_t and output_t+1
    3. Masking out occluded regions (where flow is unreliable)
    4. Averaging the masked difference

    This pushes the model to learn temporally stable feature representations
    at the weight level, reducing flickering without any runtime smoothing.
    """

    def __init__(self, loss_type='l1', occlusion_threshold=1.0):
        """Initialize the temporal consistency loss.

        Args:
            loss_type: 'l1' for L1 loss (robust to outliers, recommended) or
                       'l2' for MSE (penalizes large differences more heavily)
            occlusion_threshold: flow consistency threshold in pixels.
                                 Smaller = stricter mask.
        """
        super().__init__()
        if loss_type not in ('l1', 'l2'):
            raise ValueError(f"loss_type must be 'l1' or 'l2', got {loss_type}")
        self.loss_type = loss_type
        self.occlusion_threshold = occlusion_threshold

    def forward(self, output_t, output_t1, flow_t_to_t1, flow_t1_to_t=None):
        """Compute temporal consistency loss between two consecutive outputs.

        Args:
            output_t:   [B, 3, H, W] stylized output for frame t
            output_t1:  [B, 3, H, W] stylized output for frame t+1
            flow_t_to_t1: [B, 2, H, W] optical flow from frame t to t+1
            flow_t1_to_t: [B, 2, H, W] optical flow from frame t+1 to t.
                          If provided, enables occlusion masking via
                          forward-backward consistency check. If None,
                          uses a simple magnitude-based heuristic instead.

        Returns:
            Scalar temporal consistency loss
        """
        # Warp output_t to align with output_t+1
        output_t_warped = _backward_warp(output_t, flow_t_to_t1)

        # Compute per-pixel difference
        if self.loss_type == 'l1':
            diff = torch.abs(output_t_warped - output_t1).mean(dim=1, keepdim=True)
        else:  # l2
            diff = (output_t_warped - output_t1).pow(2).mean(dim=1, keepdim=True)

        # Occlusion mask: exclude regions where flow is unreliable
        if flow_t1_to_t is not None:
            # High-quality masking via forward-backward consistency
            mask = _flow_consistency_mask(
                flow_t_to_t1, flow_t1_to_t, self.occlusion_threshold
            )
        else:
            # Fallback: mask out large-displacement regions
            # (large displacements are more likely to be occluded or errors)
            flow_mag = torch.norm(flow_t_to_t1, dim=1, keepdim=True)
            mask = (flow_mag < 20.0).float()  # heuristic: 20-pixel threshold

        # Masked mean over valid pixels
        mask = mask.detach()
        masked_diff = diff * mask
        # Avoid division by zero
        mask_sum = mask.sum() + 1e-8
        loss = masked_diff.sum() / mask_sum

        return loss

    def extra_repr(self):
        return (f"loss_type={self.loss_type}, "
                f"occlusion_threshold={self.occlusion_threshold}")
