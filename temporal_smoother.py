"""EMA-based temporal smoothing for video style transfer.

Provides zero-cost temporal consistency by maintaining moving-average
caches of latent-space signals across consecutive video frames.

The key insight: smoothing in latent space (content features, modulation
signals) avoids the ghosting artifacts of pixel-space smoothing while
effectively suppressing flickering caused by frame-to-frame feature jitter.

Usage:
    smoother = TemporalSmoother(momentum=0.7)
    for frame in video:
        content_feats = content_encoder(frame)
        content_feats = smoother.smooth_content(content_feats)
        ...
        output = decoder(content_feats, style_feats, w, b, alpha)
    smoother.reset()  # call on scene cuts
"""

import torch


class TemporalSmoother:
    """EMA-based temporal smoother for video style transfer signals.

    Maintains exponential moving average buffers for three signal types:
    1. Content encoder features — PRIMARY flicker source (frame→feature jitter)
    2. Style encoder features — when style image changes gradually
    3. Modulator outputs (w, b) — numerical stability for modulation signals

    All smoothing operates in latent space using EMA, producing zero
    additional computational cost beyond a few tensor additions and multiplies.
    """

    def __init__(self, momentum=0.7):
        """Initialize the temporal smoother.

        Args:
            momentum: EMA blending weight in [0, 1].
                      Higher values = stronger smoothing, more temporal stability
                      but slightly slower response to genuine changes.
                      Recommended range: 0.5 (light) to 0.9 (heavy smoothing).
                      Default 0.7 works well for most content.
        """
        if not 0 <= momentum <= 1:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        self.momentum = momentum
        self.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(prev, curr, momentum):
        """Exponential moving average: blended = momentum*prev + (1-momentum)*curr."""
        return momentum * prev + (1.0 - momentum) * curr

    @staticmethod
    def _smooth_list(prev_list, curr_list, momentum):
        """Apply EMA to corresponding tensors in two equal-length lists.

        On first call (prev_list is None), returns curr_list unchanged and
        caches deep copies.
        """
        if prev_list is None:
            return curr_list, [c.clone() for c in curr_list]

        smoothed = []
        for p, c in zip(prev_list, curr_list):
            smoothed.append(TemporalSmoother._ema(p, c, momentum))
        # Detach from computation graph: each frame is independent
        cache = [s.detach() for s in smoothed]
        return smoothed, cache

    # ------------------------------------------------------------------
    # Public smoothing API — one method per signal type
    # ------------------------------------------------------------------

    def smooth_content(self, content_feats):
        """Smooth content encoder features across frames.

        Content features are the PRIMARY source of temporal flickering:
        adjacent frames produce slightly different encoder outputs, which
        the decoder's nonlinear processing amplifies into visible texture
        jitter. Smoothing here attacks the problem at its root.

        Args:
            content_feats: list of [feat_shallow, feat_deep] from Encoder
                           Shapes: [1, C, H, W] each

        Returns:
            Temporally smoothed content features (same structure as input)
        """
        smoothed, self.prev_content_feats = self._smooth_list(
            self.prev_content_feats, content_feats, self.momentum
        )
        return smoothed

    def smooth_style(self, style_feats):
        """Smooth style encoder features across frames.

        Essential when the style image changes during video playback
        (e.g., style interpolation, multi-style transitions). When the
        style image is fixed, this becomes a no-op after the first frame.

        Args:
            style_feats: list of [feat_shallow, feat_deep] from Encoder
                         Shapes: [1, C, H, W] each

        Returns:
            Temporally smoothed style features
        """
        smoothed, self.prev_style_feats = self._smooth_list(
            self.prev_style_feats, style_feats, self.momentum
        )
        return smoothed

    def smooth_modulation(self, weights, biases):
        """Smooth modulator output signals across frames.

        Smooths both filter weights and biases from the Modulator.
        Useful for suppressing micro-jitter in modulation parameters,
        especially when numerical precision differences between frames
        would otherwise cause visible artifacts.

        Args:
            weights: list of [w_shallow, w_deep] from Modulator,
                     each shape [1, C, 1, 1]
            biases:  list of [b_shallow, b_deep] from Modulator,
                     each shape [1, C, 1, 1]

        Returns:
            (smoothed_weights, smoothed_biases) — same structure as inputs
        """
        smoothed_w, self.prev_weights = self._smooth_list(
            self.prev_weights, weights, self.momentum
        )
        smoothed_b, self.prev_biases = self._smooth_list(
            self.prev_biases, biases, self.momentum
        )
        return smoothed_w, smoothed_b

    def reset(self):
        """Reset all temporal buffers.

        Call this on scene cuts, shot boundaries, or when starting a new
        video to prevent signal bleeding across unrelated frames.
        """
        self.prev_content_feats = None
        self.prev_style_feats = None
        self.prev_weights = None
        self.prev_biases = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_first_frame(self):
        """True if no previous frame has been cached (i.e., first frame)."""
        return self.prev_content_feats is None

    def __repr__(self):
        return f"TemporalSmoother(momentum={self.momentum:.2f})"
