"""
Flow-matching model for PianoMotion10M hand-pose generation, conditioned on audio.

READ GUIDE.md FIRST. Short version of the assumptions baked into this file:

- You have cloned https://github.com/agnJason/PianoMotion10M somewhere, and
  train.py adds it to sys.path so `datasets.PianoPose.PianoPose` is importable.
- PianoPose's __getitem__ returns a dict with keys 'audio', 'right', 'left'
  (confirmed from the repo's own draw.py usage). 'right'/'left' are treated
  as [T, D_hand] pose tensors; D_hand is INFERRED at runtime from the data,
  never hardcoded here, because we don't know the exact column count with
  certainty (see GUIDE.md on this).
- 'audio' is treated as a 1D raw waveform tensor.
- Right + left hand are modeled jointly by concatenating them into one
  [T, D] pose vector per frame. This is a v1 simplification -- see GUIDE.md
  "possible upgrades" for the decoupled dual-stream alternative discussed
  earlier in our conversation.
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Turns a scalar flow-time t in [0,1] into a small vector embedding."""

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "embedding dim must be even"
        self.dim = dim

    def forward(self, t):
        # t: [B, 1] -> [B, dim]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t * freqs.unsqueeze(0)  # [B, half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]


class AudioEncoder(nn.Module):
    """
    Small 1D-conv encoder over raw audio, producing one embedding per output
    frame, resampled in time to exactly match the pose sequence length.

    Deliberately NOT a pretrained model (no wav2vec2/HuBERT dependency) so v1
    has zero external model downloads to fight with. PianoMotion10M's own
    baseline uses a pretrained audio backbone -- swapping one in here is the
    natural upgrade once this whole pipeline runs end to end. See GUIDE.md.
    """

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(64, hidden_dim, kernel_size=9, stride=4, padding=4), nn.GELU(),
        )

    def forward(self, audio, num_frames):
        """
        audio: [B, T_audio] raw waveform
        num_frames: target sequence length to resample to (the pose window length)
        returns: [B, num_frames, hidden_dim]
        """
        x = audio.unsqueeze(1)  # [B, 1, T_audio]
        x = self.net(x)  # [B, hidden_dim, T_conv]
        x = torch.nn.functional.interpolate(
            x, size=num_frames, mode="linear", align_corners=False
        )
        x = x.transpose(1, 2)  # [B, num_frames, hidden_dim]
        return x


class FlowMatchingTransformer(nn.Module):
    """
    Predicts velocity v_theta(x_t, t, c) for a [B, T, pose_dim] noisy pose
    sequence, conditioned on a per-frame audio embedding c.

    Structure: self-attention across time, cross-attention to the audio
    conditioning, repeated for num_layers blocks. Flow time t is injected
    once per block via addition to every frame's representation (simplest
    version -- adaptive layer norm is a documented upgrade, not required
    for v1).
    """

    def __init__(self, pose_dim, hidden_dim=256, num_layers=4, num_heads=4, max_frames=2000):
        super().__init__()
        self.pose_in = nn.Linear(pose_dim, hidden_dim)
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames, hidden_dim) * 0.02)

        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.cross_norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.out_proj = nn.Linear(hidden_dim, pose_dim)

    def forward(self, x_t, t, cond):
        """
        x_t:  [B, T, pose_dim]   noisy pose sequence
        t:    [B, 1]             flow time, one scalar per batch item
        cond: [B, T, hidden_dim] per-frame conditioning from AudioEncoder
        returns: [B, T, pose_dim] predicted velocity
        """
        B, T, _ = x_t.shape
        h = self.pose_in(x_t) + self.pos_embed[:, :T, :]

        t_emb = self.time_mlp(self.time_embed(t))  # [B, hidden_dim]
        h = h + t_emb.unsqueeze(1)  # broadcast the same t across every frame

        for self_attn, cross_attn, norm in zip(
            self.self_attn_layers, self.cross_attn_layers, self.cross_norm_layers
        ):
            h = self_attn(h)
            attn_out, _ = cross_attn(h, cond, cond)
            h = norm(h + attn_out)

        return self.out_proj(h)


class BaselineRegressor(nn.Module):
    """
    Same shape of encoder/conditioning as FlowMatchingTransformer, but a
    direct MSE regressor: no noise, no t, straight audio-conditioning to
    pose. This is the "naive" comparison model from our plan (Phase 5) --
    train it and compare its output against the flow-matching model's output
    on the same out-of-training audio (e.g. Fur Elise) to check for the
    mode-averaging / blurriness the whole project exists to avoid.
    """

    def __init__(self, pose_dim, hidden_dim=256, num_layers=4, num_heads=4, max_frames=2000):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, max_frames, hidden_dim) * 0.02)
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.cross_norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hidden_dim, pose_dim)

    def forward(self, cond):
        """
        cond: [B, T, hidden_dim] per-frame conditioning from AudioEncoder
        returns: [B, T, pose_dim] directly predicted pose (no flow matching)
        """
        B, T, _ = cond.shape
        h = self.query[:, :T, :].expand(B, -1, -1)
        for self_attn, cross_attn, norm in zip(
            self.self_attn_layers, self.cross_attn_layers, self.cross_norm_layers
        ):
            h = self_attn(h)
            attn_out, _ = cross_attn(h, cond, cond)
            h = norm(h + attn_out)
        return self.out_proj(h)
