"""Temporal-mixing perturbation guidance: ACG for the convolutional TemporalUnet.

ACG (Park et al., ICRA 2026, arXiv 2510.22201) builds an *incoherent* velocity
field by replacing the self-attention map over action tokens with the identity,
so every action token only sees itself, and extrapolates away from it:

    v_ACG = (1 + λ) v_θ(x, t) − λ v_θ^IC(x, t)                 (ACG Eq. 8)

The ``fm_baseline.models.TemporalUnet`` used here has no attention (``attention: false``).
Information is mixed along the planning horizon only by the temporal Conv1d
kernels (size 5 in ``Conv1dBlock``).  The conv analogue of "each token only
attends to itself" is therefore to keep only the centre tap of those kernels:

    W~[:, :, k] = W[:, :, k]  if k = K // 2,   0 otherwise

i.e. a 1x1 convolution with the centre weights.  Channel mixing, the time
embedding, GroupNorm and the residual path are untouched, like ACG keeps the
token-wise MLPs.  A second mode, ``skip``, replaces a whole
``ResidualTemporalBlock`` by its residual path (STG-R, Hyung et al., CVPR 2025).

Nothing in the model code or in the checkpoint is modified: the perturbation is a
forward hook that only rewrites the outputs of the *second half* of a batch.
``ACGModel`` runs [x, x] in one forward pass, so v_θ and v_θ^IC cost one batched
call, and GroupNorm (per sample) keeps the two halves independent.  With
``scale = 0`` or no site selected, ``ACGModel`` returns ``model(x, t)`` exactly.

Sites (levels of the U-Net; resolutions for Pushing, horizon 16, dim_mults 1,4,8)

    down0 16 | down1 8 | down2 4 | mid 4 | up0 4 | up1 8 | final 16

``up{i}`` blocks run *before* their upsample, so ``up1`` works at resolution 8
and its output is upsampled to 16.  ``final`` is the last k=5 Conv1dBlock.
Downsample / Upsample convolutions are never perturbed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MODES = ("center", "skip")


def site_modules(model):
    """Ordered {site: [modules]} for an fm_baseline.models TemporalUnet."""
    sites = {}
    for i, (res1, res2, _attn, _down) in enumerate(model.downs):
        sites[f"down{i}"] = [res1, res2]
    sites["mid"] = [model.mid_block1, model.mid_block2]
    for i, (res1, res2, _attn, _up) in enumerate(model.ups):
        sites[f"up{i}"] = [res1, res2]
    sites["final"] = [model.final_conv[0]]
    return sites


@torch.no_grad()
def site_resolutions(model, horizon, transition_dim, device="cpu"):
    """Temporal length seen by each site (one dummy forward)."""
    seen, handles = {}, []
    for name, mods in site_modules(model).items():
        def hook(_m, inp, _out, name=name):
            seen.setdefault(name, int(inp[0].shape[-1]))
        handles.append(mods[0].register_forward_hook(hook))
    try:
        model(torch.zeros(1, horizon, transition_dim, device=device), torch.zeros((), device=device))
    finally:
        for h in handles:
            h.remove()
    return seen


class TemporalPerturbation:
    """Forward hooks that make the selected sites temporally incoherent.

    ``split = None``  hooks are inert (bit-identical model)
    ``split = s``     rows ``s:`` of the batch get the perturbed output
    """

    def __init__(self, model, sites, mode="center"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        table = site_modules(model)
        unknown = [s for s in sites if s not in table]
        if unknown:
            raise ValueError(f"unknown ACG sites {unknown}; available: {list(table)}")
        if not sites:
            raise ValueError("no ACG site selected")
        if mode == "skip" and "final" in sites:
            raise ValueError("skip mode on 'final' removes the output block (not a perturbation); use center")
        self.sites, self.mode, self.split = list(sites), mode, None
        self.n_hooked = 0
        self._handles = []
        for s in self.sites:
            for m in table[s]:
                if mode == "center":
                    for conv in m.modules():
                        if isinstance(conv, nn.Conv1d) and conv.kernel_size[0] > 1:
                            self._handles.append(conv.register_forward_hook(self._center_hook))
                            self.n_hooked += 1
                else:
                    self._handles.append(m.register_forward_hook(self._skip_hook))
                    self.n_hooked += 1

    # -- hooks -----------------------------------------------------------------
    def _center_hook(self, conv, inp, out):
        s = self.split
        if s is None:
            return None
        c = conv.kernel_size[0] // 2
        w = conv.weight[:, :, c:c + 1]
        out = out.clone()
        out[s:] = F.conv1d(inp[0][s:], w, conv.bias, stride=conv.stride, groups=conv.groups)
        return out

    def _skip_hook(self, block, inp, out):
        s = self.split
        if s is None:
            return None
        x = inp[0][s:]
        res = block.residual_conv(x) if hasattr(block, "residual_conv") else x   # final Conv1dBlock: identity
        out = out.clone()
        out[s:] = res
        return out

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def describe(self):
        return {"sites": self.sites, "mode": self.mode, "hooked_modules": self.n_hooked}


class ACGModel:
    """Callable with the ``model(x, t)`` signature used by every sampler here.

    v = (1 + λ) v_θ − λ v_θ^IC   for t in [t_min, t_max),  plain v_θ otherwise.

    Guidance acts on the whole plan tensor (actions and predicted observations),
    as ACG acts on the whole action chunk.  The samplers zero the conditioned
    t=0 observation row of the returned velocity afterwards, as for v_θ.
    """

    def __init__(self, model, perturbation: TemporalPerturbation, scale, t_min=0.0, t_max=1.0):
        self.model, self.pert = model, perturbation
        self.scale, self.t_min, self.t_max = float(scale), float(t_min), float(t_max)
        self.calls = 0
        self.guided_calls = 0
        self._gap = 0.0          # running sum of ||v − v_IC|| / ||v||  (how different the IC field is)

    def __call__(self, x, t):
        self.calls += 1
        tf = float(t)
        if self.scale == 0.0 or not (self.t_min <= tf < self.t_max):
            return self.model(x, t)
        b = x.shape[0]
        self.pert.split = b
        try:
            out = self.model(torch.cat([x, x], dim=0), t)
        finally:
            self.pert.split = None
        v, v_ic = out[:b], out[b:]
        self.guided_calls += 1
        self._gap += float((v - v_ic).norm() / (v.norm() + 1e-12))
        return (1.0 + self.scale) * v - self.scale * v_ic

    def reset_stats(self):
        self.calls = self.guided_calls = 0
        self._gap = 0.0

    def stats(self):
        g = self.guided_calls
        return {"acg_guided_calls": g, "acg_rel_gap": self._gap / g if g else 0.0}

    def describe(self):
        return {"name": "acg", "scale": self.scale, "t_range": [self.t_min, self.t_max], **self.pert.describe()}


class PerturbedModel:
    """The incoherent model alone (diagnostics only): v_θ^IC(x, t) for the whole batch."""

    def __init__(self, model, perturbation: TemporalPerturbation):
        self.model, self.pert = model, perturbation

    def __call__(self, x, t):
        self.pert.split = 0
        try:
            return self.model(x, t)
        finally:
            self.pert.split = None
