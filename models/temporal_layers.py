"""
Delay-space neural network layers for temporal arithmetic CNNs.

Implements the temporal arithmetic framework from:
  "Energy Efficient Convolutions with Temporal Arithmetic"
  (Gretsch et al., ASPLOS '24)

All computation happens in delay space where x' = -ln(x) (Eq. 1-2).

Core delay-space operation mapping:
  Multiplication:  x * y   ->  x' + y'               (delay addition)
  Addition:        x + y   ->  nLSE(x', y')           (Eq. 4, Eq. 6 approx)
  Subtraction:     x - y   ->  nLDE(x', y')           (Eq. 5, Eq. 7 approx)

Negative numbers use split-value representation <x_pos, x_neg>
(paper Sections 2.2, 4.4).

Delay-space addition uses the hardware-faithful min-of-max nLSE
approximation from paper Eq. 6 (see ``utils.temporal_artithmetic.nlse``):

    nLSE(x', y') ~= min(x', y', max_i(x' + C_i, y' + D_i))

with the (C, D) constants loaded from ``constants/orig_constants.pt``.
The n-ary reduction required by conv / linear layers is performed by
folding this pairwise approximation over the kernel dimension via a
tree reduction.  No ``logsumexp`` is used anywhere inside the layers.
"""

import math
from pathlib import Path
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPS = 1e-7
DELAY_CAP = 50.0
NLSE_MAX_TERMS = 7
NLDE_MAX_TERMS = 7


# ---------------------------------------------------------------------------
# nLSE approximation constants (paper Eq. 6)
# ---------------------------------------------------------------------------
# Loaded once at import time from the shared constants bundle so every
# delay-space layer uses the same (C, D) vectors that `utils.temporal_artithmetic`
# and the test / optimisation scripts use.  Lazily moved to the right
# device / dtype on first use and cached.
_CONSTANTS_PATH = (
    Path(__file__).resolve().parent.parent / "constants" / "orig_constants.pt"
)
_constants_bundle = torch.load(_CONSTANTS_PATH, map_location="cpu")
_C_BASE = _constants_bundle["C_VALUES"][NLSE_MAX_TERMS].reshape(-1).detach()
_D_BASE = _constants_bundle["D_VALUES"][NLSE_MAX_TERMS].reshape(-1).detach()
_CONST_CACHE: dict = {}


def set_nlse_constants(C: torch.Tensor, D: torch.Tensor) -> None:
    """Override the (C, D) vectors used by the delay-space nLSE approximation.

    Useful for swapping between ``orig_constants.pt`` and ``learned_constants.pt``
    at runtime without editing the module.  Shape must be 1-D of equal length.
    """
    global _C_BASE, _D_BASE, _CONST_CACHE
    C_flat = C.reshape(-1).detach().cpu()
    D_flat = D.reshape(-1).detach().cpu()
    if C_flat.shape != D_flat.shape:
        raise ValueError("C and D must have the same number of terms.")
    _C_BASE, _D_BASE = C_flat, D_flat
    _CONST_CACHE = {}


def _get_cd(device: torch.device, dtype: torch.dtype):
    key = (device, dtype)
    cached = _CONST_CACHE.get(key)
    if cached is None:
        cached = (
            _C_BASE.to(device=device, dtype=dtype),
            _D_BASE.to(device=device, dtype=dtype),
        )
        _CONST_CACHE[key] = cached
    return cached


# ---------------------------------------------------------------------------
# SplitValue — signed delay-space representation  (paper Section 2.2, 4.4)
# ---------------------------------------------------------------------------
class SplitValue(NamedTuple):
    """Pair <pos, neg> of delay-space tensors representing a signed value.

    importance-space value = exp(-pos) - exp(-neg).
    A delay of DELAY_CAP (≈ inf) means zero importance.
    """

    pos: torch.Tensor
    neg: torch.Tensor


# ---------------------------------------------------------------------------
# Primitive delay-space operations
# ---------------------------------------------------------------------------

def to_delay(x: torch.Tensor) -> torch.Tensor:
    """Importance -> delay:  x' = -ln(x)  (Eq. 1)."""
    return -torch.log(torch.clamp(x, min=EPS))


def to_importance(x_p: torch.Tensor) -> torch.Tensor:
    """Delay -> importance:  x = e^{-x'}  (Eq. 2)."""
    return torch.exp(-x_p)


def nlse_2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise min-of-max nLSE approximation (paper Eq. 6).

    Shape-agnostic version of ``utils.temporal_artithmetic.nlse``: ``a`` and
    ``b`` may be any broadcast-compatible delay tensors.

    **Hardware role of the K-shift.** On silicon, a delay is a physical time
    offset: delay lines, arbiters, and pulse generators only represent
    *non-negative* delays. The optimised constants ``C_i``, ``D_i`` are
    negative offsets; without lifting the operands, quantities such as
    ``x' + C_i`` can fall below zero and have no direct physical encoding.
    The uniform shift ``K = -min(C ∪ D)`` is applied to *both* inputs before
    the min-of-max network so every intermediate that must be realised as a
    delay in the hardware stays in the non-negative range the circuits
    implement. The same shift is subtracted at the output so the *logical*
    delay (what Eq. 6 denotes) is unchanged—translation in delay space is an
    overall scaling in importance space that must cancel at the end.

    In IEEE-754 this ``+K`` / ``-K`` pair is algebraically redundant (the
    min/max structure is translation-invariant), but we keep it so
    simulation matches the hardware datapath and stays numerically aligned
    with ``utils.temporal_artithmetic.nlse``.
    """
    a = torch.clamp(a, max=DELAY_CAP)
    b = torch.clamp(b, max=DELAY_CAP)

    C, D = _get_cd(a.device, a.dtype)
    K = -torch.min(torch.cat((C, D)))

    a = a + K
    b = b + K

    # Larger delay (smaller importance) pairs with C; smaller delay with D.
    hi = torch.maximum(a, b)
    lo = torch.minimum(a, b)

    X = hi.unsqueeze(-1) + C  # (..., M)
    Y = lo.unsqueeze(-1) + D  # (..., M)
    max_terms_min = torch.maximum(X, Y).min(dim=-1).values  # min_i max(X_i, Y_i)

    out = torch.minimum(torch.minimum(hi, lo), max_terms_min)
    return out - K


def nary_nlse(delays: torch.Tensor, dim: int) -> torch.Tensor:
    """n-ary nLSE via pairwise tree reduction of the Eq. 6 approximation.

    The hardware operator is strictly 2-ary, so a kernel-sum of K delay
    values is realised by folding ``nlse_2`` over the operand dimension
    in a balanced binary tree (ceil(log2 K) rounds).  Odd-sized levels
    are padded with ``DELAY_CAP`` (zero importance), which is an
    identity element for nLSE.

    Large delays are clamped to ``DELAY_CAP`` so all-zero-importance
    edge cases don't overflow the shift term.
    """
    x = torch.clamp(delays, max=DELAY_CAP).movedim(dim, -1)

    while x.shape[-1] > 1:
        if x.shape[-1] % 2 == 1:
            pad = torch.full_like(x[..., :1], DELAY_CAP)
            x = torch.cat([x, pad], dim=-1)
        left = x[..., 0::2]
        right = x[..., 1::2]
        x = nlse_2(left, right)

    return x.squeeze(-1)


# ---------------------------------------------------------------------------
# Encoding / decoding layers
# ---------------------------------------------------------------------------

class DelayEncoder(nn.Module):
    """Importance-space [0, 1] tensor  ->  SplitValue in delay space."""

    def forward(self, x: torch.Tensor) -> SplitValue:
        return SplitValue(
            pos=to_delay(x),
            neg=torch.full_like(x, DELAY_CAP),
        )


class DelayDecoder(nn.Module):
    """SplitValue in delay space  ->  importance-space tensor."""

    def forward(self, sv: SplitValue) -> torch.Tensor:
        return to_importance(sv.pos) - to_importance(sv.neg)


# ---------------------------------------------------------------------------
# Activation & pooling
# ---------------------------------------------------------------------------

class TemporalReLU(nn.Module):
    """ReLU in split-value delay space: discard the negative component."""

    def forward(self, sv: SplitValue) -> SplitValue:
        return SplitValue(sv.pos, torch.full_like(sv.neg, DELAY_CAP))


class TemporalMaxPool2d(nn.Module):
    """Max-pool in delay space:  max importance  =  min delay.

    Implements min-pool via  min(x) = -max(-x).
    """

    def __init__(self, kernel_size: int, stride: Optional[int] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, sv: SplitValue) -> SplitValue:
        pos = torch.clamp(sv.pos, max=DELAY_CAP)
        neg = torch.clamp(sv.neg, max=DELAY_CAP)
        out_pos = -F.max_pool2d(-pos, self.kernel_size, self.stride)
        out_neg = -F.max_pool2d(-neg, self.kernel_size, self.stride)
        return SplitValue(out_pos, out_neg)


# ---------------------------------------------------------------------------
# Shape utilities
# ---------------------------------------------------------------------------

class TemporalFlatten(nn.Module):
    """Flatten spatial dims of a SplitValue:  (B,C,H,W) -> (B, C*H*W)."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, sv: SplitValue) -> SplitValue:
        return SplitValue(
            torch.flatten(sv.pos, self.start_dim, self.end_dim),
            torch.flatten(sv.neg, self.start_dim, self.end_dim),
        )


# ---------------------------------------------------------------------------
# Convolution  (paper Sections 4.3–4.4)
# ---------------------------------------------------------------------------

class TemporalConv2d(nn.Module):
    """2-D convolution operating entirely in delay space.

    Weights live in importance space and are split into positive/negative
    components then converted to delay each forward pass.

    For every output position the layer computes:
      - Multiply  (delay add):  product' = input' + weight'
      - Accumulate (nLSE):      sum products across the kernel

    With split values (Section 4.4):
      out_pos' = nLSE(in_pos' + w_pos',  in_neg' + w_neg')
      out_neg' = nLSE(in_pos' + w_neg',  in_neg' + w_pos')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, sv: SplitValue) -> SplitValue:
        B, C_in, H, W = sv.pos.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding

        # -- weights -> split-value delay --------------------------------
        w_pos_d = to_delay(torch.clamp(self.weight, min=0))       # (Co,Ci,kH,kW)
        w_neg_d = to_delay(torch.clamp(-self.weight, min=0))
        K = C_in * kH * kW
        w_pos_flat = w_pos_d.reshape(self.out_channels, K)        # (Co, K)
        w_neg_flat = w_neg_d.reshape(self.out_channels, K)

        # -- pad input with DELAY_CAP (zero importance) then unfold ------
        if pH > 0 or pW > 0:
            in_pos = F.pad(sv.pos, (pW, pW, pH, pH), value=DELAY_CAP)
            in_neg = F.pad(sv.neg, (pW, pW, pH, pH), value=DELAY_CAP)
        else:
            in_pos, in_neg = sv.pos, sv.neg

        ip = F.unfold(in_pos, self.kernel_size, stride=self.stride)  # (B, K, L)
        in_ = F.unfold(in_neg, self.kernel_size, stride=self.stride)
        L = ip.shape[2]

        # -- broadcast & product  (B,1,K,L) + (1,Co,K,1) ---------------
        ip = ip.unsqueeze(1)                          # (B, 1, K, L)
        in_ = in_.unsqueeze(1)
        wp = w_pos_flat[None, :, :, None]             # (1, Co, K, 1)
        wn = w_neg_flat[None, :, :, None]

        all_pos = torch.cat([ip + wp, in_ + wn], dim=2)   # (B, Co, 2K, L)
        all_neg = torch.cat([ip + wn, in_ + wp], dim=2)

        out_pos = nary_nlse(all_pos, dim=2)                # (B, Co, L)
        out_neg = nary_nlse(all_neg, dim=2)

        # -- bias --------------------------------------------------------
        if self.bias is not None:
            bp = to_delay(torch.clamp(self.bias, min=0))[None, :, None]
            bn = to_delay(torch.clamp(-self.bias, min=0))[None, :, None]
            out_pos = nlse_2(out_pos, bp)
            out_neg = nlse_2(out_neg, bn)

        # -- reshape to spatial ------------------------------------------
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1
        return SplitValue(
            out_pos.reshape(B, self.out_channels, H_out, W_out),
            out_neg.reshape(B, self.out_channels, H_out, W_out),
        )


# ---------------------------------------------------------------------------
# Fully-connected  (same split-value logic, matrix multiply)
# ---------------------------------------------------------------------------

class TemporalLinear(nn.Module):
    """Fully-connected layer in delay space."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, sv: SplitValue) -> SplitValue:
        w_pos_d = to_delay(torch.clamp(self.weight, min=0))     # (O, I)
        w_neg_d = to_delay(torch.clamp(-self.weight, min=0))

        inp = sv.pos[:, None, :]   # (B, 1, I)
        inn = sv.neg[:, None, :]
        wp = w_pos_d[None, :, :]   # (1, O, I)
        wn = w_neg_d[None, :, :]

        all_pos = torch.cat([inp + wp, inn + wn], dim=2)   # (B, O, 2I)
        all_neg = torch.cat([inp + wn, inn + wp], dim=2)

        out_pos = nary_nlse(all_pos, dim=2)                # (B, O)
        out_neg = nary_nlse(all_neg, dim=2)

        if self.bias is not None:
            bp = to_delay(torch.clamp(self.bias, min=0))[None, :]
            bn = to_delay(torch.clamp(-self.bias, min=0))[None, :]
            out_pos = nlse_2(out_pos, bp)
            out_neg = nlse_2(out_neg, bn)

        return SplitValue(out_pos, out_neg)


# ---------------------------------------------------------------------------
# Regularisation
# ---------------------------------------------------------------------------

class TemporalDropout(nn.Module):
    """Dropout in delay space.

    Dropped activations become zero importance (DELAY_CAP).
    Surviving activations are scaled by 1/(1-p) in importance space,
    i.e. their delay decreases by ln(1-p).
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, sv: SplitValue) -> SplitValue:
        if not self.training or self.p == 0.0:
            return sv
        keep = torch.bernoulli(
            torch.full_like(sv.pos, 1.0 - self.p)
        ).bool()
        scale = math.log(1.0 - self.p)  # negative -> reduces delay
        cap = torch.full_like(sv.pos, DELAY_CAP)
        return SplitValue(
            torch.where(keep, sv.pos + scale, cap),
            torch.where(keep, sv.neg + scale, cap),
        )


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

class TemporalSequential(nn.Module):
    """Sequential that threads Tensor | SplitValue through a layer stack."""

    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pair(v):
    return (v, v) if isinstance(v, int) else v
