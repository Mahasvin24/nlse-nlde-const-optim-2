"""
Delay-space neural network layers for temporal arithmetic CNNs.

Implements the temporal arithmetic framework from:
  "Energy Efficient Convolutions with Temporal Arithmetic"
  (Gretsch et al., ASPLOS '24)

All computation happens in delay space where x' = -ln(x) (Eq. 1-2).

Core delay-space operation mapping:
  Multiplication:  x * y   ->  x' + y'               (delay addition)
  Addition:        x + y   ->  nLSE(x', y')           (Eq. 4)
  Subtraction:     x - y   ->  nLDE(x', y')           (Eq. 5)

Negative numbers use split-value representation <x_pos, x_neg>
(paper Sections 2.2, 4.4).

Training uses the exact nLSE formula via logsumexp for smooth
gradients; the min-of-max approximation (Eq. 6) with C/D constants
can be swapped in for hardware-accurate inference.
"""

import math
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPS = 1e-7
DELAY_CAP = 50.0


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


def nary_nlse(delays: torch.Tensor, dim: int) -> torch.Tensor:
    """Exact n-ary nLSE (delay-space addition of *n* values).

    nLSE(x1', …, xn')  =  -ln(Σ e^{-xi'})
                         =  -logsumexp(-x', dim)

    Large delays are clamped to DELAY_CAP so that all-zero-importance
    edge cases don't produce NaN.
    """
    return -torch.logsumexp(-torch.clamp(delays, max=DELAY_CAP), dim=dim)


def nlse_2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact pairwise nLSE: nLSE(a', b') = -logaddexp(-a', -b')."""
    return -torch.logaddexp(
        -torch.clamp(a, max=DELAY_CAP),
        -torch.clamp(b, max=DELAY_CAP),
    )


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
