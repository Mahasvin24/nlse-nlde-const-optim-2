# models/ — Agent Knowledge Base

## Files

| File | Purpose |
|------|---------|
| `temporal_layers.py` | All delay-space neural network layers (Conv2d, Linear, ReLU, MaxPool, etc.) |
| `temporal_mnist.ipynb` | Training notebook: MNIST digit classifier using the temporal layers |
| `digit_classifier.ipynb` | Reference: plain NumPy MLP for MNIST (no temporal arithmetic) |

## Design Decisions

### Weight storage and gradient flow

Weights live in **importance space** as standard `nn.Parameter` tensors. During forward, each weight tensor is split into positive/negative components via `clamp`, then converted to delay space with `to_delay`. This lets standard initialisation (He/Kaiming) and optimisers (Adam) work unmodified. The log-domain gradient (`-1/x`) naturally amplifies small-weight gradients, which compensates for the nLSE soft-min reducing their contribution — tested and confirmed that even zero-initialised bias gets non-trivial gradient.

Reference: paper Section 2.2 (split-value representation), Section 4.4 (dedicated kernel handling).

### Exact nLSE for training, approximation for inference

The n-ary nLSE is computed exactly as `-logsumexp(-delays, dim)` during training. This gives smooth gradients through the full computation graph. The min-of-max approximation (paper Eq. 6) with C/D constants from `constants/` can be substituted for hardware-accurate inference evaluation. The layers in `temporal_layers.py` use the exact path only; approximate mode is future work.

Reference: paper Eq. 4-6, Section 2.1.

### DELAY_CAP sentinel (50.0)

Rather than using `float('inf')` for "zero importance", all layers use `DELAY_CAP = 50.0`. This avoids NaN from `logsumexp` on all-inf inputs while being functionally equivalent (`exp(-50) ≈ 2e-22`). The cap is applied inside `nary_nlse` and `nlse_2`, and used as the fill value in `DelayEncoder`, `TemporalReLU`, padding, and dropout.

### Padding in TemporalConv2d

Standard `F.unfold` zero-pads, but zero in delay space means importance=1 (wrong). The conv layer uses `F.pad(tensor, ..., value=DELAY_CAP)` to pad with zero-importance before calling `F.unfold` with no additional padding.

### TemporalMaxPool2d

Max importance = min delay. Implemented as `min(x) = -max(-x)` using `F.max_pool2d`. Delays are clamped to DELAY_CAP before negation to avoid inf/NaN in the pooling kernel.

### Split-value convolution (general case)

The four product terms handle arbitrary signed inputs:
- `out_pos = nLSE(in_pos + w_pos, in_neg + w_neg)` — positive products
- `out_neg = nLSE(in_pos + w_neg, in_neg + w_pos)` — negative products

After `TemporalReLU`, `in_neg` is DELAY_CAP, so the `in_neg + w_*` terms contribute `exp(-DELAY_CAP) ≈ 0` and drop out of the nLSE naturally. No special-casing needed.

Reference: paper Section 4.4.

## Lessons Learned / Pitfalls

- **Gradient clipping is essential.** The log-domain gradient (`-1/x`) amplifies gradients for near-zero weights. Without `clip_grad_norm_(params, 5.0)`, training diverges. A learning rate of `3e-4` with Adam works well.

- **Zero bias is fine.** Despite the bias delay being huge at init (`-ln(ε) ≈ 16`), the log-domain gradient compensates: the total gradient w.r.t. bias ≈ constant regardless of bias magnitude (verified numerically).

- **`F.pad` with `value=DELAY_CAP` works for autograd.** Padded positions are constants with zero gradient; non-padded gradients flow through `F.unfold` → products → nLSE normally.

- **Training is ~50-80× slower per epoch than standard CNN** due to explicit product tensor construction (`(B, C_out, 2*K, L)`) and logsumexp reduction. Acceptable for MNIST; larger datasets would need custom CUDA kernels or the importance-space shortcut (mathematically equivalent for exact nLSE).

- **The model reaches ~90% val accuracy in 3 epochs** on MNIST with the architecture `Conv(1→8) → Conv(8→16) → FC(784→64) → FC(64→10)`. Standard CNN equivalent reaches ~98% in 10 epochs.

## Paper Section Quick Reference

| Concept | Section | Equations |
|---------|---------|-----------|
| Delay encoding | 2 | Eq. 1-2 |
| nLSE (delay-space addition) | 2.1 | Eq. 4, 6 |
| nLDE (delay-space subtraction) | 2.2 | Eq. 5, 7 |
| Split-value representation | 2.2 | — |
| Approximation constants C, D, E, F | 2.1-2.2 | Eq. 6-7 |
| Recurrence architecture | 3 | — |
| Convolution engine | 4.3 | — |
| Split-value convolution | 4.4 | — |
