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

### nLSE uses the min-of-max approximation (Eq. 6) everywhere

`temporal_layers.py` does **not** use `logsumexp`. Both the pairwise `nlse_2` and the n-ary `nary_nlse` implement the hardware-faithful min-of-max approximation from paper Eq. 6, matching `utils/temporal_artithmetic.py::nlse` bit-for-bit on pairwise inputs (verified numerically). The `(C, D)` constants are loaded once at import time from `constants/orig_constants.pt` (using `NLSE_MAX_TERMS = 7`) and cached per (device, dtype). Call `set_nlse_constants(C, D)` to swap in `learned_constants.pt` without touching the module.

The hardware operator is strictly 2-ary, so n-ary kernel sums are implemented as a balanced binary tree of `nlse_2` calls (`ceil(log2 K)` rounds). Odd-sized levels are padded with `DELAY_CAP`, which is the identity element for nLSE.

Reference: paper Eq. 4-6, Section 2.1; `utils/temporal_artithmetic.py`.

### Gradient flow through the approximation

The Eq. 6 form is piecewise: `out = min(a, b, min_i max(a + C_i, b + D_i))`. Gradient flows only through the argmin / argmax paths (sparse, like ReLU / max-pool), not through all operands like `logsumexp` would. Training still works because:
- The tree reduction distributes gradient across the K kernel operands over log2(K) rounds.
- Split-value weights receive gradient via both `a` and `b` sides of the pairwise min.
- The log-domain gradient amplification of small weights (see below) compensates for the sparser routing.

Empirically, the same architecture / LR that worked with the exact path continues to converge with the approximation; gradient clipping at 5.0 remains essential.

### DELAY_CAP sentinel (50.0)

Rather than using `float('inf')` for "zero importance", all layers use `DELAY_CAP = 50.0` (`exp(-50) ≈ 2e-22`). The cap is applied inside `nary_nlse` and `nlse_2`, and used as the fill value in `DelayEncoder`, `TemporalReLU`, conv padding, dropout, and the odd-level padding inside the `nary_nlse` tree reduction.

Pairwise `nlse_2` applies the same `K = -min(C ∪ D)` shift as `utils/temporal_artithmetic.nlse`: on hardware, delays are non-negative physical times while `C_i`, `D_i` are negative offsets, so operands are lifted by `K` before the min-of-max network and `K` is subtracted at the output so the logical delay is unchanged. In float this pair is redundant but kept for parity with silicon.

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

- **Training is ~50-80× slower per epoch than standard CNN** due to explicit product tensor construction (`(B, C_out, 2*K, L)`) and the tree-reduced min-of-max reduction. Peak activation memory is ~`M × batch × Co × K × L` floats during the first round of `nary_nlse` (M = `NLSE_MAX_TERMS`); shrink `NLSE_MAX_TERMS` or batch size if you run out of memory on larger kernels. Acceptable for MNIST at the default config.

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
