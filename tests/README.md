# nLSE/nLDE Test Notes

This directory contains reference and validation code for the delay-space approximations from the ASPLOS 2024 paper:

- *Energy Efficient Convolutions with Temporal Arithmetic*
- Key equations used here:
  - nLSE approximation: Eq. (6), min-of-max form
  - nLDE approximation: Eq. (7), min-of-inhibit form

Constants are loaded from `constants/orig_constants.pt` (`C_VALUES`, `D_VALUES`, `E_VALUES`, `F_VALUES`).

## What Was Wrong

### 1) nLDE inhibit pairing/order bug (critical)

The main issue was in how inhibit arguments were paired with constants in nLDE.

Incorrect pattern (old):

- `inhibit(x' + E_i, y' + F_i)`

Correct pattern (fixed):

- `inhibit(y' + E_i, x' + F_i)`

Why:

- `inhibit(t_i, t_d)` outputs `t_d` only when `t_d < t_i`, else `+inf`.
- For subtraction with `a >= b`, delay values satisfy `x' <= y'` where:
  - `x'` is delay of larger importance value (`a`)
  - `y'` is delay of smaller importance value (`b`)
- In this setup, `y' + E_i` is the inhibiting side and `x' + F_i` is the data side.

If this is flipped, many terms become `+inf`, collapsing approximation quality.

### 2) Double K-shift in vectorized nLDE path

Old test path applied K to both inputs and constants, then subtracted K:

- `nlde(x_p + K, y_p + K, E + K, F + K) - K`

This effectively double-shifts terms inside `x_p + E` and `y_p + F`.

Fix:

- call directly with raw delay-space inputs/constants:
  - `nlde(x_p, y_p, E, F)`

For inhibit-based nLDE, uniform shifts are not needed in this software approximation path.

### 3) nLSE imported from wrong location

`tests/nlse.py` previously imported `nlse` from `utils.helpers`, but that module does not define it.

Fix:

- replace with a local vectorized implementation in `tests/nlse.py`.

### 4) Biased test sampling in nLSE test

`tests/nlse.py` had `y = x`, forcing identical pairs and distorting the benchmark.

Fix:

- use independently sampled `x` and `y`.

## Current File Roles

- `tests/nonvec-approx.py`
  - Scalar-style reference functions:
    - `nLSE(a: int, b: int, terms=10) -> float`
    - `nLDE(a: int, b: int, terms=10) -> float`
  - Includes helper delay-space approximations and conversion routines.

- `tests/nlse.py`
  - Local vectorized nLSE implementation (importance-space input, approximation in delay space, converted back).
  - Sweeps term counts and reports error.

- `tests/nlde.py`
  - Local vectorized nLDE implementation with corrected inhibit ordering.
  - Uses direct delay-space invocation without double K-shift.
  - Sweeps term counts and reports error.

## Expected Behavior

After the fixes:

- nLSE error decreases as term count increases.
- nLDE error also decreases with term count (typically needing more terms than nLSE for similar quality).
- Scalar and vectorized implementations are consistent with the paper’s approximation structure.

