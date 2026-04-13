# Agent rules for this repository

## Purpose

This repository implements and experiments with ideas from a foundational research paper. The goal is **not** to write a paper — it is to build, test, and iterate on code that realizes concepts from the paper. Treat every task **methodically**: reason through steps before acting, prefer well-justified choices over quick hacks, and **document** what you did and why so work can be reviewed, reproduced, and built on.

## Foundational paper

The paper **"Energy Efficient Convolutions with Temporal Arithmetic"** (Gretsch et al., ASPLOS '24) lives at `paper/research-paper.pdf`. It is the primary reference for all work in this repo.

**Before making non-trivial implementation decisions**, consult the paper to ensure alignment with its definitions, equations, and architectural choices. Key concepts to stay grounded in:

- **Delay space encoding**: values encoded as `x' = -ln(x)` (Eq. 1–2), importance-space ↔ delay-space conversions.
- **nLSE / nLDE**: the negative-log-sum-exponential and negative-log-difference-exponential functions for delay-space addition and subtraction (Eq. 4–5), and their min-of-max / min-of-inhibit approximations (Eq. 6–7).
- **Approximation constant optimization**: fitting the `C_i`, `D_i` (nLSE) and `E_i`, `F_i` (nLDE) constants via Pyomo + KNITRO (Sections 2.1–2.2).
- **Recurrence architecture & rolling-shutter convolution**: reference-frame shifting, compute-on-arrival, and the dedicated convolution engine (Sections 3–4).
- **Split-value representation**: handling negative numbers with `<x_pos, x_neg>` pairs and nLDE re-normalization (Sections 2.2, 4.4).
- **Noise and accuracy trade-offs**: PSIJ, RJ, unit scale, and the interplay between number of approximation terms and hardware noise (Section 5).

When something in the code looks wrong or a design choice is ambiguous, **re-read the relevant section of the paper** before changing behavior.

## Documentation while working in a folder

Whenever you work in a folder (or add meaningful behavior under a path), keep a living record for future agents and collaborators.

### Where to write it

Write this material in **`AGENTS.md`**:

- Place it in the **folder where the work applies** (e.g. `data/AGENTS.md`, `experiments/AGENTS.md`). If the scope is repo-wide, use or update the root **`AGENTS.md`**.
- Do **not** rely on chat memory alone; persist what matters in **`AGENTS.md`**.

### What to record

1. **Important facts** — assumptions, data layout, naming conventions, environment quirks, invariants, and anything non-obvious someone would need to know to work safely in that area.
2. **Bugs and issues** — what went wrong, symptoms, root cause (once known), and the fix or workaround so the same mistake is less likely to recur.
3. **Wisdom** — lessons learned, dead ends to avoid, references to specific paper sections or equations, and short "if you touch X, remember Y" notes.

Keep entries **dated or ordered** when useful (e.g. short dated bullets or a reverse-chronological log) so history stays readable.

### README vs AGENTS.md

Use **`README.md`** in a folder for **human-oriented** overview (how to run something, what the folder is for). Use **`AGENTS.md`** for the **agent-focused** facts, issue log, and retained wisdom described above. If both exist, avoid duplicating large blocks; cross-link when helpful.

---

Agents should **read** the relevant **`AGENTS.md`** before substantial edits in a subtree and **update** it when the session surfaces new facts, failures, or durable lessons. When in doubt about any mathematical or architectural detail, refer back to `paper/research-paper.pdf`.
