import torch

from utils.helpers import guassian_noise, inhibit


def nlse(x_p: torch.Tensor, y_p: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Min-of-max nLSE approximation (paper Eq. 6).

    All tensors are in delay space. Returns delay-space output.

    The uniform shift ``K = -min(C ∪ D)`` applied before the min-of-max tree
    and removed after (``out - K``) matches the hardware datapath: physical
    delays are non-negative, while the optimised ``C_i``, ``D_i`` are
    negative—lifting both operands by ``K`` keeps intermediate delay-line
    values representable on chip. Algebraically the shift cancels in
    floating point; it is kept for silicon fidelity.

    Args:
        x_p: column vector (N, 1)
        y_p: column vector (N, 1)
        C, D: row vectors (1, max_terms) of optimized constants
    """
    if x_p.shape != y_p.shape:
        raise ValueError("Arguments x_p and y_p must have the same shape.")
    if C.shape != D.shape:
        raise ValueError("Arguments C and D must have the same shape.")

    K = -torch.min(torch.cat((C, D))).to(x_p.device)
    x_p = x_p + K
    y_p = y_p + K

    # Order so first operand has larger delay (paper Section 2.1: keep first >= second)
    x_p, y_p = (
        torch.maximum(x_p, y_p).reshape(-1, 1),
        torch.minimum(x_p, y_p).reshape(-1, 1),
    )

    X = x_p + C
    Y = y_p + D
    maximum_terms = torch.maximum(X, Y)
    all_terms = torch.cat((x_p, y_p, maximum_terms), dim=1)

    out, _ = torch.min(all_terms, dim=1)
    return out - K


def nlde(x_p: torch.Tensor, y_p: torch.Tensor, E: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Min-of-inhibit nLDE approximation (paper Eq. 7).

    All tensors are in delay space. Returns delay-space output.

    Args:
        x_p: delay of larger importance value (smaller delay), shape (N, 1)
        y_p: delay of smaller importance value (larger delay), shape (N, 1)
        E, F: row vectors (1, max_terms); E pairs with inhibitor (y_p), F with data (x_p)

    inhibit(t_i, t_d) passes t_d when t_d < t_i else +inf.
    """
    if x_p.shape != y_p.shape:
        raise ValueError("Arguments x_p and y_p must have the same shape.")
    if E.shape != F.shape:
        raise ValueError("Arguments E and F must have the same shape.")

    inhibitor = y_p + E
    data_event = x_p + F
    inhibit_terms = inhibit(inhibitor, data_event)

    out, _ = torch.min(inhibit_terms, dim=1)
    return out


def nlse_noisy(x: torch.Tensor, y: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """nLSE with pre/post VTC noise on importance and delay values.

    Args:
        x, y: importance-space column vectors (N, 1)
        C, D: row vectors (1, max_terms)

    Returns delay-space output (same convention as ``nlse``).
    """
    if x.shape != y.shape:
        raise ValueError("Arguments x and y must have the same shape.")
    if C.shape != D.shape:
        raise ValueError("Arguments C and D must have the same shape.")

    epsilon = 1e-9
    max_importance = 1.0
    max_delay = -torch.log(torch.tensor(epsilon, device=x.device))
    device = x.device

    x_noisy = torch.clamp(x + guassian_noise(x.shape[0], device=device), min=epsilon, max=max_importance)
    y_noisy = torch.clamp(y + guassian_noise(y.shape[0], device=device), min=epsilon, max=max_importance)

    x_p = -torch.log(x_noisy)
    y_p = -torch.log(y_noisy)

    x_p = torch.clamp(x_p + guassian_noise(x_p.shape[0], device=device), min=epsilon, max=max_delay)
    y_p = torch.clamp(y_p + guassian_noise(y_p.shape[0], device=device), min=epsilon, max=max_delay)

    return nlse(x_p, y_p, C, D)
