import torch

EPS = 1e-12

data = torch.load("constants/orig_constants.pt")
C_VALUES = data.get("C_VALUES")
D_VALUES = data.get("D_VALUES")
E_VALUES = data.get("E_VALUES")
F_VALUES = data.get("F_VALUES")


def inhibit(t_i: float, t_d: float) -> float:
    """Paper definition: output t_d iff t_d < t_i, else +inf."""
    return t_d if t_d < t_i else float("inf")


def nlse_delay_approx(x_p: float, y_p: float, C: torch.Tensor, D: torch.Tensor) -> float:
    """Equation (6) with Section 2.3 shift and operand ordering."""
    c = C.reshape(-1).tolist()
    d = D.reshape(-1).tolist()
    k = -min(min(c), min(d))
    x_k = x_p + k
    y_k = y_p + k

    if x_k < y_k:
        x_k, y_k = y_k, x_k

    terms = [x_k, y_k]
    for c_i, d_i in zip(c, d):
        terms.append(max(x_k + c_i, y_k + d_i))
    return min(terms) - k


def nlde_delay_approx(x_p: float, y_p: float, E: torch.Tensor, F: torch.Tensor) -> float:
    """Equation (7): min-of-inhibit approximation for nLDE.

    x_p must be the delay of the LARGER importance value (smaller delay),
    y_p must be the delay of the SMALLER importance value (larger delay),
    so x_p <= y_p.  E constants pair with the larger delay (inhibitor)
    and F constants pair with the smaller delay (data).
    """
    e = E.reshape(-1).tolist()
    f = F.reshape(-1).tolist()
    terms = []
    for e_i, f_i in zip(e, f):
        terms.append(inhibit(y_p + e_i, x_p + f_i))
    return min(terms)


def to_delay(v: float) -> float:
    return float(-torch.log(torch.tensor(max(v, EPS), dtype=torch.float64)).item())


def to_importance(v_p: float) -> float:
    return float(torch.exp(torch.tensor(-v_p, dtype=torch.float64)).item())


DEFAULT_TERMS = 10


def nLSE(a: int, b: int, terms: int = DEFAULT_TERMS) -> float:
    """Approximate addition in delay space: returns a + b (importance space).

    Takes two positive importance-space values (e.g. integers), converts to
    delay space, applies the min-of-max nLSE approximation (Eq. 6), and
    converts the result back to importance space.
    """
    if C_VALUES is None or D_VALUES is None:
        raise RuntimeError("Missing C_VALUES/D_VALUES in constants/orig_constants.pt")
    a_f = max(float(a), EPS)
    b_f = max(float(b), EPS)
    x_p = to_delay(a_f)
    y_p = to_delay(b_f)
    C = C_VALUES[terms]
    D = D_VALUES[terms]
    approx_delay = nlse_delay_approx(x_p, y_p, C, D)
    return to_importance(approx_delay)


def nLDE(a: int, b: int, terms: int = DEFAULT_TERMS) -> float:
    """Approximate subtraction in delay space: returns a - b (importance space).

    Takes two positive importance-space values with a >= b, converts to
    delay space, applies the min-of-inhibit nLDE approximation (Eq. 7), and
    converts the result back to importance space.
    """
    if E_VALUES is None or F_VALUES is None:
        raise RuntimeError("Missing E_VALUES/F_VALUES in constants/orig_constants.pt")
    a_f = max(float(a), EPS)
    b_f = max(float(b), EPS)
    if a_f < b_f:
        raise ValueError(f"nLDE requires a >= b, got a={a}, b={b}")
    x_p = to_delay(a_f)   # smaller delay (larger importance)
    y_p = to_delay(b_f)   # larger delay  (smaller importance)
    E = E_VALUES[terms]
    F = F_VALUES[terms]
    approx_delay = nlde_delay_approx(x_p, y_p, E, F)
    return to_importance(approx_delay)


def main() -> None:
    print("=== nLSE (addition) examples ===")
    for a, b in [(3, 5), (10, 7), (1, 1)]:
        approx = nLSE(a, b)
        print(f"  nLSE({a}, {b}) = {approx:.6f}  (exact {a + b})")

    print()
    print("=== nLDE (subtraction) examples ===")
    for a, b in [(8, 3), (10, 7), (5, 1)]:
        approx = nLDE(a, b)
        print(f"  nLDE({a}, {b}) = {approx:.6f}  (exact {a - b})")


if __name__ == "__main__":
    main()





