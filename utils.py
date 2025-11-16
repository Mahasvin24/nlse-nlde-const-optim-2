import torch

# Uniform test value generation
def uniform_values(count: int) -> torch.Tensor:
    """
    Creates uniform values in range [0, 1)

    Returns a column vector tensor
    """
    return torch.rand(count).reshape(-1, 1)

# Matches definition in paper (a INHIBITS b)
def inhibit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """ Fully vectorized version that mimics the temporal inhibit function """
    return torch.where(b < a, b, torch.inf)

# nLSE
def nlse(x_p: torch.Tensor, y_p: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Performs the nLSE described in the paper
    Beyond 7-10, max_terms, there isn't much of a benefit (according to ASPLOS paper)
    max_terms are decided by the shape of C and D

    It is assumed that arguments are passed with the proper shape.

    Args:
        x_p: column vector of values shape=(N, 1)
        y_p: column vector of values shape=(N, 1)
        C: row vector of values from constants.py of shape=(1, max_terms)
        D: row vector of values from constants.py of shape=(1, max_terms)
        test: prints the matrix before minum when set to true
    """
    # Mismatched lengths
    if x_p.shape != y_p.shape:
        raise ValueError("Arguments x_p and y_p must have the same shape.")
    if C.shape != D.shape:
        raise ValueError("Arguments C and D must have the same shape.")

    # Injecting noise
    # max_terms = C.shape[1]
    # std_dev = 0.05 * max_terms
    # C_noise = torch.normal(mean=0, std=std_dev, size=C.shape) # gaussian
    # D_noise = torch.normal(mean=0, std=std_dev, size=D.shape)
    # C = C + C_noise
    # D = D + D_noise

    X = x_p + C # shape=(N, max_terms) --> each row is an example
    Y = y_p + D # shape=(N, max_terms) --> each row is an example

    # max(x + C, y + D) --> each row is an example
    maximum_terms = torch.maximum(X, Y) # shape=(N, max_terms) --> element wise maximum

    all_terms = torch.cat((x_p, y_p, maximum_terms), dim=1) # shape=(N, max_terms + 2)
    
    nlse, _ = torch.min(all_terms, dim=1) # shape=(N,)

    return nlse

# nLDE
def nlde(x_p: torch.Tensor, y_p: torch.Tensor, E: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """
    Performs the nLDE described in the paper
    20 or 25 max_terms is best (according to paper)
    max_terms are decided by the shape of E and F

    It is assumed that arguments are passed with the proper shape.

    Args:
        x_p: column vector of values shape=(N, 1)
        y_p: column vector of values shape=(N, 1)
        E: row vector of values from constants.py of shape=(1, max_terms)
        F: row vector of values from constants.py of shape=(1, max_terms)
        test: prints the matrix before minum when set to true
    """
    # Mismatched lengths
    if x_p.shape != y_p.shape:
        raise ValueError("Arguments x_p and y_p must have the same shape.")
    if E.shape != F.shape:
        raise ValueError("Arguments E and F must have the same shape.")
    
    # Injecting noise
    # max_terms = E.shape[1]
    # std_dev = 0.05 * max_terms
    # E_noise = torch.normal(mean=0, std=std_dev, size=E.shape) # gaussian
    # F_noise = torch.normal(mean=0, std=std_dev, size=F.shape)
    # E = E + E_noise
    # F = F + F_noise

    X = x_p + E # shape=(N, max_terms) --> each row is an example
    Y = y_p + F # shape=(N, max_terms) --> each row is an example

    # max(x + E, y + F) --> each row is an example
    inhibit_terms = inhibit(X, Y) # shape=(N, max_terms) --> element wise maximum
    
    nlde, _ = torch.min(inhibit_terms, dim=1) # shape=(N,)

    return nlde