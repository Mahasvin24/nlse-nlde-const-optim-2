import torch
import matplotlib.pyplot as plt

# Loading constants
data = torch.load("constants/orig_constants.pt")
E_VALUES = data["E_VALUES"]
F_VALUES = data["F_VALUES"]

# Uniform test value generation
def uniform_values(count: int) -> torch.Tensor:
    """
    Creates uniform values in range [0, 1)

    Returns a column vector tensor
    """
    return torch.rand(count).reshape(-1, 1)

# This might be flipped... or not...
def inhibit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """ Fully vectorized version that mimics the temporal inhibit function """
    inf_tensor = torch.full_like(a, torch.inf)
    return torch.where(b < a, b, inf_tensor) # element-wise

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

    X = x_p + E # shape=(N, max_terms) --> each row is an example
    Y = y_p + F # shape=(N, max_terms) --> each row is an example

    # inhibit(x + E, y + F) --> each row is an example
    inhibit_terms = inhibit(X, Y) # shape=(N, max_terms) --> element wise (X INHIBITS Y)
    
    nlde, _ = torch.min(inhibit_terms, dim=1) # shape=(N,)

    return nlde

def test_nlde(max_terms: int, device: torch.device, print_stats: bool = False):
    """
    Evaluate the nLDE operator for a given number of expansion terms.

    EPS is the threshold used to decide when exact values are treated as
    'effectively zero'. The ASPLOS paper uses EPS = 1e-6.
    When exact < EPS:
        - relative error is undefined (division by very small true values)
        - these samples are excluded from relative-error averaging
        - absolute error may be recorded separately if desired
    """
    # Can be adjusted 
    epsilon = 1e-6

    # Number of random input samples
    count = 10_000_000

    # Draw uniform samples in [0,1)
    x = uniform_values(count).to(device)
    y = uniform_values(count).to(device)

    # Clamp extremely small values to prevent -log(0) → ∞
    x = torch.clamp(x, min=epsilon)
    y = torch.clamp(y, min=epsilon)

    # Ensure x >= y so "subtraction" is non-negative
    x, y = torch.max(x, y), torch.min(x, y)

    # Exact subtraction in importance space
    exact = (x - y).reshape(-1)

    # Convert to delay domain for nLDE
    x_p = - torch.log(x)
    y_p = - torch.log(y)

    # Load E, F constants for this term count
    E = E_VALUES[max_terms].to(device).reshape(-1)
    F = F_VALUES[max_terms].to(device).reshape(-1)

    # Constant shift K ensures all constants are non-negative
    # minimum_constant = min(E.min().item(), F.min().item())
    # K = -minimum_constant if minimum_constant < 0 else 0.0
    K = E_VALUES[max_terms][-1]

    # nLDE approximation: nLDE + K = nLDE(x + K, y + K, inhibit(E_0 + K, F_0 + K)...)
    temporal_output = nlde(x_p + K, y_p + K, E + K, F + K) - K
    importance_output = torch.exp(-temporal_output)

    # RNRMSE
    rmse = torch.sqrt(torch.mean((exact - importance_output) ** 2))
    range = torch.max(exact) - torch.min(exact)
    error = (rmse / range) * 100

    # Print
    if print_stats:
        print(f"Error for {max_terms} max terms: {error:.2f}%")

    return error


if __name__ == "__main__":
    # Device for potential GPU acceleration
    device_type = 'cpu'
    if torch.cuda.is_available():
        device_type = 'cuda'
    elif torch.backends.mps.is_available():
        device_type = 'mps'

    device_type = 'cpu'

    device = torch.device(device_type)

    print(f"Using device {device_type}.")

    accuracy = []
    all_max_terms = [*range(1, 11), 15, 20]
    # all_max_terms = [5]

    for max_terms in all_max_terms:
        accuracy.append(100 - test_nlde(max_terms=max_terms, device=device, print_stats=True))

    plt.plot(all_max_terms, accuracy, marker='o', linestyle='-', color='orange')
    
    plt.title("nLDE Accuracy Using Given Constants")
    plt.xlabel("Number of Max Terms")
    plt.ylabel("Accuracy (avg)")

    plt.ylim(0, 100)

    plt.grid(True)

    plt.show()
    