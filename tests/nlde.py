import torch
import matplotlib.pyplot as plt
import os

# Loading constants
data = torch.load("constants/orig_constants.pt")
if "E_VALUES" in data and "F_VALUES" in data:
    E_VALUES = data["E_VALUES"]
    F_VALUES = data["F_VALUES"]
else:
    # Fallback for quick pipeline testing: reuse nLSE constants as nLDE constants.
    # If you later add true E_VALUES/F_VALUES, this will automatically switch back.
    E_VALUES = data["C_VALUES"]
    F_VALUES = data["D_VALUES"]

# Uniform test value generation
def uniform_values(count: int) -> torch.Tensor:
    """
    Creates uniform values in range [0, 1)

    Returns a column vector tensor
    """
    return torch.rand(count).reshape(-1, 1)

def inhibit(t_i: torch.Tensor, t_d: torch.Tensor) -> torch.Tensor:
    """Vectorized temporal inhibit: output t_d where t_d < t_i, else +inf."""
    return torch.where(t_d < t_i, t_d, torch.full_like(t_d, torch.inf))


def nlde(x_p: torch.Tensor, y_p: torch.Tensor, E: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Vectorized nLDE approximation (Eq. 7).

    Args:
        x_p: delay of the LARGER importance value (smaller delay), shape (N, 1)
        y_p: delay of the SMALLER importance value (larger delay),  shape (N, 1)
        E: inhibit-term constants, shape (1, max_terms) or (max_terms,)
        F: inhibit-term constants, shape (1, max_terms) or (max_terms,)

    E pairs with the larger delay (y_p, inhibitor side),
    F pairs with the smaller delay (x_p, data side).
    """
    if x_p.shape != y_p.shape:
        raise ValueError("Arguments x_p and y_p must have the same shape.")
    if E.shape != F.shape:
        raise ValueError("Arguments E and F must have the same shape.")

    inhibitor = y_p + E   # (N, max_terms) -- larger delay + E
    data_event = x_p + F  # (N, max_terms) -- smaller delay + F

    inhibit_terms = inhibit(inhibitor, data_event)  # (N, max_terms)

    result, _ = torch.min(inhibit_terms, dim=1)  # (N,)
    return result

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
    count = int(os.getenv("NLDE_COUNT", "10000000"))

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
    E = E_VALUES[max_terms].to(device)
    F = F_VALUES[max_terms].to(device)

    # nLDE approximation (K-shift is a no-op for inhibit-based nLDE)
    temporal_output = nlde(x_p, y_p, E, F)
    importance_output = torch.exp(-temporal_output)

    # RNRMSE
    rmse = torch.sqrt(torch.mean((exact - importance_output) ** 2))
    val_range = torch.max(exact) - torch.min(exact)
    error = (rmse / val_range) * 100

    # Print
    if print_stats:
        print(f"Error for {max_terms} max terms: {error:.2f}%")

    return error.item()


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
    