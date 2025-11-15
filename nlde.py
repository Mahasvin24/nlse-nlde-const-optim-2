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
    return torch.where(b < a, b, torch.inf)

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

def test_nlde(max_terms: int, device: torch.device, print_stats: bool = False):
    count = 4
    x = uniform_values(count).to(device)
    y = uniform_values(count).to(device)

    # Reordering so x > y
    x, y = torch.min(x, y), torch.max(x, y)

    # Subtraction in importance space
    exact = (y - x).reshape(-1)

    # Conversion to delay space (column vectors)
    x_p = - torch.log(x) 
    y_p = - torch.log(y)

    # Getting constants (row vectors)
    E = E_VALUES[max_terms].to(device)
    F = F_VALUES[max_terms].to(device)

    temporal_output = nlde(x_p, y_p, E, F)
    importance_output = torch.exp(- temporal_output)

    error = torch.mean(torch.abs(importance_output - exact) / exact * 100)

    # Printing (for testing)
    torch.set_printoptions(sci_mode=False, precision=2)
    if count < 10 and print_stats:
        print()
        print(f"Expected      : {[f"{v:.2f}" for v in exact.tolist()]}")
        print(f"Approximation : {[f'{v:.2f}' for v in importance_output.tolist()]}")
        print(f"Error         : {error.item():.2f}%\n")
    elif print_stats:
        print(f"Error ({max_terms}): {error.item():.2f}%")
    
    return error.item()

if __name__ == "__main__":
    # Device for potential GPU acceleration
    device_type = 'cpu'
    if torch.cuda.is_available():
        device_type = 'cuda'
    elif torch.backends.mps.is_available():
        device_type = 'mps'

    device = torch.device(device_type)

    print(f"Using device {device_type}.")

    test_nlde(max_terms=10, device=device, print_stats=True)
    
    """
    errors = []
    all_max_terms = [*range(0, 11), 15, 20]
    for max_terms in all_max_terms:
        errors.append(test_nlde(max_terms=max_terms, device=device, print_stats=True))

    
    plt.plot(all_max_terms, errors, marker='o', linestyle='-', color='red')
    
    plt.title("nLSE Error Using Given Constants")
    plt.xlabel("Number of Max Terms")
    plt.ylabel("Error (avg)")

    plt.grid(True)

    plt.show()
    """