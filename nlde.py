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
    return torch.where(b < a, b, torch.inf) # element-wise

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

    # X and Y must contain positive values
    X = torch.relu(X)
    Y = torch.relu(Y)

    # inhibit(x + E, y + F) --> each row is an example
    inhibit_terms = inhibit(X, Y) # shape=(N, max_terms) --> element wise (X INHIBITS Y)
    
    nlde, _ = torch.min(inhibit_terms, dim=1) # shape=(N,)

    return nlde

def test_nlde(max_terms: int, device: torch.device, print_stats: bool = False):
    count = 10000000
    x = uniform_values(count).to(device)
    y = uniform_values(count).to(device)

    # print(f"\nx before: {x[0:5].reshape(-1)}")
    # print(f"y before: {y[0:5].reshape(-1)}")
    # print(f"All pos? {torch.all(x >= 0) and torch.all(y >= 0)}\n")

    # Reordering so x > y
    x, y = torch.max(x, y), torch.min(x, y)

    # print(f"x after: {x[0:5].reshape(-1)}")
    # print(f"y after: {y[0:5].reshape(-1)}")
    # print(f"All pos? {torch.all(x >= 0) and torch.all(y >= 0)}\n")

    # Subtraction in importance space
    exact = (x - y).reshape(-1)

    # print(f"x - y: {exact[0:5].reshape(-1)}")
    # print(f"All pos? {torch.all(exact >= 0)}\n")

    # Conversion to delay space (column vectors)
    # NOTE: x_p < y_p 
    x_p = - torch.log(x)
    y_p = - torch.log(y)

    # print(f"x_p: {x_p[0:5].reshape(-1)}")
    # print(f"y_p: {y_p[0:5].reshape(-1)}")
    # print(f"All pos? {torch.all(x_p >= 0) and torch.all(y_p >= 0)}\n")

    # Getting constants (row vectors)
    E = E_VALUES[max_terms].to(device)
    F = F_VALUES[max_terms].to(device)

    # print(f"x_p + E: {x_p[0] + E}")
    # print(f"y_p + F: {y_p[0] + F}")
    # print(f"All pos? {torch.all(x_p + E > 0) and torch.all(y_p + F >= 0)} <-- usually False\n") 

    temporal_output = nlde(x_p, y_p, E, F)
    
    importance_output = torch.exp(- temporal_output) # temporal -> importance space

    error = torch.mean(torch.abs(importance_output - exact) / (exact + 1e-12) * 100)

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
    
    """
    errors = []
    all_max_terms = [*range(1, 11), 15, 20]
    for max_terms in all_max_terms:
        errors.append(test_nlde(max_terms=max_terms, device=device, print_stats=True))

    plt.plot(all_max_terms, errors, marker='o', linestyle='-', color='red')
    
    plt.title("nLSE Error Using Given Constants")
    plt.xlabel("Number of Max Terms")
    plt.ylabel("Error (avg)")

    plt.grid(True)

    plt.show()
    """

# Take a look at the <x_pos, x_neg> stuff
# I don't understand exact how that workings
# Implemeting that stuff properly might fix things
# Ask for help
# It's confusing tho b/c all the values should be pos

"""
<x_pos, x_neg> Formula

for a value x (either positive for negative)

x_pos = x if x > 0 else 0
x_neg = -x if x < 0 else 0

Note: both x_pos and x_neg are non-negative values

Paper link: https://sites.cs.ucsb.edu/~sherwood/pubs/ASPLOS-24-temparith.pdf
"""