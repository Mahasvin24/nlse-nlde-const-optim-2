import torch
import matplotlib.pyplot as plt

# Loading constants
data = torch.load("constants/orig_constants.pt")
C_VALUES = data["C_VALUES"]
D_VALUES = data["D_VALUES"]

# Uniform test value generation
def uniform_values(count: int) -> torch.Tensor:
    """
    Creates uniform values in range [0, 1)

    Returns a column vector tensor
    """
    return torch.rand(count).reshape(-1, 1)

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

def test_nlse(max_terms: int, device: torch.device, print_stats: bool = False):
    count = 10000000
    x = uniform_values(count).to(device)
    y = uniform_values(count).to(device)

    # x = np.reshape(gaussian_noise(3), shape=(-1, 1))
    # y = np.reshape(gaussian_noise(3), shape=(-1, 1))

    # Addition in importance space
    exact = (x + y).reshape(-1)

    # Conversion to delay space (column vectors)
    x_p = - torch.log(x) 
    y_p = - torch.log(y)

    # Getting constants (row vectors)
    C = C_VALUES[max_terms].to(device)
    D = D_VALUES[max_terms].to(device)

    temporal_output = nlse(x_p, y_p, C, D)
    importance_output = torch.exp(- temporal_output)

    error = torch.mean(torch.abs(importance_output - exact) / exact * 100)

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
    all_max_terms = [*range(0, 11), 15, 20]
    for max_terms in all_max_terms:
        accuracy.append(100 - test_nlse(max_terms=max_terms, device=device, print_stats=True))

    plt.plot(all_max_terms, accuracy, marker='o', linestyle='-', color='red')
    
    plt.title("nLSE Accuracy Using Given Constants")
    plt.xlabel("Number of Max Terms")
    plt.ylabel("Accuracy (avg)")

    plt.grid(True)

    plt.show()

# Why is my accuracy so much lower...


    