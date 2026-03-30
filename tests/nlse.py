import torch
import matplotlib.pyplot as plt

# Loading constants
data = torch.load("constants/orig_constants.pt")
C_VALUES = data["C_VALUES"]
D_VALUES = data["D_VALUES"]

EPS = 1e-12


def uniform_values(count: int) -> torch.Tensor:
    """Creates uniform values in range (0, 1) as a column vector."""
    return torch.rand(count).reshape(-1, 1).clamp(min=EPS)


def nlse(x: torch.Tensor, y: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Vectorized nLSE approximation (Eq. 6).

    Takes importance-space column vectors x, y of shape (N, 1) and
    row-vector constants C, D of shape (1, max_terms).  Returns the
    approximate sum (importance space) as a 1-D tensor of shape (N,).
    """
    x_p = -torch.log(x)
    y_p = -torch.log(y)

    K = -torch.min(torch.cat((C, D))).to(x_p.device)
    x_p = x_p + K
    y_p = y_p + K

    # Ensure first operand is always the larger delay (paper Section 2.1)
    x_p, y_p = (
        torch.maximum(x_p, y_p).reshape(-1, 1),
        torch.minimum(x_p, y_p).reshape(-1, 1),
    )

    X = x_p + C  # (N, max_terms)
    Y = y_p + D  # (N, max_terms)

    max_terms = torch.maximum(X, Y)
    all_terms = torch.cat((x_p, y_p, max_terms), dim=1)  # (N, max_terms+2)

    result_delay, _ = torch.min(all_terms, dim=1)  # (N,)
    result_delay = result_delay - K

    return torch.exp(-result_delay)


def test_nlse(max_terms: int, device: torch.device, print_stats: bool = False):
    count = 1_000_000

    torch.random.manual_seed(0)
    x = uniform_values(count).to(device)
    y = uniform_values(count).to(device)

    C = C_VALUES[max_terms].to(device)
    D = D_VALUES[max_terms].to(device)

    exact = (x + y).reshape(-1)
    approx = nlse(x, y, C, D)

    rmse = torch.sqrt(torch.mean((exact - approx) ** 2))
    val_range = torch.max(exact) - torch.min(exact)
    error = (rmse / val_range) * 100

    torch.set_printoptions(sci_mode=False, precision=2)
    if count < 10 and print_stats:
        print()
        print(f"Expected      : {[f'{v:.2f}' for v in exact.tolist()]}")
        print(f"Approximation : {[f'{v:.2f}' for v in approx.tolist()]}")
        print(f"Error         : {error.item():.2f}%\n")
    elif print_stats:
        print(f"Error ({max_terms}): {error.item():.2f}%")

    return error.item()

if __name__ == "__main__":
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
    for max_terms in all_max_terms:
        accuracy.append(100 - test_nlse(max_terms=max_terms, device=device, print_stats=True))

    plt.plot(all_max_terms, accuracy, marker='o', linestyle='-', color='blue')
    plt.title("nLSE Accuracy Using Given Constants")
    plt.xlabel("Number of Max Terms")
    plt.ylabel("Accuracy (avg)")
    plt.ylim(90, 100)
    plt.grid(True)
    plt.show()
