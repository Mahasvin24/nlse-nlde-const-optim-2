import torch
import matplotlib.pyplot as plt
from utils import uniform_values, nlse

# Loading constants
data = torch.load("constants/learned_constants.pt")
C_VALUES = data["C_VALUES"]
D_VALUES = data["D_VALUES"]

def test_nlse(max_terms: int, device: torch.device, print_stats: bool = False):
    # Test set size
    count = 1_000_000
    
    # Data generation
    torch.random.manual_seed(0)
    x = uniform_values(count).to(device)
    y = uniform_values(count).to(device)
    y = x

    # Getting constants (row vectors)
    C = C_VALUES[max_terms].to(device)
    D = D_VALUES[max_terms].to(device)
    # C = torch.tensor([-0.9136939708203404,-1.2973538765893073,-1.819365705152768,-2.5803398662255184,-3.8923941692624195], device=device)
    # D = torch.tensor([-0.6231807228844553,-0.40209627157867595,-0.23638990973619267,-0.11750696238635272,-0.040148174670775916], device=device)

    # Addition in importance space
    exact = (x + y).reshape(-1)
    approx = nlse(x, y, C, D)

    # Error calculation
    rmse = torch.sqrt(torch.mean((exact - approx) ** 2))
    range = torch.max(exact) - torch.min(exact)
    error = (rmse / range) * 100

    # Printing
    torch.set_printoptions(sci_mode=False, precision=2)
    if count < 10 and print_stats:
        print()
        print(f"Expected      : {[f"{v:.2f}" for v in exact.tolist()]}")
        print(f"Approximation : {[f'{v:.2f}' for v in approx.tolist()]}")
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

    # Device override
    device_type = 'cpu'

    # Device creation
    device = torch.device(device_type)
    print(f"Using device {device_type}.")

    # Accuracy testing for different max_terms
    accuracy = []
    all_max_terms = [*range(0, 11), 15, 20]
    # all_max_terms = [5]
    for max_terms in all_max_terms:
        accuracy.append(100 - test_nlse(max_terms=max_terms, device=device, print_stats=True))

    # Graphing
    plt.plot(all_max_terms, accuracy, marker='o', linestyle='-', color='blue')
    plt.title("nLSE Accuracy Using Given Constants")
    plt.xlabel("Number of Max Terms")
    plt.ylabel("Accuracy (avg)")
    plt.ylim(90, 100) 
    plt.grid(True)
    plt.show()



    