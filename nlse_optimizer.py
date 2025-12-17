import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import nlse, nlse_noisy, uniform_values, rnrmse

# HYPERPARAMETERS
num_epochs = 1000
batch_size = 100_000
test_size = 1_000_000
learning_rate = 1e-3

# Loading constants
INPUT_FILE = "learned_constants.pt"
data = torch.load(f"constants/{INPUT_FILE}")
C_VALUES = data["C_VALUES"]
D_VALUES = data["D_VALUES"]

# Optimizer model
class nLSEModel(nn.Module):
    def __init__(self, max_terms, batch_size):
        super().__init__()

        # Constants
        self.max_terms = max_terms
        self.batch_size = batch_size

        # Model parameters
        self.C = nn.Parameter(C_VALUES[max_terms].clone().detach())
        self.D = nn.Parameter(D_VALUES[max_terms].clone().detach())

    def forward(self, x_p, y_p, noisy: bool = False) -> torch.tensor:
        if not noisy:
            return nlse(x_p, y_p, self.C, self.D)
        else:
            return nlse_noisy(x_p, y_p, self.C, self.D)
        
class nLSETrainer:
    def __init__(self, model, batch_size, test_size, num_epochs, learning_rate, noisy, device: torch.device):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = lambda pred, target: rnrmse(pred, target)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.test_size = test_size
        self.device = device
        self.noisy = noisy

    def train(self):
        for _ in range(self.num_epochs):
            # Clearing gradients
            self.optimizer.zero_grad() 

            # Generating uniform importance values
            x = uniform_values(self.batch_size).to(self.device)
            y = uniform_values(self.batch_size).to(self.device)

            # Converting to delay space values
            x_p = - torch.log(x)
            y_p = - torch.log(y)

            # Exact and approximation (delay space)
            exact = - torch.log(torch.exp(-x_p) + torch.exp(-y_p))
            approx = self.model(x_p, y_p, noisy=self.noisy)

            # Exact and approximation (importance space)
            exact = torch.exp(-exact).reshape(-1)
            approx = torch.exp(-approx)

            # Loss Calculation
            loss = self.loss_fn(approx, exact) 

            # Compute gradients
            loss.backward() 

            # Gradient descent (Adam optimized)
            self.optimizer.step() 

        return self.model.C.detach().cpu(), self.model.D.detach().cpu()
    
    def evaluate_error(self):
        # Generating uniform importance values
        x = uniform_values(self.test_size).to(self.device)
        y = uniform_values(self.test_size).to(self.device)

        # Converting to delay space values
        x_p = - torch.log(x)
        y_p = - torch.log(y)

        # Exact and approximation (delay space)
        exact_delay = - torch.log(torch.exp(-x_p) + torch.exp(-y_p))
        approx_delay = self.model(x_p, y_p, noisy=self.noisy)

        # Exact and approximation (importance space)
        exact = torch.exp(-exact_delay).reshape(-1)
        approx = torch.exp(-approx_delay)

        return self.loss_fn(approx, exact).item()

def test_model(max_terms, print_stats: bool = False):
    # Finding device type 
    device_type = 'cpu'
    if torch.cuda.is_available():
        device_type = 'cuda'
    elif torch.backends.mps.is_available():
        device_type = 'mps'
    
    # TEMPORARY OVERRIDE
    # device_type = 'cpu'

    # Creating device (for potential GPU acceleration)
    device = torch.device(device_type)

    if print_stats:
        print(f"Running with device {device_type}.")
        print()
        print(f"C={C_VALUES[max_terms]}")
        print(f"D={D_VALUES[max_terms]}")
        print()

    model = nLSEModel(max_terms, batch_size).to(device)
    trainer = nLSETrainer(
        model=model, 
        batch_size=batch_size, 
        test_size=test_size,
        num_epochs=num_epochs, 
        learning_rate=learning_rate, 
        noisy=False,
        device=device
    )

    # Evaluate error BEFORE training
    loss_before = trainer.evaluate_error() * 100
    if print_stats:
        print(f"Error before training: {loss_before:.2f}%")

    # Trained values
    C_new, D_new = trainer.train()

    # Replace model's C and D with the newly learned ones for evaluation
    model.C.data = C_new.to(device)
    model.D.data = D_new.to(device)

    # Evaluate error AFTER training
    loss_after = trainer.evaluate_error() * 100

    # Save new constants
    if loss_after < loss_before and INPUT_FILE != "orig_constants.pt":
        C_VALUES[max_terms] = C_new
        D_VALUES[max_terms] = D_new
        torch.save(
            {"C_VALUES": C_VALUES, "D_VALUES": D_VALUES},
            "constants/learned_constants.pt"
        )
    
    # Overall improvement
    print(f"Error for {max_terms:<2} maxterms: {loss_before:>5.5f}% -> {min(loss_after, loss_before):>4.5f}% (improvement {max(0,loss_before-loss_after):>5.5f}%)")

    # New constants
    if print_stats:
        print(f"\nLearned C={C_new}")
        print(f"Learned D={D_new}")

    # Update parameters in constants/updated_constants.pt
    # C_VALUES[max_terms] = C
    # D_VALUES[max_terms] = D

    return loss_after

if __name__ == '__main__':
    accuracy = []
    all_max_terms = [*range(1, 11), 15, 20]
    # all_max_terms = [20] # TEMPORARY OVERRIDE FOR TESTING
    for max_terms in all_max_terms:
        err = test_model(max_terms=max_terms, print_stats=False)
        accuracy.append(100 - err)

    plt.plot(all_max_terms, accuracy, marker='o', linestyle='-', color='blue')
    
    plt.title("nLSE Accuracy With Learned Constants")
    plt.xlabel("Number of Max Terms")
    plt.ylabel("Accuracy (avg)")
    plt.ylim(90, 100) # to match the style of the paper
    plt.grid(True)
    plt.show()
