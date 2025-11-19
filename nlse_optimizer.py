import torch
import torch.nn as nn
from utils import nlse, uniform_values

# HYPERPARAMETERS
num_epochs = 1_000
batch_size = 100_000
learning_rate = 3e-2

# Loading constants
data = torch.load("constants/learned_constants.pt")
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


    def forward(self, x, y):
        # Calculating x_p and y_p
        x_p = - torch.log(x)
        y_p = - torch.log(y)

        # Forward nLSE pass
        delay_approx = nlse(x_p, y_p, self.C, self.D)

        # Converting to importance space
        approx = torch.exp(- delay_approx)

        return approx
    
    def error(self, x, y):
        # Inference
        exact = (x + y).reshape(-1)
        approx = self.forward(x, y)

        # Convert to importance space
        exact = torch.exp(-exact)
        approx = torch.exp(-approx)

        # Calculate 
        error = torch.mean(torch.abs(approx - exact) / exact * 100)

        return error

class nLSETrainer:
    def __init__(self, model, batch_size, num_epochs, learning_rate, device: torch.device):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device

    def train(self):
        for _ in range(self.num_epochs):
            self.optimizer.zero_grad() # clearing gradients

            # Generating data
            x = uniform_values(self.batch_size).to(self.device)
            y = uniform_values(self.batch_size).to(self.device)

            # Exact output
            exact = (x + y).reshape(-1)

            # Forward pass through model
            approx = self.model(x, y) 

            # Loss Calculation
            loss = self.loss_fn(approx, exact) 

            # Compute gradients
            loss.backward() 

            # Gradient descent (Adam optimized)
            self.optimizer.step() 

        return self.model.C.detach().cpu(), self.model.D.detach().cpu()
    
    def evaluate_error(self):
        # Creating data
        x = uniform_values(self.batch_size * 10).to(self.device)
        y = uniform_values(self.batch_size * 10).to(self.device)

        return self.model.error(x, y)

def test_model(max_terms):
    # Finding device type 
    device_type = 'cpu'
    if torch.cuda.is_available():
        device_type = 'cuda'
    elif torch.backends.mps.is_available():
        device_type = 'mps'
    
    # Creating device (for potential GPU acceleration)
    device = torch.device(device_type)
    print(f"Running with device {device_type}.")

    print()
    print(f"C={C_VALUES[max_terms]}")
    print(f"D={D_VALUES[max_terms]}")
    print()

    model = nLSEModel(max_terms, batch_size).to(device)
    trainer = nLSETrainer(
        model=model, 
        batch_size=batch_size, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate, 
        device=device
    )

    # Evaluate error BEFORE training
    loss_before = trainer.evaluate_error()
    print(f"Error before training: {loss_before:.2f}%")

    # Trained values
    C_new, D_new = trainer.train()

    # Replace model's C and D with the newly learned ones for evaluation
    model.C.data = C_new.to(device)
    model.D.data = D_new.to(device)

    # Evaluate error AFTER training
    loss_after = trainer.evaluate_error()
    print(f"Error after training: {loss_after:.2f}%")

    # New constants
    print(f"\nLearned C={C_new}")
    print(f"Learned D={D_new}")

    # Update parameters in constants/updated_constants.pt
    # C_VALUES[max_terms] = C
    # D_VALUES[max_terms] = D

if __name__ == '__main__':
    test_model(7)

""" A more complete version of the training loop (in theory)
if __name__ == '__main__':
    # Device for potential GPU acceleration
    device_type = 'cpu'
    if torch.cuda.is_available():
        device_type = 'cuda'
    elif torch.backends.mps.is_available():
        device_type = 'mps'

    device = torch.device(device_type)

    # Array of all possible max_terms
    all_max_terms = [*range(1, 11), 15, 20]

    all_max_terms = [10] # MANUAL OVERRIDE JUST FOR TESTING

    for max_terms in all_max_terms:
        # Create and train model
        model = nLSEModel(max_terms, batch_size).to(device)
        trainer = nLSETrainer(model=model, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate, device=device)
        C, D = trainer.train()

        # Update parameters in constants/updated_constants.pt
        C_VALUES[max_terms] = C
        D_VALUES[max_terms] = D
"""
