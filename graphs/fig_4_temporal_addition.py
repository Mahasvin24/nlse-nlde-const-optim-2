import torch
import matplotlib.pyplot as plt
from utils.helpers import rnrmse
from utils.temporal_artithmetic import nlse

# Getting number of max terms from user
max_terms = int(input("How many max terms? "))

# Load pytorch files
orig_data = torch.load("constants/orig_constants.pt")
learned_data = torch.load("constants/learned_constants.pt")

# Storing constants
C_old = orig_data["C_VALUES"][max_terms]
D_old = orig_data["D_VALUES"][max_terms]
C_new = learned_data["C_VALUES"][max_terms]
D_new = learned_data["D_VALUES"][max_terms]

# Printing constants for comparison
print()
print(C_old)
print(C_new)

# Data generation (linear x' values)
points = 800
x_delay = torch.linspace(0, 2.0, points)

# Exact curve (delay space)
ys_exact_delay = -torch.log(torch.exp(-x_delay) + torch.exp(-(-x_delay)))

# Approximations
ys_approx_old = nlse(x_delay, -x_delay, C_old, D_old)
ys_approx_new = nlse(x_delay, -x_delay, C_new, D_new)

print()
print(f"Old line error: {rnrmse(ys_approx_old, ys_exact_delay)}")
print(f"New line error: {rnrmse(ys_approx_new, ys_exact_delay)}")

# Plot
plt.figure(figsize=(7, 4))

# Exact addition
plt.plot(
    x_delay.numpy(), 
    ys_exact_delay.numpy(),
    label="Exact LSE(x', -x')", 
    color="blue"
)

# nLSE with old values
plt.plot(
    x_delay.numpy(), 
    ys_approx_old.numpy(),
    label=f"{max_terms} max-term approx (old)", 
    color="orange"
)

# nLSE with new values
plt.plot(
    x_delay.numpy(), 
    ys_approx_new.numpy(),
    label=f"{max_terms} max-term approx (new)",
    color="#66cc66", alpha=0.6
) 

# Graphing
plt.xlabel("x'")
plt.ylabel("y'")
plt.title("nLSE(x', -x'): Exact vs Old vs New Constants")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.xlim(0, 2)
plt.tight_layout()
plt.show()
 