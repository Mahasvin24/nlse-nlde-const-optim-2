import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Loading constants
data = torch.load("constants/orig_constants.pt")
if "E_VALUES" not in data or "F_VALUES" not in data:
    print("Value fetch failed.")
    sys.exit()
E_VALUES = data["E_VALUES"]
F_VALUES = data["F_VALUES"]

# From delaynet repo (nohistogram):
# export APPROX_E='4.022519660949931,2.0592571987398305,0.9954588501779946,0.16190406334166876,-0.6408836354345162,-1.5755039057092928,-3.000991236364557'
# export APPROX_F='4.022519660949931,2.10719415702947,1.2071748895294314,0.6611901397236745,0.3126857477233991,0.10153442821645398,0.0'

print(f"E_values: {E_VALUES[7]}")
print(f"F_values: {F_VALUES[7]}")
print()

def inhibit(a, b):
    return b if b < a else torch.tensor(float('inf'))

def paper_nlde(x_p, y_p, max_terms=7):
    """Nonvectorized min-of-inhibit nLDE approximation."""
    # Compute inhibit terms for each constant pair
    inhibit_terms = []

    E = E_VALUES[max_terms]
    F = F_VALUES[max_terms]

    for i in range(max_terms):
        e_val = E[0][i].item()
        f_val = F[0][i].item()

        term = inhibit(x_p + e_val, y_p + f_val)
        inhibit_terms.append(term)

    # Find minimum across all terms
    min_term = min(inhibit_terms)
    return min_term


def my_nlde(x_p, y_p, max_terms=7):
    """Nonvectorized min-of-inhibit nLDE approximation."""
    # Compute inhibit terms for each constant pair
    inhibit_terms = []

    E = E_VALUES[max_terms]
    F = F_VALUES[max_terms]

    for i in range(max_terms):
        e_val = E[0][i].item()
        f_val = F[0][i].item()

        term = inhibit(y_p + e_val, x_p + f_val)
        inhibit_terms.append(term)

    # Find minimum across all terms
    min_term = min(inhibit_terms)
    return min_term

if __name__ == "__main__":
    # Testing values
    x = 0.5
    y = 0.3

    # Delay space conversion
    x_p = - np.log(x)
    y_p = - np.log(y)

    x_p = torch.tensor(x_p)
    y_p = torch.tensor(y_p)

    my_sol = my_nlde(x_p, y_p)
    paper_sol = paper_nlde(x_p, y_p)

    # Convert back to importance space
    my_sol = torch.exp(- my_sol)
    paper_sol = torch.exp(- paper_sol)

    print(f"My nLDE approximation: {my_sol:.3f}")
    print(f"Paper nLDE approximation: {paper_sol:.3f}")


