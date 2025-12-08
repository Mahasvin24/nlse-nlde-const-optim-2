import torch
import matplotlib.pyplot as plt
from utils import nlse

# Load constants
orig_data = torch.load("constants/orig_constants.pt")
learned_data = torch.load("constants/learned_constants.pt")

C_OLD = orig_data["C_VALUES"]
D_OLD = orig_data["D_VALUES"]

C_NEW = learned_data["C_VALUES"]
D_NEW = learned_data["D_VALUES"]


def nlse_addition_graph(max_terms: int):
    C_old = C_OLD[max_terms]
    D_old = D_OLD[max_terms]

    C_new = C_NEW[max_terms]
    D_new = D_NEW[max_terms]

    print(C_old)
    print(C_new)

    xs = torch.linspace(0, 2.0, 400)

    # Exact curve (delay space)
    ys_exact_delay = -torch.log(torch.exp(-xs) + torch.exp(xs))

    # Approximation helper
    def compute_approx(xs, C, D):
        vals = []
        for x_prime in xs:
            y_prime = -x_prime
            candidates = [x_prime, y_prime]
            for Ci, Di in zip(C.flatten(), D.flatten()):
                candidates.append(torch.max(x_prime + Ci, y_prime + Di))
            vals.append(torch.min(torch.stack(candidates)))
        return torch.stack(vals)

    ys_approx_old = compute_approx(xs, C_old, D_old)
    ys_approx_new = compute_approx(xs, C_new, D_new)

    # Plot
    plt.figure(figsize=(7, 4))

    # Exact addition
    plt.plot(
        xs.numpy(), 
        ys_exact_delay.numpy(),
        label="Exact LSE(x', -x')", 
        color="blue"
    )

    # nLSE with old values
    plt.plot(
        xs.numpy(), 
        ys_approx_old.numpy(),
        label=f"{max_terms} max-term approx (old)", 
        color="orange"
    )

    # nLSE with new values
    plt.plot(
        xs.numpy(), 
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


if __name__ == "__main__":
    nlse_addition_graph(3)
 