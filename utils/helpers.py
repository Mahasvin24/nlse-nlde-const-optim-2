import torch

# ---------------------------
# Helper Methods
# ---------------------------

# Uniform test value generation
def uniform_values(count: int) -> torch.Tensor:
    """
    Creates uniform values in range [0, 1)

    Returns a column vector tensor
    """
    return torch.rand(count).reshape(-1, 1)

# Matches definition in paper (a INHIBITS b)
def inhibit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """ Fully vectorized version that mimics the temporal inhibit function """
    return torch.where(b < a, b, torch.inf)

# Range-normalized Root Mean Squared Error (used as error metric in paper)
def rnrmse(pred, target):
    mse = torch.mean((pred - target)**2)
    rmse = torch.sqrt(mse)
    data_range = target.max() - target.min()
    return rmse / data_range

# Gaussian noise
def guassian_noise(n, device, mean=0.0, std=0.01):
    """
    Generate Gaussian noise centered at 0.
    """
    return torch.normal(mean=mean, std=std, size=(n, 1), device=device)