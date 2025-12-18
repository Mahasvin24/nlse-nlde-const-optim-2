import torch
import numpy
import math
import matplotlib.pyplot as plt
from utils.helpers import uniform_values

# Generating data (using set to remove duplicates)
x = uniform_values(500_000).squeeze()

# Temporal values
x_p = (- torch.log(x)).cpu().numpy()

# Range
minn = numpy.min(x_p)
maxx = numpy.max(x_p)
print(f"\nMin: {minn}")
print(f"Max: {maxx:.2f}\n")

# Graphing distrubution
plt.hist(x_p, 500, density=True)
plt.xlabel("x'")
plt.ylabel("Density")
plt.xlim(math.floor(minn), math.ceil(maxx))
plt.title("Distrbution in Delay Space of Uniform Importance Values")
plt.show()

