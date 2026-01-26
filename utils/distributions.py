import torch
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

class BucketDistribution:
    def __init__(self, bucket_size=1.0):
        self.data = defaultdict(int)  # bucket_index : count
        self.data_points = 0
        self.bucket_size = bucket_size

        # Sampling state
        self.bucket_indices = np.array([])
        self.bucket_counts = np.array([])
        self.probs = np.array([])
        self.update_sampling = True

    def add(self, val):
        """
        Add a scalar or torch.Tensor of values (vectorized)
        """
        if isinstance(val, torch.Tensor):
            val = val.flatten()
            # Compute integer bucket indices
            bucket_indices = (val / self.bucket_size).to(torch.int64)
            # Count occurrences using bincount
            min_idx = bucket_indices.min().item()
            # Shift indices to start at 0 for bincount
            shifted_indices = bucket_indices - min_idx
            counts = torch.bincount(shifted_indices)
            # Update defaultdict using actual bucket indices
            for i, count in enumerate(counts.tolist()):
                if count > 0:
                    self.data[i + min_idx] += count
            self.data_points += val.numel()
        else:
            idx = int(val / self.bucket_size)
            self.data[idx] += 1
            self.data_points += 1

        self.update_sampling = True

    def update(self):
        """
        Recalculate bucket arrays and probabilities for sampling
        """
        if self.data_points == 0:
            raise ValueError("No data to update.")
        self.bucket_indices = np.array(list(self.data.keys()), dtype=np.int64)
        self.bucket_counts = np.array([self.data[idx] for idx in self.bucket_indices], dtype=float)
        self.probs = self.bucket_counts / self.data_points
        self.update_sampling = False

    def sample(self, n: int):
        """
        Sample n points from the bucketed distribution (returns torch.Tensor of floats)
        """
        if self.data_points == 0:
            raise ValueError("No data points to sample from.")
        if n <= 0:
            raise ValueError("n must be positive.")

        if self.update_sampling:
            self.update()

        # Sample bucket indices according to probabilities
        sampled_buckets = np.random.choice(self.bucket_indices, size=n, p=self.probs)
        sampled_buckets = torch.tensor(sampled_buckets, dtype=torch.float32)
        # Uniformly sample within each bucket
        samples = sampled_buckets * self.bucket_size + torch.rand(n) * self.bucket_size
        return samples

    def graph(self):
        """
        Display a histogram of the distribution
        """
        if self.data_points == 0:
            raise ValueError("No data to graph.")
        buckets = sorted(self.data.keys())
        counts = np.array([self.data[b] for b in buckets], dtype=float)
        percents = 100.0 * counts / self.data_points
        x = [(b + 0.5) * self.bucket_size for b in buckets]

        plt.figure()
        plt.bar(x, percents, width=self.bucket_size, align="center")
        plt.xlabel("Value")
        plt.ylabel("Percent (%)")
        plt.title("Bucketed Probability Distribution")
        plt.show()
