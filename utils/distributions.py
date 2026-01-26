import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Simple distribution class (for numerical values -- floats)
class BucketDistribution:
    def __init__(self, bucket_size=1.0, min_val=None, max_val=None):
        # Field variables
        self.data = defaultdict(int) # bucket : count
        self.data_points = 0
        self.bucket_size = bucket_size
        self.min_val = min_val
        self.max_val = max_val

        # Sampling state
        self.bucket_indicies = []
        self.bucket_counts = np.array([])  
        self.probs = np.array([])         
        self.update_sampling = True

    def bucket_of(self, val: float):
        """
        Calculating the bucket a data point belongs to
        """
        return int(np.floor(val / self.bucket_size))
    

    def add(self, val: float):
        """
        Adding to the data distribution
        """
        self.data[self.bucket_of(val)] += 1 # add to bucket
        self.data_points += 1               # add to overall class count
        self.update_sampling = True         # reupdate with new info

    def update(self):
        """
        Recalculating bucket indicies and probabilites for sampling
        """
        self.bucket_indicies = list(self.data.keys())
        self.bucket_counts = np.array(
            [self.data[idx] for idx in self.bucket_indicies], dtype=float
        )
        self.probs = self.bucket_counts / self.data_points
        self.update_sampling = False       

    def sample(self, n:int):
        """
        Sampling from the bucketed distribution
        """
        if self.data_points == 0:
            raise ValueError("No data points to sample from")
        if n <= 0:
            raise ValueError("n must be positive")
        
        if self.update_sampling:
            self.update()

        # Randomly choosing buckets
        sampled_buckets = np.random.choice(
            self.bucket_indicies, size=n, p=self.probs
        )
        
        # Sample uniformly within each bucket
        return np.array([
            np.random.uniform(
                idx * self.bucket_size,
                (idx + 1) * self.bucket_size
            )
            for idx in sampled_buckets
        ])
    
    def graph(self):
        """
        Graphing a bucketed histogram from distribution
        """
        if self.data_points == 0:
            raise ValueError("No data to graph.")

        # Sort buckets
        buckets = sorted(self.data.keys())
        counts = np.array([self.data[b] for b in buckets], dtype=float)

        # Convert counts to percentages
        percents = 100.0 * counts / self.data_points

        # Convert bucket indices to bucket centers
        x = [(b + 0.5) * self.bucket_size for b in buckets]

        plt.figure()
        plt.bar(x, percents, width=self.bucket_size, align="center")
        plt.xlabel("Value")
        plt.ylabel("Percent (%)")
        plt.title("Bucketed Probability Distribution")
        plt.show()

