from utils.distributions import BucketDistribution
import time
import torch

# Timer
start = time.time()

dist = BucketDistribution(bucket_size=0.2)

# Testing with normal vals
count = 100000
vals = torch.normal(mean=12, std=2, size=(count,))

dist.add(vals)

# Stats n stuff
print("Total data points:", dist.data_points)
print("Buckets:")
for bucket in sorted(dist.data):
    lo = bucket * dist.bucket_size
    hi = lo + dist.bucket_size
    print(f"[{lo:.1f}, {hi:.1f}): {dist.data[bucket]}")

# Sampling
samples = dist.sample(10)
print("\nSamples:", samples)

# End
end = time.time()
print(f"TIME: {end - start:.5f} (for {count} data points)")

# Graphing
dist.graph()