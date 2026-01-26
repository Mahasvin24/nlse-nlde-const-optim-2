from utils.distributions import BucketDistribution
import time

# Timer
start = time.time()

dist = BucketDistribution(bucket_size=0.5)

# Add a variety of values
vals = [
    -2.3, -2.1, -1.9, -1.2, -1.0, -0.8,
    -0.1,  0.0,  0.2, 0.6,  0.7,  0.9,
     1.4,  1.6,  1.8, 2.1,  2.4,  2.6,
     10, 12, 12, 12, 12, 12, 12, 12, 12
]

for v in vals:
    dist.add(v)

# # Show internal state
print("Total data points:", dist.data_points)
print("Buckets:")
for bucket in sorted(dist.data):
    lo = bucket * dist.bucket_size
    hi = lo + dist.bucket_size
    print(f"[{lo:.1f}, {hi:.1f}): {dist.data[bucket]}")

# Sample from the distribution
samples = dist.sample(10)
print("\nSamples:", samples)

# End
end = time.time()
print(f"TIME: {end - start:.5f}")

# Visualize
dist.graph()