import torch

orig_data = torch.load("constants/orig_constants.pt")
learned_data = torch.load("constants/learned_constants.pt")

C_OLD = orig_data["C_VALUES"]
C_NEW = learned_data["C_VALUES"]

print(C_OLD[4])
print(C_NEW[4])

