import ray
import torch

# ray.init()
print(torch.cuda.is_available())
# print(ray.cluster_resources())


# @ray.remote
def func(a):
    return a.cpu().numpy()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = torch.tensor([1, 2, 3], device=device, dtype=torch.float)

print(func(a))

print(a)
b = func.remote(a)
print(ray.get(b))
