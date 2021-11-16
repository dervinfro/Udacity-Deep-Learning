import torch.nn.functional as F
import torch
import sys
sys.path.insert(1, '/Users/user/Downloads/deep-learning-v2-pytorch-master/intro-to-pytorch/')
import helper
from torch import nn
from torchvision import datasets, transforms

x = torch.randn(2,2, requires_grad=True)
print(x)
y = x ** 2
print(y)
print(y.grad_fn)
z = y.mean()
print(z)
print(x.grad_fn)
z.backward()
print(x.grad)
print(x/2)
