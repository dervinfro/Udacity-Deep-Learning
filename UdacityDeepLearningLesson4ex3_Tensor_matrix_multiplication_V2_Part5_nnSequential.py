import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/Users/user/Downloads/deep-learning-v2-pytorch-master/intro-to-pytorch/')
import helper
from torch import nn
from torchvision import datasets, transforms
from collections import OrderedDict


transform = transforms.Compose([transforms.ToTensor(),
					transforms.Normalize((0.5,), (0.5,)),
					])
					
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#Hyperparameters for our network
input_size = 784
hidden_size = [128, 64]
output_size = 10

model = nn.Sequential(OrderedDict([
						('fc1', nn.Linear(input_size, hidden_size[0])),
						('relu1', nn.ReLU()),
						('fc2', nn.Linear(hidden_size[0], hidden_size[1])),
						('relu2', nn.ReLU()),
						('output', nn.Linear(hidden_size[1], output_size)),
						('softmax', nn.Softmax(dim = 1))]))
						
						
#print(model)

images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)	
plt.show()		