import torch
import numpy as np
import sys
sys.path.insert(1, '/Users/user/Downloads/deep-learning-v2-pytorch-master/intro-to-pytorch/')
import helper
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
					transforms.Normalize((0.5,), (0.5,)),
					])
					
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
'''
NOTE: Exercise: Flatten the batch of images images. Then build a multi-layer network with 784 input units, 256 hidden units, and 10 output units using random tensors for the weights and biases. For now, use a sigmoid activation for the hidden layer. Leave the output layer without an activation, we'll add one that gives us a probability distribution next.
'''


def activation(x):
	# sigmoid activation function
	return 1/(1 + torch.exp(-x))
	
def softmax(x):
	return torch.exp(x) / torch.sum(torch.exp(x), dim = 1).view(-1,1)

dataiter = iter(trainloader)
images, labels = dataiter.next()
inputs = images.view(images.shape[0], -1)
print(type(images))
print(images.shape)
print(labels.shape)

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');
plt.show()

'''
image.shape[0] refers to the index of the image list.
 takes in the first batch size, in this case 64 (SEE ABOVE: trainloader...batch_size = 64)
 and uses -1 to flatten the image.
 -1 is a shortcut to the input units.  It's used in replace of knowing the exact input unit value.
'''

W1 = torch.randn(784,256) 
# 784 is the 28 * 28 image (2D Array) flattened out into a 1D Array. What does the 256 relate to?
#weights for the inputs to the hidden layer
B1 = torch.randn(256)
#bias for the hidden layer

W2 = torch.randn(256,10)
#weights for the hidden to the output layer
# 10 output units...one for each of the classes.  In this case, one output for each of the numbers in the MNIST dataset.
B2 = torch.randn(10)
#bias for the output layer

hidden_layer_one = activation(torch.mm(inputs,W1) + B1)

output_layer = torch.mm(hidden_layer_one, W2) + B2

class_probabilities = softmax(output_layer)

print(class_probabilities.shape)

print(class_probabilities.sum(dim = 1))




	
