import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/Users/user/Downloads/deep-learning-v2-pytorch-master/intro-to-pytorch/')
import helper
from torch import nn
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
					transforms.Normalize((0.5,), (0.5,)),
					])
					
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


class Network(nn.Module):
	def __init__(self):
		super().__init__()
		
		#hidden object is created.  The input layer is 784 and the hidden layer is 128.
		self.fc1 = nn.Linear(784, 128)
		
#		#hidden object is created.  The hidden layers are 128 and 64.
		self.fc2 = nn.Linear(128, 64)
		
		
		self.fc3 = nn.Linear(64, 10)
		
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.softmax(self.fc3(x), dim = 1)
		'''
		ReLU (Rectified Linear Unit) object above is an activation function that is almost exclusively used in hidden layers.
		It is an economical choice for activation functions.
		'''
		
		return x
		
model = Network()
print(model)
#print(model.fc1.weight)
#print(model.fc1.bias)
model.fc1.bias.data.fill_(0)
model.fc1.weight.data.normal_(std=0.01)
#print(model, '\n', model.fc1.weight, '\n', model.fc1.bias)

dataiter = iter(trainloader)
images, labels = dataiter.next()
#Resize images into a 1D vector.  New shape is (batch size, color, image pixels (28 * 28))
#The [0] in images.shape, is to automatically get the batch size w/o having to know the exact batch size
#it is the same as images.shape[64] (which is the batch size for this network)
#images.resize_(images.shape[0], 1, 784)
images.resize_(64,1,784)
print(type(images))
print(images.shape)
print(labels.shape)

#plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');
#plt.show()

img_idx = 0
#Get the class probabilities (labeled here as ps)
ps = model.forward(images[img_idx, :])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)
plt.show()

		
		