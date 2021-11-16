import torch.nn.functional as F
import torch
import sys
sys.path.insert(1, '/Users/user/Downloads/deep-learning-v2-pytorch-master/intro-to-pytorch/')
import helper
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets, transforms
from torch import optim


#Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize((0.5), (0.5))])	#Normalize(mean, std)...only one parameter cause there is only on color.

#Download and train the data									
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)	
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#Download and test the data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

#inputs/layers/outputs....to include activation functions
#inputs = 784
#L1 = 490 (RELU)
#l2 = 64 (RELU)
#output = 10 (10 different classes) Softmax Activation
input_size = 784
hidden_size_layer_one = [256, 128]
hidden_size_layer_two = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_size_layer_one[0]),
						nn.ReLU(),
						nn.Linear(hidden_size_layer_one[0], hidden_size_layer_one[1]),
						nn.ReLU(),
						nn.Linear(hidden_size_layer_two[0], hidden_size_layer_two[1]),
						nn.ReLU(),
						nn.Linear(hidden_size_layer_two[1], output_size),
						nn.LogSoftmax(dim=1))
						

optimizer = optim.SGD(model.parameters(), lr=0.01)

#Loss Criterion
criterion = nn.NLLLoss()


epochs = 5
for e in range(epochs):
	running_loss = 0
	for image, label in trainloader:
		#flatten image into a 784 long vector
		#image.shape[0] is a shortcut for batch size (64 in this case).
		#SEE torchTensorViewExperiments.py as an example on .shape
		#The -1 is to flatten the image. 
		#-1 is a shortcut to the input units.  It's used in replace of knowing the exact input unit value
#		images = image.view(image.shape[0], -1)
		images = torch.flatten(image, start_dim=1)
		#clears the gradients of all of the optimized torch tensors.
		optimizer.zero_grad()
		
		output = model(images)
		loss = criterion(output, label)
		loss.backward()
		optimizer.step()
		
		running_loss += loss.item()
		
	else:
		print(f'Training Loss: {running_loss/len(trainloader)}')
		
		
image, label = next(iter(trainloader))
img = image[0].view(1,784)
with torch.no_grad():
	logps = model(img)
	
ps = torch.exp(logps)
helper.view_classify(img.view(1,28,28), ps)
plt.show()