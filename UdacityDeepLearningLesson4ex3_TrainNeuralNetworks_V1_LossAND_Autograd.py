import torch.nn.functional as F
import torch
import sys
sys.path.insert(1, '/Users/user/Downloads/deep-learning-v2-pytorch-master/intro-to-pytorch/')
import helper
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets, transforms
from torch import optim



transform = transforms.Compose([transforms.ToTensor(),
					transforms.Normalize((0.5,), (0.5,)),
					])
					
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

model = nn.Sequential(nn.Linear(784, 128),
						nn.ReLU(),
						nn.Linear(128,64),
						nn.ReLU(),
						nn.Linear(64, 10),
						nn.LogSoftmax(dim = 1))
						
optimizer = optim.SGD(model.parameters(), lr=0.01)

#define the loss (Negative Log Likelyhood Loss = NLLLoss)
criterion = nn.NLLLoss()

#print('Initial Weights: ', model[0].weight)

epochs = 5
for e in range(epochs):
	running_loss = 0
	for images, labels in trainloader:
		#flatten image into a 784 long vector
		#image.shape[0] is a shortcut for batch size (64 in this case).
		#SEE torchTensorViewExperiments.py as an example on .shape
		#The -1 is to flatten the image. 
		#-1 is a shortcut to the input units.  It's used in replace of knowing the exact input unit value
		images = images.view(images.shape[0], -1)
		optimizer.zero_grad()
		
		#logit layer - THE OUTPUT of the fully connected layer and produce raw prediction values.
		#logits AKA outputs
		#in the case of MNIST, the output will be a tensor of 10 values, where each value represents a score of each class (The images of the numbers 0-9)
		#the final output of this tensor will be the following shape: [batch_size, 10]....[64, 10]
		#in the end, get the final value which represents the probability of each target class.  This is done so by 
		#applying the softmax activation to the output of the logits layer (torch.nn.LogSoftmax(logits))
		output = model(images)
		loss = criterion(output, labels)
		loss.backward()
		optimizer.step()
		
		#6JUL - Research loss.item()...not sure what this is.
		running_loss += loss.item() #6JUL - Research loss.item()...not sure what this is. 
	else:
		print(f"Training Loss: {running_loss/len(trainloader)}")
		
images, labels = next(iter(trainloader))
img = images[0].view(1, 784)
with torch.no_grad():
	logps = model(img)
	
ps = torch.exp(logps)
helper.view_classify(img.view(1,28,28), ps)
plt.show()


		