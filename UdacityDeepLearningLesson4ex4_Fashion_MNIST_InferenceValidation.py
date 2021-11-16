import torch
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/Users/user/Downloads/deep-learning-v2-pytorch-master/intro-to-pytorch/')
import helper
from torch import nn
from torchvision import datasets, transforms
from torch import optim


#Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize((0.5), (0.5))])	

#Download and train the data	
#Don't touch this until the very end.	(https://elitedatascience.com/overfitting-in-machine-learning)							
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)	
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#Download and test the data
#Train and tune the model using this set.  (SEE ABOVE LINK)
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


#inputs/layers/outputs....to include activation functions
#output = 10 (10 different classes) Softmax Activation
input_size = 784
hidden_size_layer_one = [256, 128]
hidden_size_layer_two = [128, 64]
output_size = 10

#use Dropout in the layers to assist with overfitting
model = nn.Sequential(nn.Linear(input_size, hidden_size_layer_one[0]),
						nn.ReLU(),
						nn.Dropout(p=0.2),
						nn.Linear(hidden_size_layer_one[0], hidden_size_layer_one[1]),
						nn.ReLU(),
						nn.Dropout(p=0.2),
						nn.Linear(hidden_size_layer_two[0], hidden_size_layer_two[1]),
						nn.ReLU(),
						nn.Dropout(p=0.2),
						nn.Linear(hidden_size_layer_two[1], output_size),
						nn.LogSoftmax(dim=1))
						

#Negative Log Likelyhood Loss
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.003)
epochs = 10
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
	running_loss = 0
	for image, label in trainloader:
		optimizer.zero_grad()
		
		images = torch.flatten(image, start_dim=1)
		log_ps = model(images)
		loss = criterion(log_ps, label)
		loss.backward()
		optimizer.step()
		
		running_loss += loss.item()
		
	else:
		test_loss = 0
		accuracy = 0
		
		with torch.no_grad():
			model.eval() #this turns Dropout off
			for image, label in testloader:
				image_test = torch.flatten(image, start_dim=1)
				log_ps = model(image_test)
				test_loss += criterion(log_ps, label)
				ps = torch.exp(log_ps)
				#See torchTensorViewExperiments.py for example on topk
				top_p, top_class = ps.topk(1, dim=1)
				equals = top_class == label.view(*top_class.shape)
				accuracy = torch.mean(equals.type(torch.FloatTensor))
				
		model.train()
		
		train_losses.append(running_loss/len(trainloader))
		test_losses.append(test_loss/len(testloader))
				
		print(  "Epoch: {}/{}...".format(e+1, epochs),
				"Training Loss: {:.3f}..".format(running_loss/len(trainloader)),
				"Testing Loss: {:.3f}..".format(test_loss/len(testloader)),
				"Accuracy: {:.3f}..".format(accuracy/len(testloader)))
				
plt.plot(train_losses, label='Training Losses')
plt.plot(test_losses, label='Validation Losses')
plt.legend(frameon=False)
plt.show()

#image, label = next(iter(trainloader))
#print("Label Shape: ", label.shape)
#print("Label: ", label)
#images = torch.flatten(image, start_dim=1)
##64 samples and 10 classes
#ps = torch.exp(model(images))
#print("PS Shape: ", ps.shape)
#top_p, top_class = ps.topk(1, dim=1)
#print("top class shape: ", top_class.shape)
#print("top class: ", top_class)
#equals = top_class == label.view(*top_class.shape)
#print("label.view(*top_class.shape): ", label.view(*top_class.shape))
#print("equals shape: ", equals.shape)
#print("equals: ", equals)
#accuracy = torch.mean(equals.type(torch.FloatTensor))


