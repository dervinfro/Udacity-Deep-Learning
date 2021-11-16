import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torch import optim

#number of sub processes to use for work load
num_workers = 0

#how many samples per batch to load
batch_size = 20
#percentage of training set to use for validation
valid_size = 0.2

#transform data to torch.FloatTensor
transform = transforms.ToTensor()

#choose the training and test datasets
train_data = datasets.MNIST('~/.pytorch/MNIST_data/', train=True, download=True, transform=transform)
test_data = datasets.MNIST('~/.pytorch/MNIST_data/', train=False, download=True, transform=transform)

######################################
##OBTAIN VALIDATION TRAINING INDICES##
######################################
num_train = len(train_data) #value of 60000
indices = list(range(num_train)) #creates a list of the values: 0 -> 59999
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train)) #value of 12000
train_idx, valid_idx = indices[split:], indices[:split] #indices[48000:], indices[:12000)
#split[start:stop:step]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


#prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

#obtain one batch of training images
images, labels = next(iter(train_loader))
images = images.numpy() #calling the numpy method to get access to the tensor values


##VISUALIZE A BATCH OF TRAINING DATA (FOR THE NEXT SIX (6) LINES OF CODE)
##plot the images in a batch with the corresponding label
#fig = plt.figure(figsize=(20,5))
#for image_x in np.arange(20):
#	ax = fig.add_subplot(2, 20/2, image_x+1, xticks=[], yticks=[])
#	ax.imshow(np.squeeze(images[image_x]), cmap='gray')
#	ax.set_title(str(labels[image_x].item()))
	

##VIEW AN IMAGE IN MORE DETAIL (FOR THE NEXT THIRTEEN (13) LINES OF CODE.)
##BLACK VALUES ARE ZERO (0) AND WHITE VALUES ARE ONE (1).
#img = np.squeeze(images[1])
#
#fig = plt.figure(figsize=(12,12))
#ax = fig.add_subplot(111)
#ax.imshow(img, cmap='gray')
#width, height = img.shape
#thresh = img.max()/2.5
#for x in range(width):
#	for y in range(height):
#		val = round(img[x][y],2) if img[x][y] != 0 else 0
#		ax.annotate(str(val), xy=(y,x),
#								horizontalalignment='center',
#								verticalalignment='center',
#								color='white' if img[x][y]<thresh else 'black')
#	
#plt.show()

#Define the nn Architecture
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__() 
		#super() combined with the nn.Module, creates a class that tracks the architecture that provides alot of useful methods and attributes.  It's mandatory that we inherit the nn.Module
		self.fc1 = nn.Linear(784, 512)
		self.fc2 = nn.Linear(512, 512)
		self.fc3 = nn.Linear(512, 10)
		self.dropout = nn.Dropout(0.2)
		
	#The forward method is created to pass the input tensor through each of the operations.
	def forward(self, x):
		x = torch.flatten(x, start_dim=1)
		
		#The Rectified Linear Unit object is an activation function that is almost exclusively used in hidden layers.
		x = F.relu(self.fc1(x))
		x = self.dropout(x)
		x = F.relu(self.fc2(x))
		x = self.dropout(x)
		#There is no output activation fuction applied to fc3(typically LogSoftmax for final layer).
		#The output activation function is applied in the CrossEntrophyLoss
		x = self.fc3(x)
		
		return x
		
model = Net()
		
#specify loss function
criterion = nn.CrossEntropyLoss()

#specify optimzer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

#number of epochs to train the model.  It's suggested between 20-50
n_epochs = 25

#initialize the valid loss tracker.  This will be used as a parameter to save the model
valid_loss_min = np.Inf

#the following lists will be used for graph animation
epoch_list = []
train_loss_list = []
valid_loss_list = []

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

'''
My Notes: 
loop through the number of epochs
	loop through the images and labels
		zeroize gradients
		forward pass
		loss
		optimizer step
'''

for epoch in range(n_epochs):
#	
##	#monitor the training loss
#	train_loss = 0.0
#	valid_loss = 0.0
##	
#	#####################
#	###TRAIN THE MODEL###
#	#####################
#	#sets the module in training mode
#	model.train() 
#	for image, label in train_loader:
#		#clear the gradients of all of the optimized variables
#		optimizer.zero_grad()
#		#forward pass: compute predicted outputs by passing the inputs to the model
#		output = model(image)
#		#calculate the loss
#		loss = criterion(output, label)	
#		#backward pass: compute the gradient loss with respect to the model parameters
#		loss.backward()
#		#perform a single optimzation step (parameter update)
#		optimizer.step()	
#		#update running training loss
#		train_loss += loss.item()*image.size(0)
#		
#	#####################
#	#VALIDATE THE MODEL##
#	#####################
#	model.eval()
#	for image, label in valid_loader:
#		#forward pass: compute predicted outputs by passing the inputs to the model
#		output = model(image)
#		#calculate loss
#		loss = criterion(output, label)
#		#update the running validation loss
#		valid_loss += loss.item()*image.size(0)
#		
#		
#	#calculate loss over an epoch
#	train_loss = train_loss/len(train_loader.sampler)
#	valid_loss = valid_loss/len(valid_loader.sampler)
#	
#	epoch_list.append(epoch+1)
#	train_loss_list.append(train_loss)
#	valid_loss_list.append(valid_loss)
#
#	#print training loss statistics
#	print('Epoch {} \tTraining Loss: {} \tValid Loss: {} '.format(epoch+1, train_loss, valid_loss))
#	
#	if valid_loss <= valid_loss_min:
#		print("Validation loss has decreased: {} --> {}".format(valid_loss_min, valid_loss))
#		torch.save(model.state_dict(), 'modelMLP.pt')
#		valid_loss_min = valid_loss
			
#def animate(i):
#	ax.clear()
#	ax.plot(epoch_list, train_loss_list, marker='o')
#	ax.set_xlim(0,25)
#	ax.set_ylim(0,2)
#	
#ani = animation.FuncAnimation(fig, animate, interval=1000)		
#plt.show()

model.load_state_dict(torch.load('modelMLP.pt'))

#INITIALIZE LIST TO MONITOR TESTS AND ACCURACY (FOR THE NEXT 36 LINES OF CODE)
test_loss = 0
class_correct = list(0. for i in range(10)) 
class_total =  list(0. for i in range(10))

model.eval() #prep model for evaluation

'''
loop through the images and labels **MY PSEUDOCODE**
		forward pass
		loss
		optimizer step
'''

##################
##TEST THE MODEL##
##################
for image, label in test_loader:
	#forward pass: compute predicted outputs by passing inputs to the model
	output = model(image)
	#calculate loss
	loss = criterion(output, label)
	#update test loss
	test_loss += loss.item()*image.size(0)
	#convert output probabilites to predicted class
	_, pred = torch.max(output, 1)
	#compare predictions to true labels
	correct = np.squeeze(pred.eq(label.data.view_as(pred)))
	
	for i in range(len(label)):
		label_correct = label.data[i]
		class_correct[label_correct] += correct[i].item()
		class_total[label_correct] += 1
		
#calculate and print average loss
test_loss = test_loss/len(test_loader.sampler)
print('Test Loss: {}\n'.format(test_loss))

for i in range(10):
	if class_total[i] > 0:
		print('Test Accuracy of {}, {}:'.format(str(i), (100 * class_correct[i]/class_total[i])))
	else:
		print('Test Accuracy of (no training examples)')

 




