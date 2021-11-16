import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler

##########################
##check if CUDA is avail##
##########################
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
	print('You do not have CUDA')
else:
	print('You do not have CUDA')
	
#numbers of subprocesses to use for data loading
num_workers = 0
#how many samples per batch to load
batch_size = 20
#percentage of training set to use as validation
valid_size = 0.2

#convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) #Normalize(mean,std)...one value each for each (Red,Blue,Green) color channel
	])

#establish the training and test datasets
train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data =datasets.CIFAR10('data', train=True, download=True, transform=transform)

######################################
##OBTAIN VALIDATION TRAINING INDICES##
######################################
num_train = len(train_data) # Value of X (ie 60000)
indices = list(range(num_train))# create a list of values 0 -> X-1 (59000)
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train)) # Value of X * 0.2 ( ie 12000 (0.2 * 60000))
train_idx, valid_idx = indices[split:], indices[:split] #indices[48000:], indices[:12000]

#define the samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

#prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

#specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

'''
#visualize a batch of training data
def imshow(img):
	img = img/2 + 0.5 #unnormalize....I'll have to tweak with this to understand
	plt.imshow(np.transpose(img, (1,2,0))) #convert from Tensor image

#obtain one batch of training data
images, labels = next(iter(train_loader))
images = images.numpy()

################################
##PLOT EXAMPLE BATCH OF IMAGES##
################################
fig = plt.figure(figsize=(10,6)) #The 10, 6 is the inches in the plot.
for imgx in np.arange(batch_size): # a range of batch_size (ie 20)
	ax = fig.add_subplot(2, 10, imgx+1, xticks=[], yticks=[])
	imshow(images[imgx])
	ax.set_title(classes[labels[imgx]])

##################################################
##PLOT THREE IMAGES AND THE VALUES OF EACH PIXEL##
##################################################
rgb_image = np.squeeze(images[3]) #squeeze - multiple tensor ranks into one tensor vector
channels =(['red channel','green channel','blue channel'])

fig = plt.figure(figsize = (10,5))
for idx in np.arange(rgb_image.shape[0]):
	ax = fig.add_subplot(1, 3, idx + 1)
	img = rgb_image[idx]
	ax.imshow(img, cmap='gray')
	ax.set_title(channels[idx])
	width, height = img.shape
	thresh = img.max()/2.5
	for x in range(width):
		for y in range(height):
			val = round(img[x][y],2) if img[x][y] != 0 else 0
			ax.annotate(str(val), xy=(y,x),
				horizontalalignment = 'center',
				verticalalignment = 'center', 
				color='white' if img[x][y]<thresh else 'black')
				
plt.show()
'''


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		#NOTE: https://stats.stackexchange.com/questions/291820/what-is-the-definition-of-a-feature-map-aka-activation-map-in-a-convolutio/292064#292064
		'''
		NOTE BELOW: 
		nn.Conv2d(3, 16, 3) = 
		3 for RGB channels of image (depth of the input), 
		16 filtered images (desired depth of output), 
		3 is equal to a filter (window) of (3x3)
		'''
		self.conv1 = nn.Conv2d(3, 16, 3, padding=1) 
		self.conv2 = nn.Conv2d(16, 32, 3, padding=1) #input 16 filters, output 32 filters, window(3x3)
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1) #input 32 filters, output 64 filters, window(3x3
		#maxpooling layer: Take Convolutional layers as inputs and reduces the x-y input and keeps only the most active (highest value) pixels
		#INPUT: a stack of feature maps
		#OUTPUT: the pixel maximum value for each window snapshot
		self.pool = nn.MaxPool2d(2, 2) # (2,2) = Window Size of 2 i.e. (2x2) and a Stride of 2
		self.fc1 = nn.Linear(64 * 4 * 4, 500) #fully connected one
		self.fc2 = nn.Linear(500, 10) #fully connected two
		self.dropout = nn.Dropout(0.25)
		
	def forward(self, x):
		#add sequence of convolutional and max pooling layers
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = torch.flatten(x, start_dim=1) #x = x.view(-1, 16 * 5 * 5)
		x = self.dropout(x)
		x = F.relu(self.fc1(x)) #1st hidden layer
		x = self.dropout(x)
		x = self.fc2(x)
		return x
		
model = Net() #create the CNN

criterion = nn.CrossEntropyLoss() #specify loss function 

optimizer = optim.SGD(model.parameters(), lr=0.01) #specify optimizer

print('*' * 21)
print('** Train the Model **')
print('*' * 21)

#number of epochs to train the model
n_epochs = 8

valid_loss_min = np.Inf #track change in validation loss

for epoch in range(1,n_epochs+1):
	
	#keep track of training and validation loss
	train_loss = 0.0
	valid_loss = 0.0
	
	###################
	##TRAIN THE MODEL##
	###################
	model.train()
	for data, target in train_loader:
		optimizer.zero_grad() #clear gradients for the optimized variable
		output = model(data) #forward pass: compute outputs by passing inputs into the model
		loss = criterion(output, target) #calculate the loss
		loss.backward() #backward pass: compute the gradient loss with respects to the model parameters
		optimizer.step() #perform a single optimization step (parameter update)
		train_loss += loss.item() * data.size(0) #update train loss
	
	######################	
	##VALIDATE THE MODEL##
	######################
	model.eval()
	for data, target in valid_loader:
		output = model(data) #forward pass: compute outputs by passing inputs into the model
		loss = criterion(output, target)
		valid_loss = loss.item() * data.size(0) #update valid loss 
		
	train_loss = train_loss/len(train_loader.dataset)
	valid_loss = valid_loss/len(valid_loader.dataset)
	
	print('Epoch: {} \tTraining Loss: {} \tValidation Loss: {}'.format(epoch, train_loss, valid_loss))
	
	if valid_loss <= valid_loss_min:
		print('Validation loss has decreased from {} --> {}'.format(valid_loss_min, valid_loss))
		torch.save(model.state_dict(), 'modelLesson5_CIFAR10.pt')
		valid_loss_min = valid_loss


	
	





	
