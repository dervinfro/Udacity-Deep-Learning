import sys
#sys.path below is used to ref "import helper"
sys.path.insert(1, '/Users/user/Downloads/deep-learning-v2-pytorch-master/intro-to-pytorch/')
import helper
import matplotlib.pyplot as plt
import time
import torch

from collections import OrderedDict
from torch import nn, optim
from torchvision import datasets, transforms, models

#define transforms
train_transforms = transforms.Compose([transforms.RandomRotation(30),
									transforms.RandomResizedCrop(224), #input images size: 224 x 224
									transforms.RandomHorizontalFlip(),
									#.ToTensor is similiar to numpy array but it can be moved to a GPU faster than a numpy array.
									transforms.ToTensor(),
									transforms.Normalize([0.485, 0.456, 0.406], 
														[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
									transforms.CenterCrop(224), #input images size: 224 x 224
									transforms.ToTensor()])


data_directory = '/Users/user/Downloads/Cat_Dog_data'


trainset = datasets.ImageFolder(data_directory + '/train', transform=train_transforms)
testset = datasets.ImageFolder(data_directory + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)

model = models.densenet121(pretrained=True)

'''
Flag the gradient descent as FALSE so that we don't backward propogate through them.
	This is flagged as FALSE because the network has already been trained.....see above
	models.densenet121
'''

for param in model.parameters():
	param.requires_grad=False
	
'''
The built-in classifier for densesnet121 is: 
(classifier): Linear(in_features=1024, out_features=1000, bias=True)

CHANGING THE CLASSIFIER:
The classifier is changed due to fact that model was trained on the ImageNet database.  In this case we're using the Cat_Dog dataset images.  Below the classifer is changed to meet the needs of our images.  Also, model.classifier = classifier is set to ensure that the new classifer is used for this training and testing.
'''

classifier = nn.Sequential(OrderedDict([
							('fc1', nn.Linear(1024, 500)),
							('relu', nn.ReLU()),
							('Dropout', nn.Dropout(0.2)),
							('fc2', nn.Linear(500, 2)),
							('output', nn.LogSoftmax(dim=1))
							]))


model.classifier = classifier
#cuda_device = torch.cuda.get_device_name(0)
#print('Cuda: ', cuda_device)

criterion = nn.NLLLoss()

#Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
	
#model.to(cuda_device) #This model.to would have worked if there was a GPU available on this macbook.

for ii, (inputs, labels) in enumerate(trainloader):
		
#	Move input and label tensors to the GPU
#	inputs, labels = inputs.to(cuda_device), labels.to(cuda_device)
		
	start = time.time()
		
	outputs = model.forward(inputs)
	loss = criterion(outputs, labels)
	loss.backward()
	optimizer.step()
		
	if ii == 3:
		break
			
print("Time per Batch: {:.3f} seconds".format(time.time() - start))
