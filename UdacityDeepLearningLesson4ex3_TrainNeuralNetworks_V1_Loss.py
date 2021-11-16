import torch.nn.functional as F
import torch
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

model = nn.Sequential(nn.Linear(784, 128),
						nn.ReLU(),
						nn.Linear(128,64),
						nn.ReLU(),
						nn.Linear(64, 10),
						nn.LogSoftmax(dim = 1))
						
#define the loss
criterion = nn.NLLLoss()

#retrieve the data
images, labels = next(iter(trainloader))

#flatten the image
#image.shape[0] is a shortcut for batch size (64 in this case).
#SEE torchTensorViewExperiments.py as an example on .shape
#The -1 is to flatten the image. 
#-1 is a shortcut to the input units.  It's used in replace of knowing the exact input unit value
images = images.view(images.shape[0], -1)

#logit layer - the output of the fully connected layer and produce raw prediction values.
#in the case of MNIST, the output will be a tensor of 10 values, where each value represents a score of each class (0-9)
#the final output of this tensor will be the following shape: [batch_size, 10]....[64, 10]
#in the end, get the final value which represents the probability of each target class.  This is done so by 
#applying the softmax activation to the output of the logits layer (torch.nn.LogSoftmax(logits))
logits = model(images)

loss = criterion(logits, labels)
print(loss)

		