import torch 
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/Users/user/Downloads/deep-learning-v2-pytorch-master/intro-to-pytorch/')
import helper
from torchvision import datasets, transforms

train_transform = transforms.Compose([transforms.Resize(255),
								transforms.CenterCrop(224),
								transforms.ToTensor()])
								
test_transform = transforms.Compose([transforms.Resize(255),
								transforms.CenterCrop(224),
								transforms.ToTensor()])

#Download and train the data.
#Don't touch this until the very end.	(https://elitedatascience.com/overfitting-in-machine-learning)							
trainset = datasets.ImageFolder('/Users/user/Downloads/Cat_Dog_data/train', transform=train_transform)
testset = datasets.ImageFolder('/Users/user/Downloads/Cat_Dog_data/test', transform=test_transform)
	
#Download and test the data.	
#Train and tune the model using this set.  (SEE ABOVE LINK)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)
'''
NOTE: .DataLoader above is a generator (). To get data out of it, it needs to be looped or iterated/next.
'''

images, labels = next(iter(testloader))
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
	ax=axes[ii]
	helper.imshow(images[ii], ax=ax,  normalize=False)
	
plt.show()
						
								