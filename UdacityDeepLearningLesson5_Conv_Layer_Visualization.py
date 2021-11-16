import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#######################
### VISUAL THE IMAGE ##
#######################
img_path = '/Users/user/Desktop/eyes.png' #import file img_path

bgr_img = cv2.imread(img_path) #load blue-green-red image read
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY) #convert to grayscale image

gray_img = gray_img.astype('float32')/255 #normalize, rescale entries to line in [0,1]

#plt.imshow(gray_img, cmap='gray')
##plt.show()
#
filter_vals = np.array([[-1,-1,1,1], 
						[-1,-1,1,1], 
						[-1,-1,1,1], 
						[-1,-1,1,1]])

#print('Filter shape: ', filter_vals.shape) #Filter shape:  (4, 4)

#define four different filters
#all of which are linear combinations fo the 'filter_vals' defined above

#define four filters
filter_1 = filter_vals #copy of filter vals
filter_2 = -filter_1 #the minus sign in front of filter_1, must mirror the array of filter_vals
filter_3 = filter_1.T #the .T must make a traverse of filter_1
filter_4 = -filter_3 #the minus sign in front of filter_3, must mirror the array of filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])
#
##print('Filter 1: ', filter_1.T)
#
###############################
###VISUALIZE ALL FOUR FILTERS##
###############################
##visualize all four filters
#fig = plt.figure(figsize=(10,5)) #figsize is 10 inches by 5 inches
#for i in range(4):
#	#xticks and yticks are blank due to the lack of need for x & y values in this filter array
#	ax = fig.add_subplot(1,4, i+1, xticks=[], yticks=[]) #not sure what xticks and yticks are.....
#	ax.imshow(filters[i], cmap='gray')
#	ax.set_title('Filter {}'.format(i+1))
#	width, height = filters[i].shape
#	for x in range(width):
#		for y in range(height):
#			ax.annotate(str(filters[i][x][y]), xy=(y,x), #not sure what annotate and some of the parameters mean.....
#						horizontalalignment='center',
#						verticalalignment='center',
#						color='white' if filters[i][x][y]<0 else 'black')

class Net(nn.Module):
	
	def __init__(self, weight):
		super(Net, self).__init__()
		#super(), combined with nn.Module, creates a class that track the architecure and provides useful methods and attributes.
		#it's mandatory to inheret the nn.Module when using super()
		#initializes the weights of the convolutional layer to be the weights fo the 4 defined layers.
		k_height, k_width = weight.shape[2:]
		#assumes there are four (4) grayscale filters
		self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
		self.conv.weight = torch.nn.Parameter(weight)
	
	def forward(self, x):
		#calculates the output of the convd layer
		#pre / post activation
		conv_x = self.conv(x)
		activated_x = F.relu(conv_x)
		
		#returns both layers
		return conv_x, activated_x
		
#instantiate the model and set the weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model  = Net(weight)
#print(model)

def viz_layer(layer, n_filter=4):
	fig = plt.figure(figsize=(8,8)) #plot a 8 * 8 inch plot
	
	for i in range(n_filter):
		ax = fig.add_subplot(1, n_filter, i+1, xticks=[], yticks=[])
		ax.imshow(np.squeeze(layer[0,i].data.numpy()), cmap='gray')
		ax.set_title('Output: {}'.format(str(i+1)))
			
			
plt.imshow(gray_img, cmap='gray') #plot the original image

#visualize all four (4) filters
fig = plt.figure(figsize=(8,6))
#fig.subplots_adjust(left=0, right=1.5, bottom=0.8, top=1, hspace=0.05, wspace=0.05)
for i in range(4):
	ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
	ax.imshow(filters[i], cmap='gray')
	ax.set_title('Filter: {}'.format(str(i+1)))
	
gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1) #research unsqueeze...don't know exactly what this does

#get the convd layer (pre/post activation)
conv_layer, activation_layer = model(gray_img_tensor)
viz_layer(conv_layer)
viz_layer(activation_layer)
plt.show()


