from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import torch
import timeit
import torch.optim as optim
import torchvision.transforms as transforms
import requests
from torchvision import datasets, models


#using VGG19 model 

#get the features portion of the VGG19 (we will not need the classifier portion)
vgg = models.vgg19(pretrained=True).features

#freeze all VGG parameters since we are only optimizing the target image
for param in vgg.parameters():
	param.requires_grad_(False)
	
#move model to GPU if available.
process_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(process_device)
vgg.to(process_device)
#print(vgg)

#load in content and style images
def load_image(img_path, max_size=400, shape=None):
	'''
	Load in and transform an image.  Ensure that it is less than 400 pixels in x/y dimensions
	'''
	image = Image.open(img_path).convert('RGB')
	
	if max(image.size) > max_size:
		size = max_size
	else:
		size = max(image.size)
		
		
	if shape is not None:
		size = shape
		
	transform = transforms.Compose([
				transforms.Resize(size),
				transforms.ToTensor(),
				transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) 
				#Normalize(mean-RGB,std-RGB)..one value each for respective RGB channels 
				])	
				
	image = transform(image)[:3,:,:].unsqueeze(0) #research this 3 variable...not sure what this is
	
	return image
	
#load content and style image
content = load_image('/Users/user/Downloads/deep-learning-v2-pytorch-master/style-transfer/images/octopus.jpg').to(process_device)

#resize style to match content image...makes coding easier
style = load_image('/Users/user/Downloads/deep-learning-v2-pytorch-master/style-transfer/images/hockney.jpg', shape=content.shape[-2:]).to(process_device)

#helper function for un-normalizing an Image
#and converting it from a Tensor image to a Numpy Image
def im_convert(tensor):
	'''Display a Tensor as an image'''
	image = tensor.to("cpu").clone().detach()
	image = image.numpy().squeeze()
	image = image.transpose(1,2,0)
	image = image * np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5))
	image = image.clip(0,1)
	
	return image
	
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(style))
#plt.show()

def get_features(image, model, layers=None):
	'''
	Run an image forward through a model and get the features for a set of layers.
	Default layers are for VGGNet matching Gatys et al (2016)
	'''
	
	if layers is None:
		layers = {	'0' : 'conv1_1',	
					'5' : 'conv2_1',
					'10' : 'conv3_1',
					'19' : 'conv4_1',
					'21' : 'conv4_2', #content representation...see paper for details.
					'28' : 'conv5_1'}
	features = {}
	x = image
	for name, layer in model._modules.items():
		x = layer(x)
		if name in layers:
			features[layers[name]] = x
			
	return features

def gram_matrix(tensor):
	#get the batch size, height, depth, width of each tensor
	_, d, h, w = tensor.size()
	
	#reshape it so we are multiplying the features (inputs) for each channel
	tensor = tensor.view(d, h * w)
	
	#tensor.t = tensor transpose (SEE: Examples-Tensor Transpose.py)
	gram = torch.mm(tensor, tensor.t())
	
	return gram
	

	
#get the style and content features only once before forming the target image
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

#calculate the gram matrices for each layer of our style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

#calculate a third target image and prep it for change
#start off with the target as a copy of our content image and iteritavely change the style
target = content.clone().requires_grad_(True).to(process_device)

#weights for each style layer
#recommended to keep the individual weight values between 0 - 1
#heavy weights on earlier layers results in large stylistic effects
style_weights = {	'conv1_1' : 1.,	
					'conv2_1' : 0.8,
					'conv3_1' : 0.5,
					'conv4_1' : 0.3,
					'conv5_1' : 0.1}
					
content_weight = 1 #alpha
style_weight = 1e6 #beta	

#display the target image intermittently
show_every = 400

#iteration hyperparameters
optimizer = optim.Adam([target], lr =0.003)
steps = 2000 #how many steps to be used to update target image

for x in range(1, steps+1):
	
	#get the features from your target image
	target_features = get_features(target, vgg)
	
	#the content loss
	content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
	
	#the style loss
	style_loss = 0
	for layer in style_weights:
		target_feature = target_features[layer]
		target_gram = gram_matrix(target_feature)
		_, d, h, w = target_feature.shape
		#get the 'style' style representation
		style_gram = style_grams[layer]
		layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
		style_loss = layer_style_loss / (d * h * w)
		
	total_loss = content_weight * content_loss + style_weight + style_loss
	
	#update the target image
	optimizer.zero_grad()
	total_loss.backward()
	optimizer.step()
	print(x, '-', timeit.default_timer())
	
	if x % show_every == 0:
		print('Total Loss: ', total_loss.item())
		plt.imshow(im_convert(target))
		plt.show()
		
		
#display the content and the final Image
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 10))
ax1.imshow(im_convert(target))
ax2.imshow(im_convert(content))	
plt.show()