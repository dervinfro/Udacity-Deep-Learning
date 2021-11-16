import torch
import numpy as np

torch.manual_seed(7)

features = torch.randn((1,3))
#features AKA inputs for the neural network

n_input = features.shape[1]
n_hidden = 2
n_output = 1

W1 = torch.randn(n_input, n_hidden)
#weights for the inputs to the hidden layer
B1 = torch.randn((1, n_hidden))

W2 = torch.randn(n_hidden, n_output)
#weights for the hidden to the output layer
B2 = torch.randn((1, n_output))

def activation(x):
	''' 
	Sigmoid activation function
	
	'''
	return 1/(1+torch.exp(-x))
	

layer_one = activation(torch.mm(features, W1) + B1)
print("Layer One: ", layer_one)
total_output = activation(torch.mm(layer_one, W2) + B2)
print("Sigmoid Activation Function Output: ", total_output)


'''
BONUS:
	See below for how to bridge a numpy array over to a pytorch tensor.
'''

a = np.random.rand(4,3,2) #establish a random numpy array
b = torch.from_numpy(a) #bridge numpy array over to pytorch tensor
c = b.numpy() #bridge pytorch tensor BACK to a numpy array
#d = b.mul_(2) #multiply pytorch tensor by variable

print("Numpy Array: ", a,
		"\nPytorch Tensor: ", b,
		"\nNumpy Array: ", c,)
#		"\nTensor Prodcut: ", d)

