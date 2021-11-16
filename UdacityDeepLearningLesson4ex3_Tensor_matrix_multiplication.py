import torch

def activation(x):
	''' 
	Sigmoid activation function
	
	'''
	
	return 1/(1+torch.exp(-x))
	
torch.manual_seed(7)

features = torch.randn((1,5))
weights = torch.randn_like(features)
bias = torch.randn((1,1))

tor_mm = torch.mm(features, weights.view(5,1))
'''
NOTE: Remember that for matrix multiplications, the number of columns in the first tensor must equal to the number of rows in the second column.  If the above view function (weights.view(5,1)) is not added to the mm (or matul) function, the output will error as such:
		
			tor_mm = torch.mm(features, weights)
			RuntimeError: size mismatch, m1: [1 x 5], m2: [1 x 5] at ../aten/src/TH/generic/THTensorMath.cpp:41
	REMEMBER: THE NUMBER OF COLUMNS IN THE FIRST TENSOR MUST EQUAL TO THE NUMBER OF ROWS IN THE SECOND COLUMN.  This is done by weights.view(tensor matrix) in this case tensor matrix is 5,1.
'''
print('matrix multiplication: ', tor_mm)

sig_total_output = activation(torch.sum(features * weights) + bias)
print("Activation Function V2: ", sig_total_output )

sigmoid_act = activation(torch.mm(features, weights.view(5,1)) + bias)
print("Activation Function: ", sigmoid_act)



