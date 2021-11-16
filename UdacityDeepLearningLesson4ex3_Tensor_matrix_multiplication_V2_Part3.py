from torch import nn

class Network(nn.Module):
	#inheriting from the nn.Module
	def __init__(self):
		super().__init__() #combined with the nn.Module, creates a class that tracks the architecture and provides a lot of useful methods and attributes.  Mandatory that we inherit nn.Module
		
		#Hidden object is created.  Inputs layer (784 units) to the hidden layer (256 units)
		self.hidden = nn.Linear(784, 256)
		'''
		NOTE: Once the network is created, we can access weights and bias with the following:
			net.hidden.weights OR net.hidden.bias
		'''
		
		#Output object is created.  Hidden layer (256 units) to the output layer (10 units)
		#the 10 units of the output layer correspond to the classification of the ten different numbers (0-9) in the MNIST dataset
		self.output = nn.Linear(256, 10)
		
		#Sigmoid object is created
		self.sigmoid = nn.Sigmoid()
		#Softmax object is created.  NOTE: dim = 1 (sum across columns)
		self.softmax = nn.Softmax(dim = 1)
		
	def forward(self, x):
		#Forward method is created to pass the input tensor through each of the operations
		'''
		NOTE: Pytorch networks created with the nn.Module MUST HAVE a forward method
		'''
		x = self.hidden(x)
		x = self.output(x)
		x = self.sigmoid(x)
		x = self.softmax(x)
		
		return x
		
model = Network()
print(model)