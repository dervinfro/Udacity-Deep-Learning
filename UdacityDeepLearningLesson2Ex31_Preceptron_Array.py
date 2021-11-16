import numpy as np
'''
 NOTE: See the Intro to Neural Networks / Lesson 2 / Exercise 31 - Question 1.
The following code writes out the perceptron portion of the question.  IT DOES NOT address the sigmoid function.  See my SigmoidFunctionExample.py for running data through that activation function.
'''

probability_array = np.array([0.4,0.6,0.9])
weight_array = np.array([2,6,3])



def preceptron(x,y,b):
	
	
	for z in range(0, len(probability_array)):
		np_prodcut_result = (np.multiply(x[z], y[z])) + b
		print(np_prodcut_result)
		

preceptron(probability_array, weight_array, 6.0)
	
	

