import numpy as np

#Y=[1,0,1,1] and P=[0.4,0.6,0.1,0.5] used in the Udacity example for Cross E
#y = [1,0,1,1]
#p = [0.4,0.6,0.1,0.5]
y = [1,1,0]
p = [0.8,0.7,0.9]
def ce(y,p):
	Y = np.float_(y)
	P = np.float_(p)

	return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

r = ce(y,p)
print(r)
'''
NOTE: 
	This is truely a summation of the ln(x).  If y is a 1, it will cancel out the (1 - Y) * np.log(1 - P).
	If y is a 0 , it will cancel out the Y * np.log(P).
	*** 0 times anything is 0 ***
'''


