from math import e
import numpy as np
num_list = [0,1,2,3]
new_num_list = []
new_numlist = []
result_list = []

'''
NOTE: In my SoftMaxFunc function, I over programmed the same results that were produced in the Udacity softmax function.  I could have used the np.exp to input the list/array of integers.  However, I used the e constant in the For loop.  Then....appended a new number list, then created a new For loop to produce the percentages of new number list floats divided by the sum of the new number list floats.  Same results as the more efficient softmax function but numerous more lines.
'''
def SoftMaxFunc(L):
	'''
	The above Math e print line is just for me to understand the results of the e constant.
	'''
	for y in L:
		expOutput = e ** y
		new_num_list.append(expOutput)
	sumList = sum(new_num_list)
	for z in new_num_list:
		print(z/sumList)
		
def softmaxfunc_v2(L):
	for x in L:
		new_numlist.append(e ** x)

	for x in L:
		result_list.append((e ** x) / sum(new_numlist))
			
	return result_list
	
		
def softmax(L):
	expL = np.exp(L)
	sumExpL = sum(expL)
	result = []
	for i in expL:
		result.append(i*1.0/sumExpL)
#	print(result)
	return result	
	#A return statement is requried if the output of the function is to be used/displayed outside of the function.
	#NOTE: Run the above subtract function with and without the return statement.
	#With return = the output is printed
	#Without return = the output is None	

SoftMaxFunc(num_list)

print('*' * 20 )
print('r: ', softmax(num_list))

print('*' * 20 )
rV2 = softmaxfunc_v2(num_list)
print('rV2: ', rV2)
print('SUM rV2: ', sum(rV2))


