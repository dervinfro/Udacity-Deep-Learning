import pandas as pd
import numpy as np
import math
import time
import timeit

'''Wine quality datasets from the following URL: https://archive.ics.uci.edu/ml/datasets/wine+quality
	This dataset was used in the first problem from the Udacity AWS ML course.'''
#df = pd.read_csv('/Users/user/Downloads/winequality-red.csv', sep=';')
#df.columns = df.columns.str.replace(' ','_')
#print(df.head())
#'''
#The function below was the answer from the quiz.
#'''
#def numeric_to_buckets(df, column_name):
#	median = df[column_name].median()
#	for i, val in enumerate(df[column_name]):
#		if val >= median:
#			df.loc[i, column_name] = 'high'
#		else:
#			df.loc[i, column_name] = 'low' 
#
#for feature in df.columns[:-1]:
#	numeric_to_buckets(df, feature)
#	print(df.groupby(feature).quality.mean(), '\n')
'''
The following code is the Holiday Gift optimization problem.
See the URL (https://github.com/joshxinjie/Data_Scientist_Nanodegree/blob/master/Exercises/Term%202/2.%20Software%20Engineering%20Practices%20Pt%201/data/gift_costs.txt) for the text file that has the gift 
costs.
'''
with open('/Users/user/Downloads/gift_costs.txt') as f:
	gift_costs = f.read().split('\n')
	
gift_costs = np.array(gift_costs).astype(int)  # convert string to int
start = time.time()

#gc25 = gift_costs < 25
#print(gift_costs[gc25]) #https://numpy.org/doc/stable/user/basics.indexing.html#boolean-or-mask-index-arrays
#total_price = (gift_costs < 25).sum() 
total_price = (gift_costs[gift_costs < 25]).sum() * 1.08
print('Total Price: - $',  total_price)

	
#print('Column 11: ', new_columns[11],'\n') #how to ref a specific column using an integer
#print('DF ILOC: ', df.iloc[[4]]) #ref specific row using integer location using double brackets and the integer 
#print('DF LOC: \n', df.loc[[4]])
#print('DF ILOC specific field: \n', df.iloc[4,[3]]) #ref specific field (row X column) using single bracket (row) and internal bracket (column).
#print(df[['quality']]) #prints contents of row "quality"
#df_quality_sum = df[['quality']].sum() 
#df_quality_len = len(df[['quality']])
#df_quality_mean = df[['quality']].mean()
#print('Non-Math Import Mean: ', df_quality_sum / df_quality_len)
#print('Len: ', df_quality_len)
#print('Mean: ', df_quality_mean)
#
#
#print('DF Quality Sum:', df.quality.sum())
#print('DF Quality Mean: ', df.quality.mean())
#print('DF Quality Median: ', df.quality.median())
print('Duration {} seconds'.format(time.time() - start))