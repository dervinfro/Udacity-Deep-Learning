import math
import matplotlib.pyplot as plt
import statistics

class Gaussian():
	""" Gaussian distribution class for calculating and 
	visualizing a Gaussian distribution.
	
	Attributes:
		mean (float) representing the mean value of the distribution
		stdev (float) representing the standard deviation of the distribution
		data_list (list of floats) a list of floats extracted from the data file
			
	"""
	def __init__(self, mu = 0, sigma = 1):
		
		self.mean = mu
		self.stdev = sigma
		self.data = []


	
	def calculate_mean(self):
	
		"""Method to calculate the mean of the data set.
		
		Args: 
			None
		
		Returns: 
			float: mean of the data set
	
		"""
		
		#TODO: Calculate the mean of the data set. Remember that the data set is stored in self.data
		# Change the value of the mean attribute to be the mean of the data set
		# Return the mean of the data set           
#		pass
		avg = sum(self.data) / len(self.data)
		self.mean = avg
		return self.mean
				


	def calculate_stdev(self, sample=True):

		"""Method to calculate the standard deviation of the data set.
		
		Args: 
			sample (bool): whether the data represents a sample or population
		
		Returns: 
			float: standard deviation of the data set
	
		"""

		# TODO:
		#   Calculate the standard deviation of the data set
		#   
		#   The sample variable determines if the data set contains a sample or a population
		#   If sample = True, this means the data is a sample. 
		#   Keep the value of sample in mind for calculating the standard deviation
		#
		#   Make sure to update self.stdev and return the standard deviation as well    
			
#		pass
		if sample:
			n = len(self.data) - 1
		else:
			n = len(self.data)
		mean = self.mean
		sigma = 0 
		
		'''
		NOTE: The formula for Standard Deviation comes from: https://www.mathsisfun.com/data/standard-deviation-formulas.html
		'''
		for d in self.data:
			sigma += (d - mean) ** 2
#			x - mean = x_minus_mean
			'''
			NOTE: The line above (#75) did not work.  This was my original code.  
					1 - it did not loop through the for statement and put all of the results on one variable (sigma).
					2 - it was inside the for loop, when it should have been outside the loop.  This would have altered the stdev end value.  
#			Since line 75 did not work, there is no need for line 80 below
#			'''
#			x_minus_mean ** 2 = x_minus_mean_squared
		sigma = math.sqrt(sigma / n)
#			sum(sigma) = x_sumed
#			self.stdev = math.sqrt(x_sumed)
		self.stdev = sigma
		
		return self.stdev

	def read_data_file(self, file_name, sample=True):
	
		"""Method to read in data from a txt file. The txt file should have
		one number (float) per line. The numbers are stored in the data attribute. 
		After reading in the file, the mean and standard deviation are calculated
				
		Args:
			file_name (string): name of a file to read from
		
		Returns:
			None
		
		"""
		
		# This code opens a data file and appends the data to a list called data_list
		with open(file_name) as file:
			data_list = []
			line = file.readline()
			while line:
				data_list.append(int(line))
				line = file.readline()
		file.close()
	
		# TODO: 
		#   Update the self.data attribute with the data_list
		#   Update self.mean with the mean of the data_list. 
		#       You can use the calculate_mean() method with self.calculate_mean()
		#   Update self.stdev with the standard deviation of the data_list. Use the 
		#       calcaulte_stdev() method.
		self.data = data_list
		self.mean = self.calculate_mean()
		self.stdev = self.calculate_stdev(sample)
		'''
		NOTE: In the above lines, I used the statistics Class (which I imported) to find the mean and stdev.  I did not use the recommended internal classes for this method.   TBD on the outcome.
		UPDATE: The statistic class does work as a function; however, it does not work in this sense due to the fact that the calculate_mean and calculate_stdev need to be called for the test to execute properly.
		'''
				
		
	def plot_histogram(self):
		"""Method to output a histogram of the instance variable data using 
		matplotlib pyplot library.
		
		Args:
			None
			
		Returns:
			None
		"""
		
		# TODO: Plot a histogram of the data_list using the matplotlib package.
		#       Be sure to label the x and y axes and also give the chart a title
		plt.plot(self.data)
		plt.title('test')
		plt.xlabel('x label')
		plt.ylabel('y label')
				
		
	def pdf(self, x):
		"""Probability density function calculator for the gaussian distribution.
		
		Args:
			x (float): point for calculating the probability density function
			
		
		Returns:
			float: probability density function output
		"""
		
		# TODO: Calculate the probability density function of the Gaussian distribution
		#       at the value x. You'll need to use self.stdev and self.mean to do the calculation
#		pass  
		return (1.0 / (self.stdev * math.sqrt(2*math.pi))) * math.exp(-0.5*((x - self.mean) / self.stdev) ** 2)
		'''
		NOTE: I'll have to find a resource that explains the PDF calculation.  Possibly the previous math lesson on Normal Distribution.  Until that time, I'll poach the answer from answer repository.
		UPDATE: See https://www.math24.net/probability-density-function/ and look at the section of Normal Distribution for an explanation of the PDF equation.
		'''      

	def plot_histogram_pdf(self, n_spaces = 50):

		"""Method to plot the normalized histogram of the data and a plot of the 
		probability density function along the same range
		
		Args:
			n_spaces (int): number of data points 
		
		Returns:
			list: x values for the pdf plot
			list: y values for the pdf plot
			
		"""
		
		#TODO: Nothing to do for this method. Try it out and see how it works.
		
		mu = self.mean
		sigma = self.stdev

		min_range = min(self.data)
		max_range = max(self.data)
		
		 # calculates the interval between x values
		interval = 1.0 * (max_range - min_range) / n_spaces

		x = []
		y = []
		
		# calculate the x values to visualize
		for i in range(n_spaces):
			tmp = min_range + interval*i
			x.append(tmp)
			y.append(self.pdf(tmp))

		# make the plots
		fig, axes = plt.subplots(2,sharex=True)
		fig.subplots_adjust(hspace=.5)
		axes[0].hist(self.data, density=True)
		axes[0].set_title('Normed Histogram of Data')
		axes[0].set_ylabel('Density')

		axes[1].plot(x, y)
		axes[1].set_title('Normal Distribution for \n Sample Mean and Sample Standard Deviation')
		axes[0].set_ylabel('Density')
		plt.show()

		return x, y
		
	def __add__(self, other):
		result = Gaussian()
		result.mean = self.mean + other.mean
		result.stdev = math.sqrt(self.stdev ** 2 + other.stdev ** 2)
		return result
		
		"""Magic method to add together two Gaussian distributions
		
		Args:
			other (Gaussian): Gaussian instance
			
		Returns:
			Gaussian: Gaussian distribution
			
		"""
		
		# TODO: Calculate the results of summing two Gaussian distributions
		#   When summing two Gaussian distributions, the mean value is the sum
		#       of the means of each Gaussian.
		#
		#   When summing two Gaussian distributions, the standard deviation is the
		#       square root of the sum of square ie sqrt(stdev_one ^ 2 + stdev_two ^ 2)
		
		# create a new Gaussian object

	def __repr__(self):
	
		return 'mean {}, standard deviation {}'.format(self.mean, self.stdev)
		
		"""Magic method to output the characteristics of the Gaussian instance
		
		Args:
			None
		
		Returns:
			string: characteristics of the Gaussian
		
		"""
		
		# TODO: Return a string in the following format - 
		# "mean mean_value, standard deviation standard_deviation_value"
		# where mean_value is the mean of the Gaussian distribution
		# and standard_deviation_value is the standard deviation of
		# the Gaussian.
		# For example "mean 3.5, standard deviation 1.3"
		
#			pass



import unittest

class TestGaussianClass(unittest.TestCase):
	def setUp(self):
		self.gaussian = Gaussian(25, 2)

	def test_initialization(self): 
		self.assertEqual(self.gaussian.mean, 25, 'incorrect mean')
		self.assertEqual(self.gaussian.stdev, 2, 'incorrect standard deviation')

	def test_pdf(self):
		self.assertEqual(round(self.gaussian.pdf(25), 5), 0.19947,\
		 'pdf function does not give expected result') 

	def test_meancalculation(self):
		self.gaussian.read_data_file('/Users/user/Downloads/numbers.txt', True)
		self.assertEqual(self.gaussian.calculate_mean(),\
		 sum(self.gaussian.data) / float(len(self.gaussian.data)), 'calculated mean not as expected')

	def test_stdevcalculation(self):
		self.gaussian.read_data_file('/Users/user/Downloads/numbers.txt', True)
		self.assertEqual(round(self.gaussian.stdev, 2), 92.87, 'sample standard deviation incorrect')
		self.gaussian.read_data_file('/Users/user/Downloads/numbers.txt', False)
		self.assertEqual(round(self.gaussian.stdev, 2), 88.55, 'population standard deviation incorrect')
	def test_add(self):
		gaussian_one = Gaussian(25, 3)
		gaussian_two = Gaussian(30, 4)
		gaussian_sum = gaussian_one + gaussian_two
		
		self.assertEqual(gaussian_sum.mean, 55)
		self.assertEqual(gaussian_sum.stdev, 5)

	def test_repr(self):
		gaussian_one = Gaussian(25, 3)
			
		self.assertEqual(str(gaussian_one), "mean 25, standard deviation 3")
						

tests = TestGaussianClass()

tests_loaded = unittest.TestLoader().loadTestsFromModule(tests)

unittest.TextTestRunner().run(tests_loaded)