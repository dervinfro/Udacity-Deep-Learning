'''
Self tells Python where to look in the computer's memory for the shirt_one object.
'''

class Pants():
	def __init__(self, pants_color, pants_waist_size, pants_length, pants_price):
		self.color = pants_color
		self.waist_size = pants_waist_size
		self.length = pants_length
		self.price = pants_price
	def change_price(self, new_price):
		self.price = new_price
	def discount(self, discount):
		return self.price * (1 - discount)
		
		
class SalesPerson():
	
	def __init__(self, first_name, last_name, employee_id, salary):
		self.first_name = first_name
		self.last_name = last_name
		self.employee_id = employee_id
		self.salary = salary	
		self.pants_sold = []
		self.total_sales = 0
	
	def sell_pants(self, pants_object):
		self.pants_sold.append(pants_object)
		'''
		The above line is called pants_object for no other reason aside from the fact that it's reffing the Pants Class Object.
		It could be named anything....however, the naming convention of pants_object makes it easy to know what is being reffed.
		Also, if you look at the check_results() function (def() outside of a Class is called a Function), you'll notice that (ie: pants_one)is being passed to the sell_pants() method (def() inside a Class is called a Method) 
		'''
				
	def display_sales(self):
		for pants in self.pants_sold:
			print(pants.color, pants.waist_size, pants.length, pants.price)
		'''
		NOTE: Must print out not only the Class Object Name (ie: Pants) but also the Class attribute.  If you only print out the Class name (ie: print(pants)), the returned results will print out the memory location of the pants Class object.  If you print out the Class and the Attribute (see above: print(pants.color,....), this will print out the attribe contents.
		'''
			
	def calculate_sales(self):
		total = 0
		for pants in self.pants_sold:
			total += pants.price
			
		self.total_sales = total
		
		'''
		NOTE: I still need to look into the self in self.pants_sold.  Not quite sure why this is used in this manner.
		'''
		
		return total
		'''
		The following was what I used for the answer. It was wrong:
				for x in pants_sold:
				print(x)
		See the above method for the correct answer.
		I know the pseudocode on what I wanted it to say, I just didn't know the proper syntax to use.  
		A problem to be expected when I was drafting my first class.  
		Looking at my error and looking at the correct answer, I know understand where I went wrong.
		NOTE: I still need to look into the self in self.pants_sold.  Not quite sure why this is used in this manner.
		NOTE: Self tells Python where to look in the computer's memory for the shirt_one object.
		'''
	def calculate_commission(self, percentage):
		sales_total = self.calculate_sales()
		return sales_total * percentage
		'''
		The following was what I used for the answer. It was wrong:
			return self.total_commission == percentage * total_sales
		NOTE: I still need to look into the self in self.calculate_sales.  Not quite sure why this is used in this manner.
		NOTE: Self tells Python where to look in the computer's memory for the shirt_one object.

		
		'''
		

def check_results():
#	pants_one = Pants('red', 35, 36, 15.12)
#	pants_two = Pants('blue', 40, 38, 24.12)
#	pants_three = Pants('tan', 28, 30, 8.12)
#	
#	salesperson = SalesPerson('Amy', 'Gonzalez', 2581923, 40000)
#	
#	assert salesperson.first_name == 'Amy'
#	assert salesperson.last_name == 'Gonzalez'
#	assert salesperson.employee_id == 2581923
#	assert salesperson.salary == 40000
#	assert salesperson.pants_sold == []
#	assert salesperson.total_sales == 0
#	
#	salesperson.sell_pants(pants_one)
#	salesperson.pants_sold[0] == pants_one.color
#	
#	salesperson.sell_pants(pants_two)
#	salesperson.sell_pants(pants_three)
#	
#	assert len(salesperson.pants_sold) == 3
#	assert round(salesperson.calculate_sales(),2) == 47.36
#	assert round(salesperson.calculate_commission(.1),2) == 4.74
#	
#	print('Great job, you made it to the end of the code checks!')
	
	pants_one = Pants('red', 35, 36, 15.12)
	pants_two = Pants('blue', 40, 38, 24.12)
	pants_three = Pants('tan', 28, 30, 8.12)

	salesperson = SalesPerson('Amy', 'Gonzalez', 2581923, 40000)

	salesperson.sell_pants(pants_one)    
	salesperson.sell_pants(pants_two)
	salesperson.sell_pants(pants_three)

	salesperson.display_sales()

check_results()
	

'''
Write a SalesPerson class with the following characteristics:

the class name should be SalesPerson
the class attributes should include
first_name
last_name
employee_id
salary
pants_sold
total_sales
the class should have an init function that initializes all of the attributes
the class should have four methods
sell_pants() a method to change the price attribute
calculate_sales() a method to calculate the sales
display_sales() a method to print out all the pants sold with nice formatting
calculate_commission() a method to calculate the salesperson commission based on total sales and a percentage
'''