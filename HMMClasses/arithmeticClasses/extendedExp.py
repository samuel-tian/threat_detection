import math

class EE:

	"""Custom class for extended logarithms and exponentials for HMM probability scaling

	Natural logarithm of 0 is defined as NaN
	exp(NaN) is defined as 0
	"""

	@staticmethod
	def eexp(x):
		"""Returns e^x, allows for x to be equal to NaN"""
		if (math.isnan(x)):
			return 0
		else:
			return math.exp(x)

	@staticmethod
	def eln(x):
		"""Returns ln(x), allows for x to be equal to 0"""
		if (x == 0):
			return float('nan')
		elif (x > 0):
			return math.log(x)

	@staticmethod
	def elnsum(x, y):
		"""Returns ln(a+b), with x=ln(a) and y=ln(b)"""
		if (math.isnan(x) or math.isnan(y)):
			if (math.isnan(x)):
				return y
			else:
				return x
		else:
			if (x > y):
				return x + EE.eln(1 + math.exp(y - x))
			else:
				return y + EE.eln(1 + math.exp(x - y))

	@staticmethod
	def elnproduct(x, y):
		"""Returns ln(a*b), where x=ln(a) and y=ln(b)"""
		if (math.isnan(x) or math.isnan(y)):
			return float('nan')
		else:
			return x + y
