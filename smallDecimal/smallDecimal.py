class Decimal:
	"""class to represent small decimals that cannot be represented with floating point numbers due to the limited precision of mantissas"""

	def adjust(self):
		"""
		fixes self to put val within the range [1, 10), and adjusts exponent correspondingly
		"""
		if (self.val == 0):
			self.exp = 0
		elif (self.val > 0):
			if (self.val < 1):
				while (self.val < 1):
					self.val *= 10
					self.exp -= 1
				return False
			elif (self.val >= 10):
				while (self.val >= 10):
					self.val /= 10
					self.exp += 1
				return False
		elif (self.val < 0):
			self.val = -self.val
			self.adjust()
			self.val = -self.val
			return False
		return True

	def __init__(self, val=0):
		self.val = val
		self.adjust()

	def __init__(self, val=0, exp=0):
		self.val = val
		self.exp = exp
		self.adjust()

	def __str__(self):
		return "{0} E {1}".format(self.val, self.exp)

	def __lt__(self, other):
		if self.val == 0:
			return True 
		elif other.val == 0:
			return False
		if self.exp == other.exp:
			return self.val < other.val
		else:
			return self.exp < other.exp

	def __le__(self, other):
		if self.val == 0:
			return True
		elif other.val == 0:
			return False
		if self.exp == other.exp:
			return self.val <= other.val
		else:
			return self.exp < other.exp

	def __gt__(self, other):
		return other <= self

	def __ge__(self, other):
		return other < self

	def __eq__(self, other):
		if (self.exp == other.exp):
			return self.val == other.val
		return False

	def __ne__(self, other):
		return not (self == other)

	def __add__(self, other):
		if (self < other):
			v = self.val
			for i in range(self.exp, other.exp):
				v /= 10
			ret = Decimal(v + other.val, other.exp)
			return ret
		else:
			v = other.val
			for i in range(other.exp, self.exp):
				v /= 10
			ret = Decimal(v + self.val, self.exp)
			return ret

	def __sub__(self, other):
		other.val = -other.val
		ret = self + other
		other.val = -other.val
		return ret

	def __mul__(self, other):
		ret = Decimal(self.val * other.val, self.exp + other.exp)
		return ret

	def __truediv__(self, other):
		ret = Decimal(self.val / other.val, self.exp - other.exp)
		return ret

	def __iadd__(self, other):
		self = self + other
		return self

	def __isub__(self, other):
		self = self - other
		return self

	def __imul__(self, div):
		self = self * div
		return self

	def __idiv__(self, div):
		self = self / div
		return self

	def __neg__(self):
		self.val = -self.val

if __name__ == "__main__":
	x = Decimal(1, 100)
	y = Decimal(2, 99)
	print(x)
	print(x + y)
	print(x - y)
	print(x * y)
	print(x / y)
	x += y
	print(x)
	x -= y
	print(x)
	x *= y
	print(x)
	x /= y
	print(x)
