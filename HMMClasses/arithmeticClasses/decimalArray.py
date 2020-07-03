import numpy as np
import sys, os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from arithmeticClasses.smallDecimal import Decimal

class DecimalArray:

	def __getitem__(self, key):
		return self.arr[key]

	def __setitem__(self, key, val):
		self.arr[key] = val

	@staticmethod
	def recursive_fill(nparr, val=0):
		if (len(nparr.shape)==1):
			for i in range(len(nparr)):
				nparr[i] = Decimal(val)
		else:
			for i in range(len(nparr)):
				DecimalArray.recursive_fill(nparr[i], val)

	@staticmethod
	def recursive_set(nparr, initial):
		if (len(nparr.shape)==1):
			for i in range(len(nparr)):
				nparr[i] = Decimal(initial[i])
		else:
			for i in range(len(nparr)):
				DecimalArray.recursive_set(nparr[i], initial[i])

	@staticmethod
	def recursive_display(nparr):
		ret = ""
		if (len(nparr.shape) == 1):
			ret += "["
			for i in range(nparr.shape[0]):
				ret += str(nparr[i])
				if (i != len(nparr)-1):
					ret += ", "
			ret += "]"
		else:
			ret += "["
			for i in range(len(nparr)):
				if (i != 0):
					ret += " "
				ret += DecimalArray.recursive_display(nparr[i])
				if (i != len(nparr)-1):
					ret += "\n"
			ret += "]"
		return ret

	def __init__(self, *args):
		if (len(args)==0):
			self.arr = np.ndarray(dtype=np.object, shape=(0))
			DecimalArray.recursive_fill(self.arr)
		else:
			arg = args[0]
			if (isinstance(arg, tuple)):
				self.arr = np.ndarray(dtype=np.object, shape=arg)
				DecimalArray.recursive_fill(self.arr)
			elif (isinstance(arg, int)):
				self.arr = np.ndarray(dtype=np.object, shape=(arg))
				DecimalArray.recursive_fill(self.arr)
			else:
				if (isinstance(arg, list)):
					arg = np.array(arg)	
				self.arr = np.ndarray(dtype=np.object, shape=arg.shape)
				DecimalArray.recursive_set(self.arr, arg)

	def __str__(self):
		return DecimalArray.recursive_display(self.arr)

if __name__ == "__main__":
	x = DecimalArray([[2,2],[2,2]])
	print(x[0])
	print(x[0][0])
	print(x)
