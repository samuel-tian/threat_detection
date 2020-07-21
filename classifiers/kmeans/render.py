import matplotlib.pyplot as plt
from random import random
import re

def str_to_float(in_str):
	ret = re.split(", ", in_str)
	for i in range(len(ret)):
		ret[i] = float(ret[i])
	return ret

def render(file_name):
	name = file_name + ".dat"
	with open(name, 'r') as fin:
		n = int(fin.readline().strip())
		for i in range(n):
			centroid = str_to_float(fin.readline().strip()[1:-1])
			rgb = (random(), random(), random())
			t = int(fin.readline().strip())
			for j in range(t):
				path = str_to_float(fin.readline().strip()[1:-1])
				plt.plot(path[0], path[1], marker="o", color=rgb)
			plt.plot(centroid[0], centroid[1], marker="^", color="black")
	plt.axis([0, 20, 0, 20])
	plt.show()

if __name__ == "__main__":
	render("clustered")