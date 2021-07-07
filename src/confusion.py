import numpy as np
import matplotlib.pyplot as plt
import glob, os


if __name__ == '__main__':

	#file = glob.glob('sensor_abstract.txt')
	#print(file)

	fileDir = os.path.dirname(os.path.realpath('__file__'))

	#filename = "sensor_abstract.txt"
	#readFile(filename)

	#For accessing the file in a folder contained in the current folder
	filename = os.path.join(fileDir, 'sensor_abstract')
	#readFile(filename)

	file = open(filename, 'r')
	lines = file.readlines()

	#crop = np.loadtxt(glob.glob('sensor_abstract.txt')).astype(int)
	#weed = np.loadtxt(glob.glob('sensor_abstract.txt')).astype(int)

	res = np.zeros((14, 13))
	row = 0
	column = 0


	for line in lines:
		fline = line.split("\t")
		for v in fline:
			res[row][column] = v
			column += 1
		column = 0
		row += 1

		print("file \n")
		print(str(fline))


	plt.matshow(res)
	plt.colorbar()
	plt.xlabel('True number of weeds')
	plt.ylabel('Detected number of weeds')


	plt.show()


