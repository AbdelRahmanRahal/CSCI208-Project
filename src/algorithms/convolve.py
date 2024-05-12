import numpy as np

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	'''
	Time complexity: O(y * x * m * n)
	Space complexity: O(y * x)
	'''
	m, n = kernel.shape
	y, x = image.shape
	y = y - m + 1
	x = x - m + 1
	new_image = np.zeros((y,x))
	
	for i in range(y):
		for j in range(x):
			new_image[i][j] = np.sum(image[i:i+m, j:j+m] * kernel)
	
	return new_image