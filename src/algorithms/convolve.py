import numpy as np

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	"""
	Applies a convolution operation on an input image using a given kernel.

	This function performs a 2D convolution between the input image and the
	kernel, resulting in a new image where each pixel value is computed as
	the sum of element-wise products between the kernel and the corresponding
	region of the image.

	Parameters:
	- image (np.ndarray): The input image represented as a 2D NumPy array.
	- kernel (np.ndarray): The convolution kernel represented as a 2D NumPy array.

	Returns:
	- np.ndarray: A new image resulting from the convolution operation, represented as a 2D NumPy array.

	Time complexity: O(y * x * m * n), where y and x are the dimensions of the output image,
	and m and n are the dimensions of the kernel.

	Space complexity: O(y * x), where y and x are the dimensions of the output image.

	Note: The output image size is smaller than the input image size due to the
	nature of the convolution operation. Specifically, the output image has
	dimensions (y-m+1, x-n+1), where y and x are the dimensions of the input
	image, and m and n are the dimensions of the kernel.
	"""
	# Rotating the kernel by 180 degrees
	kernel = np.flip(kernel)

	y, x = image.shape
	m, n = kernel.shape
	
	y = y - m + 1
	x = x - m + 1
	new_image = np.zeros((y,x))
	
	for i in range(y):
		for j in range(x):
			new_image[i][j] = np.sum(image[i:i+m, j:j+m] * kernel)
	
	return new_image