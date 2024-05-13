from algorithms.convolve import convolve

import numpy as np
from PIL import Image

def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
	'''
	Time complexity: O(y * x * m * n)
	Space complexity: O(y * x)
	'''
	# Creating a 1D Gaussian kernel
	kernel_1d = [-(size // 2) + i for i in range(size)]

	# Applying the Gaussian function
	# the Guassian function:
	# f(x) = e^(-x^2 / (2 * sigma^2))
	for i in range(size):
		kernel_1d[i] = np.exp(-kernel_1d[i] ** 2 / (2 * sigma ** 2))
	
	# Normalizing the kernel
	kernel_1d /= np.sum(kernel_1d)
	
	# Creating a 2D Gaussian kernel
	kernel_2d = [[kernel_1d[i] * kernel_1d[j] for j in range(size)] for i in range(size)]
	return np.array(kernel_2d)


def gaussian_blur(image_path: str, kernel_size: int = 5, sigma: float = 1) -> Image.Image:
	# Loading the image using PIL
	image = Image.open(image_path)
	image_array = np.array(image)
	
	# Making sure the kernel size is odd
	if kernel_size % 2 == 0:
		kernel_size += 1
	
	# Creating a Gaussian kernel
	kernel = create_gaussian_kernel(kernel_size, sigma)
	m, n = kernel.shape
	
	# Calculating the new dimensions of the blurred image
	y = image_array.shape[0] - 2 * (m // 2)
	x = image_array.shape[1] - 2 * (m // 2)
	
	# Creating an output array with the new dimensions
	blurred_image = np.zeros((y, x, image_array.shape[2]))

	# Applying the kernel to each color channel (RGB) separately
	for i in range(3):
		blurred_image[:, :, i] = convolve(image_array[:, :, i], kernel)
	
	return Image.fromarray(blurred_image.astype(np.uint8))