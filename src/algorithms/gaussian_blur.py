from algorithms.convolve import convolve

import numpy as np
from typing import Union
from PIL import Image


def gaussian_blur(image: Union[str, Image.Image], kernel_size: int = 5, sigma: float = 1) -> Image.Image:
	'''
	Time complexity: O(y * x * m * n)
	Space complexity: O(y * x)
	'''
	if isinstance(image, str): # If it's a string, then it's treated as a file path
		# Loading the image using PIL
		image = Image.open(image)
	
	image_array = np.array(image)
	
	# Ensuring the kernel size is odd
	if kernel_size % 2 == 0:
		kernel_size += 1
	
	# Creating a Gaussian kernel
	kernel = create_gaussian_kernel(kernel_size, sigma)
	m, n = kernel.shape
	
	# Calculating the new dimensions of the blurred image
	y = image_array.shape[0] - 2 * (m // 2)
	x = image_array.shape[1] - 2 * (m // 2)
	
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


def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
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