from typing import Union

import numpy as np
from PIL import Image

from .convolve import convolve


def gaussian_blur(image: Union[str, Image.Image], kernel_size: int = 5, sigma: float = 1) -> Image.Image:
	"""
	Applies Gaussian blur to an input image.

	This function blurs the input image using a Gaussian kernel. The amount of blur
	is controlled by the standard deviation (`sigma`) of the Gaussian distribution
	that defines the kernel. The kernel size is automatically adjusted to be odd if
	an even size is provided.

	Parameters:
	- image (Union[str, Image.Image]): Either a file path to an image (as a string) or a PIL `Image` object.
	  If a string is provided, the image will be loaded using PIL.
	- kernel_size (int, optional): The size of the Gaussian kernel. Defaults to 5.
	- sigma (float, optional): The standard deviation of the Gaussian distribution. Controls the amount of blur. Defaults to 1.

	Returns:
	- Image.Image: A PIL `Image` object representing the blurred version of the input image.

	Time complexity: O(y * x * m * n), where y and x are the dimensions of the input image,
	and m and n are the dimensions of the Gaussian kernel.

	Space complexity: O(y * x), where y and x are the dimensions of the input image.

	Note: The Gaussian kernel is applied separately to each color channel (RGB) of the image.
	"""
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
	y = image_array.shape[0] - m + 1
	x = image_array.shape[1] - m + 1
	
	# Creating an output array with the new dimensions
	blurred_image = np.zeros((y, x, image_array.shape[2]))

	# Applying the kernel to each color channel (RGB) separately
	for i in range(3):
		blurred_image[:, :, i] = convolve(image_array[:, :, i], kernel)
	
	return Image.fromarray(blurred_image.astype(np.uint8))


def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
	"""
	Creates a 2D Gaussian kernel to be used for Gaussian blur.

	This function generates a 2D Gaussian kernel based on the provided size and standard deviation (`sigma`).
	The kernel is normalized so that the sum of all elements equals 1.

	Parameters:
	- size (int): The width and height of the square kernel.
	- sigma (float): The standard deviation of the Gaussian distribution.

	Returns:
	- np.ndarray: A 2D NumPy array representing the Gaussian kernel.
	"""
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