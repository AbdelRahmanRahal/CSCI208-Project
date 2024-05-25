from typing import Union

import numpy as np
from PIL import Image

from .convolve import convolve


def mean_blur(image: Union[str, Image.Image], kernel_size: int = 5) -> Image.Image:
	"""
	Applies mean blur to an input image.

	Mean blur, also known as box blur, is a simple image filtering operation
	that replaces each pixel's value with the average value of its neighbors,
	including itself. This process reduces image noise and detail.

	Parameters:
	- image (Union[str, Image.Image]): Either a file path to an image (as a string) or a PIL `Image` object.
	  If a string is provided, the image will be loaded using PIL.
	- kernel_size (int, optional): The size of the square kernel used for blurring. Defaults to 5.

	Returns:
	- Image.Image: A PIL `Image` object representing the blurred version of the input image.

	Time complexity: O(y * x * m * n), where y and x are the dimensions of the input image,
	and m and n are the dimensions of the kernel.

	Space complexity: O(y * x), where y and x are the dimensions of the input image.

	Note: The kernel is normalized so that the sum of all its elements equals 1,
	ensuring that the overall brightness of the image remains unchanged after
	the blur is applied. The function processes each color channel (RGB) separately
	to maintain color integrity.
	"""
	if isinstance(image, str): # If it's a string, then it's treated as a file path
		# Loading the image using PIL
		image = Image.open(image)
	
	image_array = np.array(image)

	# Ensuring kernel size is odd
	if kernel_size % 2 == 0:
		kernel_size += 1

	# Creating a kernel filled with ones
	kernel = [[1 for _ in range(kernel_size)] for _ in range(kernel_size)]
	kernel = np.array(kernel, dtype=np.float32) / (kernel_size * kernel_size)
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