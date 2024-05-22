from algorithms.convolve import convolve

import numpy as np
from typing import Union
from PIL import Image

def mean_blur(image: Union[str, Image.Image], kernel_size: int = 5) -> Image.Image:
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
	y = image_array.shape[0] - 2 * (m // 2)
	x = image_array.shape[1] - 2 * (m // 2)
	
	# Creating an output array with the new dimensions
	blurred_image = np.zeros((y, x, image_array.shape[2]))

	# Applying the kernel to each color channel (RGB) separately
	for i in range(3):
		blurred_image[:, :, i] = convolve(image_array[:, :, i], kernel)

	return Image.fromarray(blurred_image.astype(np.uint8))