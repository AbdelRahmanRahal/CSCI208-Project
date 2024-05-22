from algorithms.convolve import convolve

import numpy as np
from typing import Union
from PIL import Image

def sobel(image: Union[str, Image.Image]) -> Image.Image:
	'''
	Time complexity: O(y * x * m * n)
	Space complexity: O(y * x)
	'''
	if isinstance(image, str): # If it's a string, then it's treated as a file path
		# Loading the image using PIL
		image = Image.open(image).convert('L')

	image_array = np.array(image)

	# Sobel operators
	sobel_x = np.array([
		[-1, 0, 1],
		[-2, 0, 2],
		[-1, 0, 1]
	])
	sobel_y = np.array([
		[-1, -2, -1],
		[ 0,  0,  0],
		[ 1,  2,  1]
	])

	# Applying Sobel operators
	edges_x = convolve(image_array, sobel_x)
	edges_y = convolve(image_array, sobel_y)

	# Combining the results
	# Magnitude of the gradient:
	# ||G|| = sqrt(Gx^2 + Gy^2)
	edges = np.sqrt(edges_x ** 2 + edges_y ** 2)

	# Normalize to 0-255
	edges = edges / edges.max() * 255

	return Image.fromarray(edges.astype(np.uint8))