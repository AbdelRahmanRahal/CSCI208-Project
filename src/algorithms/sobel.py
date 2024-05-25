from typing import Union

import numpy as np
from PIL import Image

from .convolve import convolve


def sobel(image: Union[str, Image.Image]) -> Image.Image:
	"""
	Applies the Sobel edge detection algorithm to an input image.

	This function computes the gradient of the intensity of the image
	to find edges within the image. It uses the Sobel operator, which
	calculates the derivative of the image intensity at each pixel
	within a small region defined by the kernel. The result is an image
	highlighting the edges.

	Parameters:
	- image (Union[str, Image.Image]): Either a file path to an image (as a string) or a PIL `Image` object.
	  If a string is provided, the image will be loaded using PIL and converted to grayscale.

	Returns:
	- Image.Image: A PIL `Image` object representing the edges detected in the original image.

	Time complexity: O(y * x * m * n), where y and x are the dimensions of the input image,
	and m and n are the dimensions of the Sobel kernels.

	Space complexity: O(y * x), where y and x are the dimensions of the input image.

	Note: The Sobel operator is applied separately in the x and y directions, and the magnitudes of the gradients
	are combined to produce the final edge-detected image. The output image is normalized to the range 0-255.
	"""
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

	# Checking if the maximum value is zero to avoid division by zero
	max_value = edges.max()
	if max_value > 0:
		# Normalizing to 0-255
		edges = edges / max_value * 255

	return Image.fromarray(edges.astype(np.uint8))