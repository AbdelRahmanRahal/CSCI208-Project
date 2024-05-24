from typing import Union

import numpy as np
from PIL import Image


def adaptive_gamma_correction(
	image: Union[str, Image.Image], block_size: int = 16, gamma_range: tuple[float, float] = (0.5, 2.0)
) -> Image.Image:
	"""
	Applies adaptive gamma correction to an input image.

	This function improves the visibility of details in both dark and bright
	regions of an image by adjusting the gamma value adaptively based on the
	local brightness of different blocks within the image.

	Parameters:
	- image (Union[str, Image.Image]): Either a file path to an image (as a string) or a PIL `Image` object.
	  If a string is provided, the image will be loaded using PIL.
	- block_size (int, optional): The size of the blocks over which gamma correction is applied. Defaults to 16.
	- gamma_range (tuple, optional): The minimum and maximum gamma values to use for correction. Defaults to (0.5, 2.0).

	Returns:
	- Image.Image: A PIL `Image` object representing the corrected version of the input image.

	Time complexity: O(n), where n is the number of pixels in the image.

	Space complexity: O(n), where n is the number of pixels in the image.

	Note: The gamma value for each block is calculated based on the average brightness of the block,
	allowing for adaptive enhancement across the image.
	"""
	if isinstance(image, str):  # If it's a string, then it's treated as a file path
		image = Image.open(image)
	
	image_array = np.array(image)
	height, width = image_array.shape[:2]

	# Processing each block
	for y in range(0, height, block_size):
		for x in range(0, width, block_size):
			block = image_array[y:y+block_size, x:x+block_size]
			avg_brightness = np.mean(block)
			gamma = calculate_gamma(avg_brightness, gamma_range)
			
			# Applying gamma correction
			# gamma correction formula:
			# P_c = (P_uc / P_max) ^ γ * P_max
			block_corrected = np.power(block / 255.0, gamma) * 255
			image_array[y:y+block_size, x:x+block_size] = block_corrected.astype(np.uint8)

	return Image.fromarray(image_array)


def calculate_gamma(avg_brightness: np.floating, gamma_range: tuple) -> float:
	'''
	Function to calculate gamma value based on average brightness
	'''
	return gamma_range[0] + (gamma_range[1] - gamma_range[0]) * avg_brightness / 255