from typing import Union

import numpy as np
from PIL import Image


def gamma_correction(image: Union[str, Image.Image], gamma: float = 2.2) -> Image.Image:
	"""
	Applies gamma correction to an image.

	Parameters:
	- image (Union[str, Image.Image]): Either a file path to an image (as a string) or a PIL `Image` object.
	  If a string is provided, the image will be loaded using PIL.
	- gamma: The gamma value to apply. Default is 2.2, which is commonly used for screen displays.

	Returns:
	- Image.Image: A PIL `Image` object representing the gamma-corrected image.

	Time complexity: O(n), where n is the number of pixels in the image.

	Space comlexity: O(n), where n is the number of pixels in the image.
	"""
	if gamma <= 0:
		# Unexpected behaviour occurs at gamma <= 0
		raise ValueError("Gamma must be greater than 0.")

	if isinstance(image, str):  # If it's a string, then it's treated as a file path
		image = Image.open(image)
	
	if gamma == 1:
		# If gamma is 1, the image remains unchanged
		return image
	
	# Converting the image to RGB mode if it's not already
	if image.mode != 'RGB':
		image = image.convert('RGB')
	
	image_array = np.array(image)
	
	# Applying gamma correction
	# gamma correction formula:
	# P_c = (P_uc / P_max) ^ γ * P_max
	corrected_array = np.power(image_array / 255.0, gamma) * 255
	corrected_image = Image.fromarray(corrected_array.astype(np.uint8))
	
	return corrected_image