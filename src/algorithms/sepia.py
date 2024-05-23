from typing import Union

import numpy as np
from PIL import Image

from .gamma_correction import gamma_correction


def sepia(image: Union[str, Image.Image], gamma: float = 1) -> Image.Image:
	'''
	Applies a sepia filter to an input image, enhancing it with a warm brownish tone typical of old photographs.

	The sepia effect is achieved by transforming the color channels of the image according to a predefined matrix,
	followed by clipping the values to ensure they fall within the valid range for displayable colors (0-255).
	Before applying the sepia filter, the image undergoes gamma correction to adjust its brightness and contrast.

	Parameters:
	- image (Union[str, Image.Image]): Either a file path to an image (as a string) or a PIL `Image` object.
	  If a string is provided, the image will be loaded using PIL.
	- gamma (float, optional): The gamma value to apply for gamma correction. Defaults to 1, which means no gamma correction is applied.

	Returns:
	- Image.Image: A PIL `Image` object representing the sepia-enhanced version of the input image.

	Time complexity: O(n), where n is the number of pixels in the image.
	Space complexity: O(n), where n is the number of pixels in the image.
	'''
	image = gamma_correction(image, gamma)
	
	image_array = np.array(image)

	# Sepia filter matrix
	sepia = np.array([
		[0.393, 0.769, 0.189],
		[0.349, 0.686, 0.168],
		[0.272, 0.534, 0.131]
	])
	
	# Applying the sepia filter
	sepia_image_array = np.dot(image_array[..., :3], sepia.T)
	
	# Clipping values to the valid range (0-255)
	sepia_image_array = np.clip(sepia_image_array, 0, 255)

	return Image.fromarray(sepia_image_array.astype(np.uint8))