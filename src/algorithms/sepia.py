from typing import Union

import numpy as np
from PIL import Image

from .convolve import convolve
from .gamma_correction import gamma_correction


def sepia(image: Union[str, Image.Image], gamma: float = 1) -> Image.Image:
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