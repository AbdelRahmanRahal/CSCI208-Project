import numpy as np
from typing import Union
from PIL import Image


def histogram_equalization(image: Union[str, Image.Image]) -> Image.Image:
	if isinstance(image, str): # If it's a string, then it's treated as a file path
		# Loading the image using PIL
		image = Image.open(image).convert('L')
	
	image_array = np.array(image)

	# Calculating histogram
	hist = [0] * 256
	for pixel in image_array.flatten():
		hist[pixel] += 1

	# Calculating cumulative sum
	cdf = [0] * 256
	cdf[0] = hist[0]
	for i in range(1, len(hist)):
		cdf[i] = cdf[i-1] + hist[i]

	# Normalizing the cdf
	cdf_min = min(cdf)
	cdf_max = max(cdf)
	cdf_normalized = [(value - cdf_min) * 255 / (cdf_max - cdf_min) for value in cdf]

	# Using cdf_normalized as a lookup table to equalize the image
	flattened_enhanced_image = [cdf_normalized[pixel] for pixel in image_array.flatten()]
	enhanced_image = np.array(flattened_enhanced_image, dtype='uint8').reshape(image_array.shape)
	
	return Image.fromarray(enhanced_image)