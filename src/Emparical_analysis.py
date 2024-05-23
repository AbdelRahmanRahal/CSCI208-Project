import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


algorithms = {
    "Mean Blur": mean_blur,
    "Gaussian Blur": gaussian_blur,
    "Sobel Edge Detection": sobel,
    "Histogram Equalization": histogram_equalization,
    "Adaptive Gamma Correction": adaptive_gamma_correction,
    "Gamma Correction": gamma_correction
}

# Function to generate images of varying sizes
def generate_images(sizes):
    images = {}
    for size in sizes:
        img = Image.new('RGB', (size, size), color=(73, 109, 137))
        img.save(f"temp_{size}.jpg")
        images[size] = img
    return images

# Generate images of varying sizes
sizes = [100, 200, 300, 400, 500]  # i make it hard coded you can change it 
images = generate_images(sizes)

# Measure execution times for each algorithm and image size
execution_times = {}
for name, func in algorithms.items():
    times = []
    for size, img in images.items():
        start_time = time.time()
        result = func(img)
        end_time = time.time()
        times.append(end_time - start_time)
    execution_times[name] = times


fig, ax = plt.subplots()
for name, times in execution_times.items():
    ax.plot(sizes, times, label=name)

ax.set_xlabel('Image Size')
ax.set_ylabel('Execution Time (seconds)')
ax.set_title('Execution Time vs. Image Size')
ax.legend()
plt.xticks(sizes)
plt.show()
