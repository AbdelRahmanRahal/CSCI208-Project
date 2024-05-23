import sys

from algorithms.gamma_correction import gamma_correction
from algorithms.sepia import sepia
from algorithms.sobel import sobel
from algorithms.adaptive_gamma_correction import adaptive_gamma_correction
from algorithms.histogram_equalization import histogram_equalization
from algorithms.mean_blur import mean_blur
from algorithms.gaussian_blur import gaussian_blur as gb

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QFileDialog
from PyQt6.QtGui import QPixmap
from PIL import Image
import numpy as np

class ImageProcessingApp(QWidget):
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.setWindowTitle('Image Processing App')
		self.setGeometry(100, 100, 800, 600)

		# Layouts
		mainLayout = QVBoxLayout()
		imageLayout = QHBoxLayout()

		# Widgets
		self.originalImageLabel = QLabel()
		self.processedImageLabel = QLabel()
		self.processButton = QPushButton('Apply Sobel Edge Detection')

		# Load a default image
		self.loadImage('src\\rose 300x300.jpeg')

		# Add widgets to layouts
		imageLayout.addWidget(self.originalImageLabel)
		imageLayout.addWidget(self.processedImageLabel)
		mainLayout.addLayout(imageLayout)
		mainLayout.addWidget(self.processButton)

		# Connect signals and slots
		self.processButton.clicked.connect(self.processImage)

		# Set the main layout
		self.setLayout(mainLayout)

	def loadImage(self, imagePath):
		pixmap = QPixmap(imagePath)
		self.originalImageLabel.setPixmap(pixmap.scaled(300, 300, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))
		self.processedImageLabel.setPixmap(pixmap.scaled(300, 300, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))

	def processImage(self):
		# Apply Gaussian blur to the image
		blurred_image = sobel('src\\rose 300x300.jpeg')
		blurred_image.save('blurred_image.jpg')

		# Update the processed image label
		pixmap = QPixmap('blurred_image.jpg')
		self.processedImageLabel.setPixmap(pixmap.scaled(300, 300, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))

def gaussian_blur(image_path, kernel_size=5, sigma=1):
	# Assuming the gaussian_blur function is defined as previously shown
	pass

if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = ImageProcessingApp()
	ex.show()
	sys.exit(app.exec())