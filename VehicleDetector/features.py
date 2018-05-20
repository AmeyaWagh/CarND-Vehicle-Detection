import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog

class FeatureDetector(object):
	def __init__(self):
		self.color_space = cv2.COLOR_RGB2HSV
		self.orientations = 8
		self.pixels_per_cell = (12,12)
		self.cells_per_block = (2,2)
		self.image_size = (32,32)
		self.no_of_bins = 32

	def get_features(self,image):
		_image = np.copy(image)
		_image = self.convert_color_space(_image)
		color_hist = self.get_color_features(_image)
		spatial_hist = self.get_spatial_features(_image)
		hog_hist,img_hog = self.get_HOG(_image)
		features = np.concatenate((color_hist, spatial_hist, hog_hist))
		return features

	def convert_color_space(self,image):
		return cv2.cvtColor(image,self.color_space)

	def get_spatial_features(self,image):
		ch1_hist = np.histogram(image[:, :, 0], bins=self.no_of_bins)
		ch2_hist = np.histogram(image[:, :, 1], bins=self.no_of_bins)
		ch3_hist = np.histogram(image[:, :, 2], bins=self.no_of_bins)
		return np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))
	
	def get_color_features(self,image):
		ch1_featr = cv2.resize(image[:,:,0], self.image_size).ravel()
		ch2_featr = cv2.resize(image[:,:,1], self.image_size).ravel()
		ch3_featr = cv2.resize(image[:,:,2], self.image_size).ravel()
		return np.hstack((ch1_featr, ch2_featr, ch3_featr))

	def get_HOG(self,image):
		HOG_feature, img_hog = hog(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 
							orientations= self.orientations , 
							pixels_per_cell= self.pixels_per_cell , 
							cells_per_block= self.cells_per_block,
							visualise=True)
		return HOG_feature,img_hog

	def normalize_features(self):
		pass
