from VehicleDetector import pipeline
from VehicleDetector import data_handler
from VehicleDetector import features
from VehicleDetector import visualizer
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = '.'

vehicle_detector = pipeline.VehicleDetector(BASE_PATH = BASE_PATH)
data_h = data_handler.DataHandler(BASE_PATH)
data_h.load_data()


feat = features.FeatureDetector()
viz = visualizer.Visualizer()

def test_features():
	img = cv2.imread(data_h.vehicles[0])
	plt.figure()
	plt.imshow(img)
	plt.figure()
	plt.imshow(feat.convert_color_space(img))
	plt.figure()
	# plt.plot(feat.get_HOG(img)[0])
	plt.plot(feat.get_features(img))
	plt.show()

def test_windows():
	for img_file in data_h.test_images:
		img = cv2.imread(img_file)
		windows = vehicle_detector.get_windows(img)
		plt.figure()
		processed_img = np.copy(img)
		for idx,window in enumerate(windows):
			processed_img = viz.draw_bounding_box(processed_img,window,color=viz.COLORS[idx])

			# plt.figure()
			# plt.imshow(img)
		plt.imshow(processed_img)
		plt.show()
		# break
def test_car_detection():
	img = cv2.imread(data_h.test_images[2])
	vehicle_detector.process_image(img)
	# process_image = np.copy(img)
	# viz_image = np.copy(img)
	# plt.figure()
	# windows = vehicle_detector.get_windows(img)
	# # print(windows[0][0])
	# for window in windows:
	# 	for cell in window:
	# 		print(cell)
	# 		image_roi = process_image[cell[0][0]:cell[0][1],cell[1][0]:cell[1][1]]
	# 		image_roi = cv2.resize(image_roi, (32,32))
	# 		X = feat.get_features(image_roi)
	# 		plt.plot(X)
	# 		viz_image = viz.draw_bounding_box(viz_image,[cell],color=viz.GREEN)
	# 		# plt.imshow(image_roi)
	# 		plt.show()
	# 		break

		
	# plt.imshow(viz_image)
	# plt.show()

def train_pipeline():
	vehicle_detector.prepare_dataset()

def unit_tests():
	# test_features()
	# test_windows()
	test_car_detection()


if __name__ == '__main__':
	unit_tests()
	# train_pipeline()