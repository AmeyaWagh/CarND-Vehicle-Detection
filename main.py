from VehicleDetector import pipeline
from VehicleDetector import data_handler
from VehicleDetector import features
from VehicleDetector import visualizer
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

BASE_PATH = '.'

OUTPUT_FILE = 'project_output.mp4'
INPUT_FILE = 'project_video.mp4'

vehicle_detector = pipeline.VehicleDetector(BASE_PATH = BASE_PATH)
data_h = data_handler.DataHandler(BASE_PATH)
data_h.load_data()


feat = features.FeatureDetector()
viz = visualizer.Visualizer()

def test_features():
	from skimage.feature import hog
	img = cv2.imread(data_h.vehicles[0])
	plt.figure()
	plt.imshow(img)
	plt.figure()
	plt.imshow(feat.convert_color_space(img))
	plt.figure()
	feat_ch1,out_img = hog(img[:,:,0], 
                            orientations= 16 , 
                            pixels_per_cell= (12,12) , 
                            cells_per_block= (2,2),
                            visualise=True)
	plt.imshow(out_img)
	# plt.plot(feat.get_features(img))
	plt.show()

def test_windows():
	for img_file in data_h.test_images:
		img = cv2.imread(img_file)
		windows = vehicle_detector.get_windows(img)
		plt.figure()
		processed_img = np.copy(img)
		for idx,window in enumerate(windows):
			processed_img = viz.draw_bounding_box(processed_img,window,color=viz.COLORS[idx])

		plt.imshow(processed_img)
		plt.show()
		# break
def test_car_detection():
	for img_file in data_h.test_images:
		img = cv2.imread(img_file)
		final_img,heat_map,viz_img = vehicle_detector.process_image(img,visualization=True)		

		plt.figure()
		plt.imshow(final_img)
		plt.figure()
		plt.imshow(heat_map,cmap='gray')
		plt.figure()
		plt.imshow(viz_img)
		plt.show()

def train_pipeline():
	vehicle_detector.prepare_dataset()

def unit_tests():
	test_features()
	test_windows()
	test_car_detection()

def process_video():
	video = VideoFileClip(INPUT_FILE)
	processed_video = video.fl_image(vehicle_detector.process_image)
	processed_video.write_videofile(OUTPUT_FILE, audio=False)

if __name__ == '__main__':
	# unit_tests()
	# train_pipeline()
	process_video()