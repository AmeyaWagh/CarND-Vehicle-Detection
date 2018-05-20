from VehicleDetector import pipeline
from VehicleDetector import data_handler
from VehicleDetector import features
import glob
import cv2
import matplotlib.pyplot as plt

BASE_PATH = '.'

vehicle_detector = pipeline.VehicleDetector(BASE_PATH = BASE_PATH)
data_h = data_handler.DataHandler(BASE_PATH)
data_h.load_data()


feat = features.FeatureDetector()

# img = cv2.imread(data_h.vehicles[0])
# plt.figure()
# plt.imshow(img)
# plt.figure()
# plt.imshow(feat.convert_color_space(img))
# plt.figure()
# # plt.plot(feat.get_HOG(img)[0])
# plt.plot(feat.get_features(img))
# plt.show()

vehicle_detector.prepare_dataset()

# test_images = glob.glob("./test_images/*.jpg")
# print(test_images)

# vehicles = glob.glob("./vehicles/**/*.png")
# non_vehicles = glob.glob("./non-vehicles/**/*.png")

# print(len(vehicles))
# print(len(non_vehicles))

if __name__ == '__main__':
	pass