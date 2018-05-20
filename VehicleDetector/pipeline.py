from VehicleDetector import classifier
from VehicleDetector import features
from VehicleDetector import data_handler
import cv2
import numpy as np


class VehicleDetector(object):
    """
            The complete end to end Vehicle detection pipeline
    """

    def __init__(self, BASE_PATH="."):
        print("Vehicle Detector")
        self.BASE_PATH = BASE_PATH
        self.data_h = data_handler.DataHandler(BASE_PATH)
        self.data_h.load_data()
        self.feat = features.FeatureDetector()
        self.clf = classifier.Classifier()

    def prepare_dataset(self):
        """
        Prepares data of CAR and NON-CAR for training The classifier
        """
        self.features = []
        self.labels = []
        total_files = len(self.data_h.vehicles)+len(self.data_h.non_vehicles)
        count = 0
        for vehicle_file in self.data_h.vehicles:
            printProgressBar(count, total_files, prefix='Progress:',
                             suffix='Complete', length=50)
            # print(vehicle_file)
            image = cv2.imread(vehicle_file)
            feature = self.feat.get_features(image)
            self.features.append(feature)
            self.labels.append([1])
            count += 1
        for non_vehicle_file in self.data_h.non_vehicles:
            printProgressBar(count, total_files, prefix='Progress:',
                             suffix='Complete', length=50)
            image = cv2.imread(non_vehicle_file)
            feature = self.feat.get_features(image)
            self.features.append(feature)
            self.labels.append([0])
            count += 1

        self.features = np.vstack(self.features).astype(np.float64)
        self.labels = np.array(self.labels)

        print("[features size]{} [labels size]{} ".format(
            self.features.shape, self.labels.shape))

        X_train, X_test, y_train, y_test = self.data_h.prepare_trainable_data(
            self.features, self.labels)
        self.clf.train(X_train, y_train, X_test, y_test)


    def process_image(self):
        pass


def printProgressBar(iteration, total, prefix='', suffix='',
                     decimals=1, length=100, fill='█'):
    """
    ref: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:."+str(decimals)+"f}").format(100 *
                                                 (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if percent == 100.0:
        print()
