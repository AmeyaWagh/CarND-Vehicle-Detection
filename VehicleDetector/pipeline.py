from VehicleDetector import classifier
from VehicleDetector import features
from VehicleDetector import data_handler
from VehicleDetector import visualizer
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label


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
        self.viz = visualizer.Visualizer()
        self.clf.load_classifier()

        self.probability_threshold = 0.5

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


    def slide_window(self,img, x_start_stop=[None, None], y_start_stop=[None, None], 
                        xy_window=(32, 32), xy_overlap=(0.5, 0.5)):
        """
            code used from lessons
        """
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def get_windows(self,image):
        window_image = np.copy(image)

        height, width,_ = window_image.shape

        # print(width,height)
        scale_factors = [(0.4,1.0,0.55,0.8,64),
                        (0.2,1.0,0.55,0.8,100),
                        (0.4,1.0,0.55,0.9,120),
                        (0.2,1.0,0.55,0.9,140),
                        (0.4,1.0,0.55,0.9,160),
                        (0.3,1.0,0.50,0.9,180)]

        windows = list()
        for scale_factor in scale_factors:
            window_1 = self.slide_window(window_image,
                            x_start_stop=[int(scale_factor[0]*width), 
                                            int(scale_factor[1]*width)], 
                            y_start_stop=[int(scale_factor[2]*height), 
                                            int(scale_factor[3]*height)],
                            
                            xy_window=( scale_factor[4], 
                                        scale_factor[4]), 
                            xy_overlap=(0.5, 0.5))
            windows.append(window_1)
        

        return windows

    def process_image(self,image):
        process_image = np.copy(image)
        viz_image = np.copy(image)
        plt.figure()
        windows = self.get_windows(image)
        # print(windows[0][0])

        cells = []
        all_X = []
        for window in windows:
            for cell in window:
                try:
                    image_roi = process_image[cell[0][1]:cell[1][1], cell[0][0]:cell[1][0]]
                    
                    if sum(image_roi.shape)-sum((32,32,3)) > 0:
                        X = self.feat.get_features(image_roi)
                        all_X.append(X)
                        cells.append(cell)
                        
                except cv2.error as e:
                    # print(e)
                    pass
        # print(len(all_X))
        # print('-'*80)
        features = np.vstack(all_X).astype(np.float64)
        features = self.data_h.scale_vector(features)
        pred_proba = self.clf.predict(features)
        # print('-'*80)           
        # print(len(cells))
        final_bbox = []
        for i,_pred in enumerate(pred_proba):
            if _pred[1] > self.probability_threshold:
                # print(pred_proba[i])
                viz_image = self.viz.draw_bounding_box(viz_image,[cells[i]],color=self.viz.GREEN)
                final_bbox.append(cells[i])

        # plt.imshow(viz_image)
        # plt.figure()
        heat_map = self.feat.get_heatmap(image, final_bbox)
        labels = label(heat_map)
        # plt.imshow(heat_map)
        # plt.figure()
        # plt.imshow(labels[0], cmap='gray')
        final_img = self.viz.draw_labeled_bounding_box(image,labels)
        # plt.figure()
        # plt.imshow(final_img)
        # print(labels)
        # plt.show()
        return final_img





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
