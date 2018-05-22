import cv2
import numpy as np
from skimage.feature import hog

class FeatureDetector(object):
    """
        This class takes care of the feature extraction
    """
    def __init__(self):
        self.color_space = cv2.COLOR_RGB2YCrCb
        self.orientations = 9
        self.pixels_per_cell = (12,12)
        self.cells_per_block = (2,2)
        self.image_size = (32,32)
        self.color_feat_size = (64,64)
        self.no_of_bins = 32

    def get_features(self,image):
        _image = np.copy(image)
        _image = cv2.resize(_image, self.image_size)
        _image = self.convert_color_space(_image)
        
        color_hist = self.get_color_features(_image)
        spatial_hist = self.get_spatial_features(_image)
        hog_hist = self.get_HOG(_image)
        
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
        
        ch1_featr = image[:,:,0].ravel()
        ch2_featr = image[:,:,1].ravel()
        ch3_featr = image[:,:,2].ravel()

        return np.hstack((ch1_featr, ch2_featr, ch3_featr))

    def get_HOG(self,image):
        feat_ch1 = hog(image[:,:,0], 
                            orientations= self.orientations , 
                            pixels_per_cell= self.pixels_per_cell , 
                            cells_per_block= self.cells_per_block,
                            visualise=False)
        feat_ch2 = hog(image[:,:,1], 
                            orientations= self.orientations , 
                            pixels_per_cell= self.pixels_per_cell , 
                            cells_per_block= self.cells_per_block,
                            visualise=False)
        feat_ch3 = hog(image[:,:,2], 
                            orientations= self.orientations , 
                            pixels_per_cell= self.pixels_per_cell , 
                            cells_per_block= self.cells_per_block,
                            visualise=False)
        return np.concatenate((feat_ch1, feat_ch2, feat_ch3))

    def get_heatmap(self,image,bboxes,threshold=1):
        heat_map = np.zeros((image.shape[0],image.shape[1]))
        for bbox in bboxes:
            heat_map[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]+=1
        heat_map[heat_map <= threshold] = 0
        heat_map[heat_map > threshold] = 1
        return heat_map
