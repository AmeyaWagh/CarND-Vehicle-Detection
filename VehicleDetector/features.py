import cv2
import numpy as np
from skimage.feature import hog

class FeatureDetector(object):
    """
        This class takes care of the feature extraction
    """
    def __init__(self):
        self.color_space = cv2.COLOR_RGB2YCrCb
        self.orientations = 16
        self.pixels_per_cell = (12,12)
        self.cells_per_block = (2,2)
        self.image_size = (32,32)
        self.color_feat_size = (64,64)
        self.no_of_bins = 32
        self.old_heatmap = None

        self.color_features = False
        self.spatial_features = False
        self.HOG_features = True

    def get_features(self,image):
        """
            All feature of the image are computed here and
            are concatenated to form a single feature vector
        """
        _image = np.copy(image)
        _image = cv2.resize(_image, self.image_size)
        _image = self.convert_color_space(_image)

        Features = []
        
        if self.color_features:
            color_hist = self.get_color_features(_image)
            Features.append(color_hist)
        
        if self.spatial_features:    
            spatial_hist = self.get_spatial_features(_image)
            Features.append(spatial_hist)
        
        if self.HOG_features:    
            hog_hist = self.get_HOG(_image)
            Features.append(hog_hist)
        
        # features = np.concatenate((color_hist, spatial_hist, hog_hist))
        features = np.concatenate((Features))
        return features

    def convert_color_space(self,image):
        return cv2.cvtColor(image,self.color_space)

    def get_spatial_features(self,image):
        """
            returns the histogram of individual channels of 
            image in given color space.
            returns stacked feature vector of all 3 channels
        """
        ch1_hist = np.histogram(image[:, :, 0], bins=self.no_of_bins)
        ch2_hist = np.histogram(image[:, :, 1], bins=self.no_of_bins)
        ch3_hist = np.histogram(image[:, :, 2], bins=self.no_of_bins)
        return np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))
    
    def get_color_features(self,image):
        """
            flattens the given channel of the image
            returns stacked feature vector of all 3 channels
        """
        ch1_featr = image[:,:,0].ravel()
        ch2_featr = image[:,:,1].ravel()
        ch3_featr = image[:,:,2].ravel()

        return np.hstack((ch1_featr, ch2_featr, ch3_featr))

    def get_HOG(self,image):
        """
            HOG of every channel of given image is compuuted and
            is concatenated to form one single feature vector
        """
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

    def get_heatmap(self,image,bboxes,threshold=2):
        """
            A heatmap of image is created and heat is added in the region 
            covered by individual bounding box. Threshold is then applied 
            and outliers are removed
        """
        heat_map = np.zeros((image.shape[0],image.shape[1]))
        for bbox in bboxes:
            heat_map[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]+=1
        
        # adding previous heatmap is like tracking the object
        if self.old_heatmap is not None:
            heat_map += 0.99*threshold*self.old_heatmap
        heat_map[heat_map <= threshold] = 0
        heat_map[heat_map > threshold] = 1
        self.old_heatmap = heat_map
        return heat_map
