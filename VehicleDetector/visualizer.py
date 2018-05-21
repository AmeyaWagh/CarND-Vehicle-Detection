import cv2
import numpy as np



class Visualizer(object):
    RED = (255,0,0)
    GREEN = (0,255,0)
    BLUE = (0,0,255)
    CYAN = (0,255,255)
    YELLOW = (255,255,0)
    MAGENTA = (255,0,255)

    COLORS = [RED,GREEN,BLUE,CYAN,YELLOW,MAGENTA]
    def __init__(self):
        self.line_thickness = 1
    
    def draw_bounding_box(self, image, bboxes, color=YELLOW):
        bb_image = np.copy(image)
        for bbox in bboxes:
            cv2.rectangle(bb_image, bbox[0], bbox[1], 
                            color, self.line_thickness)
        return bb_image
    def draw_labeled_bounding_box(self,image,labels):
        for car in range(1,labels[1]+1):
            nonzero = (labels[0] == car).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            height, width,_ = image.shape
            if (bbox[0][0]>0.45*width) and (bbox[0][1]>0.45*height):
                cv2.rectangle(image, bbox[0], bbox[1], self.GREEN, self.line_thickness+5)
            # Return the image
        return image


