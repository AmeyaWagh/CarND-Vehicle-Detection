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

