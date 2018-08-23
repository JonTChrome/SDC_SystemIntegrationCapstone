import numpy as np
import cv2
from styx_msgs.msg import TrafficLight

class SimModel():
  def __init__(self):
    self.light_ar = 80
  def predict(self, image):
    green_img = image[:,:,1]
    red_img = image[:,:,2]
    red_area = np.sum(red_img == red_img.max())
    green_area = np.sum(green_img == green_img.max())

    if red_area >= self.light_ar and green_area <= self.light_ar:
        return TrafficLight.RED
    elif red_area >= self.light_ar and green_area >= self.light_ar:
        if 0.8 <= red_area / green_area <= 1.2:
            return TrafficLight.YELLOW 
        else:
            return TrafficLight.RED
    elif green_area >= self.light_ar:
        return TrafficLight.GREEN
    else:
        return TrafficLight.UNKNOWN