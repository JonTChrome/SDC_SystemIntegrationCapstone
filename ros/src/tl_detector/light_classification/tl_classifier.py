from styx_msgs.msg import TrafficLight
from sim_model import SimModel
import tensorflow as tf
import numpy as np
import os
import rospy

#pre-trained models detected stop lights, only way too close to the actual stopping line, didnt allow time
# for deceleration.
GRAPH_PATH = './model/frozen_inference_graph.pb'
GRAPH_PATH_SIM = './model/frozen_inference_graph_sim.pb'
THRESHOLD = 0.6

class TLClassifier(object):
    def __init__(self, is_sim):
        self.is_sim = is_sim
        if is_sim:
            self.model = SimModel()
        else:
            self.threshold = THRESHOLD
            self.graph = self.load_graph(GRAPH_PATH_SIM)
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.detect_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.detect_scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.detect_classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')
            self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light, BGR channel
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if self.is_sim:
            return self.model.predict(image)
        else: 
            with self.graph.as_default():
                image_expanded = np.expand_dims(image, axis = 0)

                (boxes, scores, classes, num) = self.sess.run([self.detect_boxes, self.detect_scores, self.detect_classes, self.num_detections], feed_dict = {self.image_tensor: image_expanded})

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes)

                if scores[0] > self.threshold:

                    if classes[0] == 1:
                        return TrafficLight.GREEN
                    elif classes[0] == 2:
                        return TrafficLight.RED
                    elif classes[0] == 3:
                        return TrafficLight.YELLOW
            return TrafficLight.UNKNOWN
            
    def load_graph(self, graph_path):
        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                read_graph = fid.read()
                graph_def.ParseFromString(read_graph)
                tf.import_graph_def(graph_def, name='')

        return graph