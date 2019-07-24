import os
import cv2
import numpy as np
import tensorflow as tf
import sys
sys.path.append('object_detection')
from utils import label_map_util
from utils import visualization_utils as vis_util
import datetime
import config
class detection():
    def __init__(self, image_name, id):
        self.IMAGE_NAME = image_name
        self.MODEL_NAME = 'inference_graph/' + config.MODEL_NAME[id]
        self.CWD_PATH = os.getcwd()
        self.PATH_TO_CKPT = os.path.join(self.CWD_PATH,self.MODEL_NAME,'frozen_inference_graph.pb')
        self.PATH_TO_LABELS = os.path.join(self.CWD_PATH, config.LABELMAP)
        self.PATH_TO_IMAGE = os.path.join(self.CWD_PATH, self.IMAGE_NAME)
        self.NUM_CLASSES = (len(config.LABELS))
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def make_detect_img(self):
        image_raw = cv2.imread(self.PATH_TO_IMAGE)
        print("Now process image " + self.PATH_TO_IMAGE)
        img_height = image_raw.shape[0]
        img_width = image_raw.shape[1]
        total_count = 0
        image_together, total_count = self.make_detect_smallimage(image_raw)
        ISOTIMEFORMAT = '%Y-%m-%d-%H-%M-%S'
        f_time = datetime.datetime.now().strftime(ISOTIMEFORMAT)
        cv2.imwrite('test/result_images/' + str(f_time) + '-result.jpg', image_together)
        print("Now end process image " + self.PATH_TO_IMAGE)

    def make_detect_cam(self, image_raw):
        image_together = self.make_detect_smallimage(image_raw)
        return image_together
        

    def make_detect_smallimage(self, image):
        #load tf model into memory
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)
        #get some info
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        #image = cv2.imread(self.PATH_TO_IMAGE)
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),#向量数组转换为置为1的数组
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=0.8)
        
        #cv2.imshow('Object detector', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        head_count = len(np.squeeze(boxes))
        return image, head_count




        
