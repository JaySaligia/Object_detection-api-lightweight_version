from detection import detection
import os
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer('model_id', 0 , 'Choose the DNN model, default is faster_rcnn_inception_v2') 
FLAGS = flags.FLAGS

def test_all(id):
    path = 'test/test_images'
    images = os.listdir(path)

    for image in images:
        detect = detection(path + '\\' + image, id)
        detect.make_detect_img()

if __name__ == "__main__":
    test_all(FLAGS.model_id)
    