import os
import tensorflow as tf
import config

flags = tf.app.flags
flags.DEFINE_integer('model_id', 0, 'choose the DNN model, default id:0')
FLAGS = flags.FLAGS

def exec_train(id):
    os.system('python object_detection/legacy/train.py --train_dir={} --pipeline_config_path={} --alsologtostderr'.format(config.TRAIN_DIR[id], config.CONFIG_FILE[id]))

if __name__ == "__main__":
    exec_train(FLAGS.model_id)
