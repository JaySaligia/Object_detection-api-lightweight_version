import os
import tensorflow as tf
import config

flags = tf.app.flags
flags.DEFINE_integer('model_id', 0, 'choose the DNN model, default id:0')
FLAGS = flags.FLAGS

def exec_inference_graph(id):
    os.system('python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path {} --trained_checkpoint_prefix {}model.ckpt-{} --output_directory inference_graph/{}'.format(config.CONFIG_FILE[id], config.TRAIN_DIR[id], config.MAX_STEPS, config.MODEL_NAME[id]))

if __name__ == "__main__":
    exec_inference_graph(FLAGS.model_id)
