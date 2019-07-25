import config
import os
#仅修改num_class,num_steps和写入相关模型文件的位置
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer('model_id', 0 ,'Choose the DNN model, default is faster_rcnn_inception_v2')
FLAGS = flags.FLAGS

def make_config(id): 
    data = ''
    count = 0
    with open(config.CONFIG_FILE[id], 'r+') as f:
        for line in f.readlines():
            if line.find('num_classes') >= 0:
                line = '    num_classes: {}'.format(len(config.LABELS)) + '\n'
            elif line.find('num_steps') >= 0:
                line = '    num_steps: {}'.format(config.MAX_STEPS) + '\n'
            elif line.find('fine_tune_checkpoint') >=0 :
                line = '    fine_tune_checkpoint: "{}"'.format(os.getcwd().replace('\\', '/') + '/' + config.MODEL_CKPT[0]) + '\n'
            elif line.find('input_path') >= 0:
                if count == 0:
                    line = '    input_path: "{}"'.format(os.getcwd().replace('\\', '/') + '/' + config.TRAIN_TF) + '\n'
                    count += 1
                else:
                    line = '    input_path: "{}"'.format(os.getcwd().replace('\\', '/') + '/' + config.TEST_TF) + '\n'
            elif line.find('label_map_path') >= 0:
                    line = '    label_map_path: "{}"'.format(os.getcwd().replace('\\', '/') + '/' + config.LABELMAP) + '\n' 
            data += line

    with open(config.CONFIG_FILE[id], 'r+') as f:
        f.write(data)            
    print('{} config file has been changed'.format(config.MODEL_NAME[id]))

def make_pbtxt():
    data = []
    for i in range(len(config.LABELS)):
        item = 'item {\nid:' + str(i+1) + '\n' + 'name:\'{}\'\n'.format(config.LABELS[i]) + '}\n'
        data.append(item)

    with open(config.LABELMAP, 'w') as f:
        f.writelines(data)
    
    print('pbtxt has been writen')

if __name__ == "__main__":
    make_config(FLAGS.model_id)
    make_pbtxt()

