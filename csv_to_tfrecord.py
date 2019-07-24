from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
import config

labels = config.LABELS

#根据label name返回id
def class_text_to_int(row_label):
    try:
        return labels.index(row_label) + 1
    except ValueError as identifier:
        None


def split(df,  group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group):
    with tf.gfile.GFile('{}'.format(group.filename), 'rb') as fid:
        encoded_jpg = fid.read()
    
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmin = row['xmin'] / width
        xmax = row['xmax'] / width
        ymin = row['ymin'] / height
        ymax = row['ymax'] / height
        if xmin>1 or xmax>1 or ymin>1 or ymax>1:
            continue 
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    #转换train.csv
    writer = tf.python_io.TFRecordWriter(config.TRAIN_TF)
    #path = os.path.join(os.getcwd(), config.IMAGES_TRAIN_DIR)
    path = config.IMAGES_TRAIN_DIR
    examples = pd.read_csv(config.TRAIN_CSV)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), config.TRAIN_TF)
    print('Successfully created TFRecords:{}'.format(output_path))
    
    #转换test.csv
    writer = tf.python_io.TFRecordWriter(config.TEST_TF)
    #path = os.path.join(os.getcwd(), config.IMAGES_TEST_DIR)
    path = config.IMAGES_TEST_DIR
    examples = pd.read_csv(config.TEST_CSV)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group)
        writer.write(tf_example.SerializeToString())
    
    writer.close()
    output_path = os.path.join(os.getcwd(), config.TEST_TF)
    print('Successfully created TFRecords:{}'.format(output_path))

if __name__ == '__main__':
    tf.app.run()