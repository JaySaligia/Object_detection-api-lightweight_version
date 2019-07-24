#图像存储位置
IMAGES_TRAIN_DIR = 'images/train/'
IMAGES_TEST_DIR = 'images/test/'
#xml文件存储位置
XML_TRAIN_DIR = 'xml/train/'
XML_TEST_DIR = 'xml/test/'
#分类标签
LABELS = ['dog', 'cat', 'person', 'car']
#csv文件位置
TRAIN_CSV = 'train.csv'
TEST_CSV = 'test.csv'
#tfrecord文件位置
TRAIN_TF = 'train.record'
TEST_TF = 'test.record'
#模型训练最大步数
MAX_STEPS = 100
#label标签位置
LABELMAP = 'training/shared/labelmap.pbtxt'
#各个模型配置文件
MODEL_NAME = ['faster_rcnn_inception_v2']
TRAIN_DIR = ['training/faster_rcnn_inception_v2/']
CONFIG_FILE = ['training/faster_rcnn_inception_v2/faster_rcnn_inception_v2.config']
MODEL_CKPT = ['pretrained/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt']
