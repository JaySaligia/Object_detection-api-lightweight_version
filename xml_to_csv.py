import xml.dom.minidom
from PIL import Image
import pandas as pd
import os
import config
import os
#xml_list_train储存训练集，xml_list_test存储测试集
xml_list_train = []
xml_list_test = []
images_train_dir = config.IMAGES_TRAIN_DIR
images_test_dir = config.IMAGES_TEST_DIR
xml_train_dir = config.XML_TRAIN_DIR
xml_test_dir = config.XML_TEST_DIR

def data_parser(xml_loc, flag):
    if flag == 0:
        images_dir = images_train_dir
    else:
        images_dir = images_test_dir
    parser = xml.dom.minidom.parse(xml_loc)
    root = parser.documentElement
    img_name = images_dir + root.getElementsByTagName('filename').item(0).childNodes[0].nodeValue
    img_size = root.getElementsByTagName('size').item(0)
    img_width = int(img_size.getElementsByTagName('width').item(0).childNodes[0].nodeValue)
    img_height = int(img_size.getElementsByTagName('height').item(0).childNodes[0].nodeValue)
    objects = root.getElementsByTagName('object')
    for obj in objects:
        label_name = obj.getElementsByTagName('name').item(0).childNodes[0].nodeValue
        bndbox = obj.getElementsByTagName('bndbox').item(0)
        xmin = int(bndbox.getElementsByTagName('xmin').item(0).childNodes[0].nodeValue)
        ymin = int(bndbox.getElementsByTagName('ymin').item(0).childNodes[0].nodeValue)
        xmax = int(bndbox.getElementsByTagName('xmax').item(0).childNodes[0].nodeValue)
        ymax = int(bndbox.getElementsByTagName('ymax').item(0).childNodes[0].nodeValue)
        value = (img_name, img_width, img_height, label_name, xmin, ymin, xmax, ymax)
        if flag == 0:
            xml_list_train.append(value)
        else:
            xml_list_test.append(value)

if __name__ == "__main__":
    xmls_train = os.listdir(xml_train_dir)
    for xml_ in xmls_train:
        data_parser(xml_train_dir + xml_, 0)

    xmls_test = os.listdir(xml_test_dir)
    for xml_ in xmls_test:
        data_parser(xml_test_dir + xml_, 1)
        
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df_train = pd.DataFrame(xml_list_train, columns=column_name)
    xml_df_test = pd.DataFrame(xml_list_test, columns=column_name)
    xml_df_train.to_csv(('train.csv'), index=None)
    xml_df_test.to_csv(('test.csv'), index = None)
    print('Successfully generate csv file')


