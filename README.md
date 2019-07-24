# Object_detection-api-lightweight_version
## TensorFlow Object_detection api轻量版本
### 仅保存了api中的object_detection部分，将很多api使用步骤精简为了可执行脚本，并且可以从config文件统一配置。用户只要将需要做目标检测的图片和标签文件放到正确位置，依次执行脚本就可以完成目标检测功能的训练和检测。
### 现在支持的预训练模型为faster_rcnn_inception_v2，用户也可以根据需要自己从(https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) 下载更多预训练模型，功能实现的部分代码和思路借鉴 (https://blog.csdn.net/csdn_6105/article/details/82933628)
## 文件结构
#### |--object_detection
#### &nbsp;&nbsp;&nbsp;|--images 需要训练的图片
#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--test 图片测试集
#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--train 图片训练集
#### &nbsp;&nbsp;&nbsp;|--xml 图片标签
#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--test 测试图片标签
#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--train 训练图片标签
#### &nbsp;&nbsp;&nbsp;|--inference_graph 存放训练完的固定参数
#### &nbsp;&nbsp;&nbsp;|--object_detection  TensorFlow objection_detection api文件
#### &nbsp;&nbsp;&nbsp;|--pretrained 存放预训练模型
#### &nbsp;&nbsp;&nbsp;|--test 存放需检测图片
#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--result_images 存放检测完的图片
#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--test_images 存放待检测图片
#### &nbsp;&nbsp;&nbsp;|--training 存放训练模型
#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--shared 存放共享参数
#### &nbsp;&nbsp;&nbsp;|--config.py 参数设定
#### &nbsp;&nbsp;&nbsp;|--xml_to_csv.py 将LabelImg形成的xml文件转换为csv文件
#### &nbsp;&nbsp;&nbsp;|--csv_to_tfrecord.py 将csv文件转换为TensorFlow使用的tfrecord文件
#### &nbsp;&nbsp;&nbsp;|--before_training.py 训练前配置文件初始化
#### &nbsp;&nbsp;&nbsp;|--begin_training.py 开始训练模型
#### &nbsp;&nbsp;&nbsp;|--after_training.py 训练模型结束后，固定化参数
#### &nbsp;&nbsp;&nbsp;|--detection.py 检测器
#### &nbsp;&nbsp;&nbsp;|--test.py 检测图片
                
## 使用方法
### Step1
#### 使用LabelImg来给图片标注数据（具体使用方法可以自行搜索），将标注前的图片，根据训练集和测试集分别存放于 images/train 和images/test 中，在LabelImg中设置，分别将记有图片标注信息的xml文件存放到 xml/train 和 xml/test 中（如果有已经生成好的tfrecord文件，直接将其置于根目录，分别命名为train.tfrecord和test.tfrecord）。
#### 打开config.py 文件，修改LABELS为需要的识别的目标标签，修改MAX_STEPS为所希望的最大训练步数，如果要增加所用模型，自行修改模型配置文件
#### 执行 python xml_to_csv.py 和 python csv_to_tfrecord.py，在根目录生成tfrecord文件，用于训练的输入
### Step2
#### 执行 python before_training.py --model=id(id为数字，默认为0，为config.py文件中配置的模型的列表的序号,下同)
#### 会在 trianing/xxx(对应模型目录) 下生成修改后的.config配置文件，并在 training/shared 下生成对应的label文件
### Step3
#### 执行 python begin_training.py --model=id，调用 object_detection 下的 legacy/train.py 开始训练模型
### Step4
#### 等待模型训练完成之后，执行 python after_training.py --model=id, 在 inference_graph/xxx(对应模型目录) 下生成固定化参数的文件，用于目标检测
### Step5
#### 执行 python test.py --model=id，会调用 inference_graph 目录下对应固定化参数的文件来做目标检测。将会对每个处于 test/test_images 下的图片进行检测，输出结果按时间方式命名，存放在 test/result_images 下

## Todo
### 加入视频中目标检测功能
