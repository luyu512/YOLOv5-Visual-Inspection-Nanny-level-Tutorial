# YOLOv5 视觉检测保姆级教程

在前一篇教程的基础上(地址：https://github.com/luyu512/YOLOv5-deployment-in-Windows-environment)

我们开始学习如何使用YOLOv5进行视觉检测

## 第一步： 下载labelImg标注软件：

项目地址：https://github.com/tzutalin/labelImg?spm=a2c6h.12873639.article-detail.15.536d6c74Hfnwa8

或者直接在终端中输入（推荐）：
```
pip install labelimg -i https://pypi.tuna.tsinghua.edu.cn/simple
labelImg
```

之后，会出现如图如图所示页面![images](https://github.com/luyu512/YOLOv5-Visual-Inspection-Nanny-level-Tutorial/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-29%20215152.png)

说明安装成功！

## 第二步：建立VOCData文件夹

在yolov5文件夹目录中，我们建立VOCData文件夹，并且在其中建立Annotations和images文件夹

images文件夹用于储存拍摄的图片，Annotations文件夹用于储存标注文件
![images](https://github.com/luyu512/YOLOv5-Visual-Inspection-Nanny-level-Tutorial/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-29%20211411.png)

## 第三步：标注数据集

打开labelImg标注软件

`labelImg`

点击左侧“打开目录”，选择images文件夹。此时可以看到图片已经导入进来
![images](https://github.com/luyu512/YOLOv5-Visual-Inspection-Nanny-level-Tutorial/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-29%20214218.png)

点击左侧“改变存放目录”，选择Annotations文件夹。这样标注好的图像数据会存储在此
![images](https://github.com/luyu512/YOLOv5-Visual-Inspection-Nanny-level-Tutorial/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-29%20214235.png)

完成后，点击左上角“查看”，选择自动保存模式

按下键盘上W键，光标变成十字准星，按住鼠标右键，此时用户可以选择标注区域
松开右键即为框选完成，此时输入标签名称后标注完成，可以看到Annotations文件夹中多出一.xml文件

## 第四步：创建图片转换和划分数据集文件

1. 在VOCData文件夹中建立split_train_val.py文件
```
# coding:utf-8
 
import os
import random
import argparse
 
parser = argparse.ArgumentParser()
# xml文件的地址，根据自己的数据进行修改 xml一般存放在Annotations下，注意以下为相对路径
parser.add_argument('--xml_path', default=r'D:\yolov5-master\VOCData\Annotations', type=str, help='input xml label path')
# 数据集的划分，地址选择自己数据下的ImageSets/Main，注意以下为相对路径
parser.add_argument('--txt_path', default=r'D:\yolov5-master\VOCData\ImageSets\Main', type=str, help='output txt label path')
opt = parser.parse_args()
 
trainval_percent = 1.0  # 训练集和验证集所占比例。 这里没有划分测试集
train_percent = 0.9  # 训练集所占比例，可自己进行调整
xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)
 
num = len(total_xml)
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
train = random.sample(trainval, tr)
 
file_trainval = open(txtsavepath + '/trainval.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')
file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')
 
for i in list_index:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        file_trainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)
 
file_trainval.close()
file_train.close()
file_val.close()
file_test.close()
```

运行之后发现VOCData中多出ImageSets\Main文件

2. 在VOCData文件夹中建立text_to_yolo.py文件
```
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import getcwd
 
sets = ['train', 'val', 'test']
classes = ["hong", "huang", "lan", "hei"]  # 改为自己的类别
abs_path = os.getcwd()
print(abs_path)
 
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h
 
def convert_annotation(image_id):
    in_file = open('D:/yolov5-master/VOCData/Annotations/%s.xml' % (image_id), encoding='UTF-8')
    out_file = open('D:/yolov5-master/VOCData/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        # difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
 
wd = getcwd()
for image_set in sets:
    # 这里是绝对路径，需要根据自己的情况修改
    if not os.path.exists('D:/yolov5-master/VOCData/labels/'):
        os.makedirs('D:/yolov5-master/VOCData/labels/')
    image_ids = open('D:/yolov5-master/VOCData/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
 
    if not os.path.exists('D:/yolov5-master/VOCData/dataSet_path/'):
        os.makedirs('D:/yolov5-master/VOCData/dataSet_path/')
 
    list_file = open('D:/yolov5-master/VOCData/dataSet_path/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('D:/yolov5-master/VOCData/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
```

运行之后发现VOCData中多出labels和dataSet_path文件夹
![images](https://github.com/luyu512/YOLOv5-Visual-Inspection-Nanny-level-Tutorial/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-29%20212509.png)
此时说明两.py文件运行正常，可进行下一步操作

## 第五步： 创建myvoc.yaml文件并修改模型文件


在data文件夹中建立myvoc.yaml文件

```
train: C:/yolov5/yolov5-master/VOCData/dataSet_path/train.txt
val: C:/yolov5/yolov5-master/VOCData/dataSet_path/val.txt     #修改成自己的路径
 
nc: 4   #改成自己标签类别数
 
names: ["hong", "huang", "lan", "hei"]  #修改成自己的标签名称
```

在models文件夹中选择模型文件（此教程选用5s，体积较小，在保持准确速度的前提下训练速度快）
修改第四行的数字为自己的标签类别数
![images](https://github.com/luyu512/YOLOv5-Visual-Inspection-Nanny-level-Tutorial/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-29%20223106.png)

## 第六步： 使用train.py训练模型

找到train.py文件，修改第569行“coco128.yaml”为“myvoc.yaml”

在第580行修改训练轮次

在第588行选择gpu or cpu
![images](https://github.com/luyu512/YOLOv5-Visual-Inspection-Nanny-level-Tutorial/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-29%20222937.png)
然后就可以开始训练了
