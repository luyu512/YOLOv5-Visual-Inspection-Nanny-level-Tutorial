# YOLOv5 视觉检测保姆级教程

在前一篇教程的基础上(地址：https://github.com/luyu512/YOLOv5-deployment-in-Windows-environment)

我们开始学习如何使用YOLOv5进行视觉检测

## 第一步： 下载labelImg标注软件：

项目地址：https://github.com/tzutalin/labelImg?spm=a2c6h.12873639.article-detail.15.536d6c74Hfnwa8

或者直接在终端中输入（推荐）：
```
git clone https://gitclone.com/github.com/HumanSignal/labelImg.git
conda install pyqt=5
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
```

之后，会出现如图如图所示页面![images](https://github.com/luyu512/YOLOv5-Visual-Inspection-Nanny-level-Tutorial/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-29%20215152.png)

说明安装成功！

## 第一步：建立VOCData文件夹

在yolov5文件夹目录中，我们建立VOCData文件夹，并且在其中建立Annotations和images文件夹
![images](https://github.com/luyu512/YOLOv5-Visual-Inspection-Nanny-level-Tutorial/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-03-29%20211411.png)


