# segment_car
### 这个仓库里存放的是使用cityscapes数据集训练unet,segformer-2b,deeplabv3网络来进行汽车语义分割额代码。

以deeplabv3为例，deeplabv3.py是完整的训练代码，deeplabv3_test.py是测试代码。

如果你克隆了代码，请注意使用正确数据集、模型权重文件的路径。

在我的训练过程中，deeplabv3和segformer-2b的效果还可以，unet效果略显糟糕。

 cityscapes数据集地址：https://www.cityscapes-dataset.com/


# English
This repository contains code for training UNet, Segformer-2b, and DeepLabv3 networks to perform semantic segmentation of cars using the Cityscapes dataset. Taking DeepLabv3 as an example, deeplabv3.py is the complete training code, and deeplabv3_test.py is the testing code.

If you clone the code, please note to use the correct paths for the dataset and model weight files. During my training process, DeepLabv3 and Segformer-2b showed acceptable performance, while the effect of UNet was slightly poor.

Cityscapes dataset address: https://www.cityscapes-dataset.com/
