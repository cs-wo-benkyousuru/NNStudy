import torch
import torch.nn as nn


# 定义卷积神经网络类
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,   # 输入、输出通道数，输出通道数可以理解为提取了几种特征
            #关于in_channels和out_channels，参考 : https://blog.csdn.net/qq_51533157/article/details/122789563
                      kernel_size=(3, 3),               # 卷积核尺寸
                      stride=(1, 1),                    # 卷积核每次移动多少个像素
                      padding=1),                       # 原图片边缘加几个空白像素
                                                        # 输入图片尺寸为 1×28×28，若为RGB则in_channels = 3
                                                        # 第一次卷积，尺寸为                 16×28×28
            nn.MaxPool2d(kernel_size=2),                # 第一次池化，尺寸为                 16×14×14
            nn.Conv2d(16, 32, 3, 1, 1),                 # 第二次卷积，尺寸为                 32×14×14
            nn.MaxPool2d(2),                            # 第二次池化，尺寸为                 32×7 ×7
                                                        # 第二次池化后输出32个7×7的矩阵
            nn.Flatten(),                               # 
            nn.Linear(32*7*7, 16),                      # in_features和out_features, 全连接层一共有16个神经元
            nn.ReLU(),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        return self.net(x)
