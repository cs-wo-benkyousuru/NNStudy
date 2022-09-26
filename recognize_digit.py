import torch.nn
import torchvision  # 导入所需要的包，这个包里包含了机器视觉有关的内容
from torch.utils.data import DataLoader  # 导入Dataloader
from model import CNNModel   # 导入构建好的模型，这里的RLS是我的ID的后三个字母
import matplotlib.pyplot as plt

# 训练数据集
train_data = torchvision.datasets.MNIST(
    root="data",    # 表示把MINST保存在data文件夹下
    download=True,  # 表示需要从网络上下载。下载过一次后，下一次就不会再重复下载了
    train=True,     # 表示这是训练数据集
    transform=torchvision.transforms.ToTensor()
                    # 要把数据集中的数据转换为pytorch能够使用的Tensor类型
)

# 测试数据集
test_data = torchvision.datasets.MNIST(
    root="data",    # 表示把MINST保存在data文件夹下
    download=True,  # 表示需要从网络上下载。下载过一次后，下一次就不会再重复下载了
    train=False,    # 表示这是测试数据集
    transform=torchvision.transforms.ToTensor()
                    # 要把数据集中的数据转换为pytorch能够使用的Tensor类型
)

# 创建两个Dataloader, 包尺寸为64
# 训练用的Dataloader
train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True)
# 测试用的Dataloader
test_dataloader = DataLoader(test_data, batch_size=256, shuffle=False)
#print(len(train_dataloader))
#len(train_dataloader) = 469, 约为60000/128，也就是说，dataloader的长度等于数据长度除以batch_size

# 实例化模型
model = CNNModel()

# 交叉熵损失函数
loss_func = torch.nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)    #parameters保存了权重和偏置量等参数

# 定义训练次数
cnt_epochs = 10  # 训练10个循环

#print(len(train_dataloader))
decent_list = []
loss_list = []
decent_num = 0
# 循环训练
for cnt in range(cnt_epochs):
    # 把训练集中的数据训练一遍
    # train_dataloader的长度是l=样本数/batch_size
    # 此处每一次循环都用batch_size的样本数进行训练，backward()里面隐含了一个for i in range batch_size:的循环
    for imgs, labels in train_dataloader:
        #此处的for循环，每次循环都取出train_dataloader的一个mini-batch中所有的imgs和labels，所以len(imgs)和len(labels)都为dataloader的batch_size  
        #print("length of imgs : ", len(imgs), "length of labels :", len(labels), end="\n") 
        #print(imgs.shape)                   # 输出torch.Size([batch_size, 1, 28, 28])
        #print(labels.shape)                 # 输出torch.Size([batch_size])
        outputs = model(imgs)                # outputs是根据输入得到的一系列预测值
        #print(outputs.shape)                # 输出torch.Size([batch_size, 10])
        loss = loss_func(outputs, labels)    # 由预测值和标签计算loss，loss是每一个(mini-)batch的loss
        #print("loss=", loss.item())
        optimizer.zero_grad()                # 注意清空优化器的梯度，防止累计
        loss.backward()                      # 对一个batch进行反向传播
        optimizer.step()                     # 由梯度更新一次参数
        decent_num += 1
        decent_list.append(decent_num)
        loss_list.append(loss.item())


    # 用测试集测试一下当前训练过的神经网络
    total_loss = 0                           # 保存这次测试总的loss
    with torch.no_grad():                    # 下面不需要反向传播，所以不需要自动求导
        for imgs, labels in test_dataloader:
            outputs = model(imgs)
            loss = loss_func(outputs, labels)
            total_loss += loss               # 累计误差
    #print("第{}次训练的Loss:{}".format(cnt + 1, total_loss))

plt.plot(decent_list, loss_list)
plt.xlabel("decent time")
plt.ylabel("loss")
plt.title("Oops")
plt.show()


# 保存训练的结果（包括模型和参数）
torch.save(model, "my_cnn.nn")
