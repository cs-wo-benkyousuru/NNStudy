import PIL
from PIL import Image
import torch  # 导入torch库
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

# 读取保存的神经网络模型和参数
model = torch.load("my_cnn.nn")


# 构造Dataset
class MyTestSet(Dataset):
    def __init__(self):
        self.transform = transforms.Compose([
            # transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.data = []
        self.labels = []
        for digit in range(10):
            for i in range(50):
                path = "../data/test_data/{}/{}.bmp".format(digit, i)
                img = Image.open(path)
                img = self.transform(img)
                self.data.append(torch.unsqueeze(img[3], dim=0))
                self.labels.append(digit)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


test_set = MyTestSet()
test_dataloader = DataLoader(test_set, batch_size=5, shuffle=False)

yes = 0
no = 0
for data, label in test_dataloader:
    out = model(data)
    for i in range(5):
        if (out[i].argmax() == label[i]):
            yes += 1
        else:
            no += 1
print(yes, no)
print(yes/(yes + no))
