import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.Linear = nn.Linear(1, 1)
        self.net = nn.Sequential(self.Linear)
    
    def forward(self, x):
        return self.net(x)

def CreateDataSet(x, i=0):
    y = torch.zeros(x.size())
    for t in x:
        y[i] = t * 2 + 1 + random.randint(-10, 10) / 10.0
        i += 1
        
    return y

x = torch.Tensor([[1],[2],[3],[4],[5],[6],[7],[8]])
y = CreateDataSet(x) #torch.Tensor([[3.7], [5.6], [8.0], [9.1], [11.9], [13.4], [15.9], [17.3]])
#print(y)

model = LinearModel()

loss_func = torch.nn.MSELoss(reduce='sum')

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epoch_times = 20

epoch_list = []
loss_list = []
#weight_list = []


for i in range(epoch_times):
    y_pred = model(x)
    loss = loss_func(y, y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_list.append(i)
    loss_list.append(loss.item())
    #weight_list.append(model.Linear.weight.item())

print("weight = ", model.Linear.weight.item())
print("bias = ", model.Linear.bias.item())

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch times')
plt.ylabel('loss')
plt.title('SGD')
plt.show()
