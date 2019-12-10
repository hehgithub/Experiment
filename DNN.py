import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from read_data import *

x,y = readData('water_final.arff')


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(23,20)
        self.hidden2 = torch.nn.Linear(20,15)
        self.hidden3 = torch.nn.Linear(15, 10)
        self.out = torch.nn.Linear(10,8)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.log_softmax(self.out(x),dim=1)
        return x

net = Net()

optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
loss_func = torch.nn.NLLLoss()

plt.ion()

for t in range(100):
    out = net(x)
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()