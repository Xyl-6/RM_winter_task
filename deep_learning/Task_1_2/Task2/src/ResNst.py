import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import *

trans = torchvision.transforms.ToTensor()
train_data = torchvision.datasets.CIFAR10("/home/alee/Desktop/deep_learning/Datasets", train=True, transform=trans,
                                          download=True)
test_data = torchvision.datasets.CIFAR10("/home/alee/Desktop/deep_learning/Datasets", train=False, transform=trans,
                                         download=True)

batch_size = 256
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, use_resnet):
        super().__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        if use_resnet:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.Conv3 = None

        # 定义批标准化层
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y+X)


net = ResNet()
net = net.cuda()

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
lr = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

epochs = 100






















