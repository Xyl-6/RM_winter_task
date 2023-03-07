import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import *

trans = torchvision.transforms.ToTensor()
train_data = torchvision.datasets.CIFAR10("/home/alee/Desktop/deep_learning/Datasets", train=True, transform=trans,
                                          download=True)
test_data = torchvision.datasets.CIFAR10("/home/alee/Desktop/deep_learning/Datasets", train=False, transform=trans,
                                         download=True)

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

vgg_block = torchvision.models.vgg16()
# print(vgg_block)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(vgg_block, nn.ReLU(), nn.Linear(1000, 10))

    def forward(self, x):
        x = self.model(x)
        return x


net = Net()
net = net.cuda()
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
lr = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

epochs = 100

for epoch in range(epochs):
    train_bar = tqdm(train_loader, desc=f"train_[{epoch+1}/{epochs}]")
    for data in train_bar:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc=f"test_[{epoch+1}/{epochs}]")
        test_acc = 0
        test_step = 0
        for data in test_bar:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            test_step += 1
            acc = (outputs.argmax(1) == targets).sum().item()
            test_acc = test_acc + acc
            test_bar.set_postfix(acc=test_acc/len(test_data))







