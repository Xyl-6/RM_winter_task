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

batch_size = 256
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True)



class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(576, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.model(x)
        return x


leNet = LeNet()
leNet = leNet.cuda()

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

lr = 0.01
optimizer = torch.optim.SGD(leNet.parameters(), lr=lr)

epoch = 100
train_step = 0
for i in range(epoch):
    train_bar = tqdm(train_loader, desc=f"Train_Epoch[{i + 1}/{epoch}]")
    train_loss = 0
    for data in train_bar:
        images, targets = data
        images = images.cuda()
        targets = targets.cuda()
        outputs = leNet(images)
        # print(outputs.shape)
        loss = loss_fn(outputs, targets)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        accuracy = 0
        test_bar = tqdm(test_loader, desc=f"Test_Epoch[{i + 1}/{epoch}")
        for data in test_bar:
            images, targets = data
            images = images.cuda()
            targets = targets.cuda()
            outputs = LeNet(images)
            loss = loss_fn(outputs, targets)
            acc = (outputs.argmax(1) == targets).sum().item()
            accuracy += acc

            test_bar.set_postfix(acc=accuracy / len(test_data))
