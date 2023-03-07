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


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 10),
            nn.Softmax()
        )

    def forward(self, x):
        # x = x.reshape(1, 1, 224, 224)
        x = self.model(x)
        return x


# 查看每层形状
net = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 10),
            nn.Softmax())
X = torch.randn(size=(1, 1, 224, 224), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

alexNet = AlexNet()
alexNet = alexNet.cuda()
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

lr = 0.01
optimizer = torch.optim.SGD(alexNet.parameters(), lr=lr)

epoch = 100
train_step = 0
for i in range(epoch):
    train_bar = tqdm(train_loader, desc=f"Train_Epoch[{i + 1}/{epoch}]")
    train_loss = 0
    for data in train_bar:
        images, targets = data
        images = images.cuda()
        targets = targets.cuda()
        outputs = alexNet(images)
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
            outputs = alexNet(images)
            loss = loss_fn(outputs, targets)
            acc = (outputs.argmax(1) == targets).sum().item()
            accuracy += acc

            test_bar.set_postfix(acc=accuracy / len(test_data))
