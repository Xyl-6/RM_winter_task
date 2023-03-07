
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import *

trans = torchvision.transforms.ToTensor()
train_data = torchvision.datasets.MNIST("/home/alee/Desktop/deep_learning/Datasets", train=True, transform=trans,
                                        download=True)
test_data = torchvision.datasets.MNIST("/home/alee/Desktop/deep_learning/Datasets", train=False, transform=trans,
                                       download=True)

batch_size = 256
train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)


class My_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(4*4*128, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


my_nn = My_nn()
my_nn = my_nn.cuda()
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
learning_rate = 0.1
optimizer = torch.optim.SGD(my_nn.parameters(), lr=learning_rate)

epoch = 30

for i in range(epoch):
    total_train_loss = 0
    train_bar = tqdm(train_loader, desc=f"Train_[{i+1}/{epoch}]")
    for imgs, targets in train_bar:
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = my_nn(imgs)
        loss = loss_fn(outputs, targets)
        total_train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print("第{}轮loss={}".format(i+1, total_train_loss))
    total_acc = 0
    test_step = 0
    test_loss = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc=f"Test_[{i+1}/{epoch}]")
        for imgs, targets in test_bar:
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = my_nn(imgs)
            loss = loss_fn(outputs, targets)
            acc = (outputs.argmax(1) == targets).sum()
            test_loss += loss
            total_acc += acc
            test_step += 1
            test_bar.set_postfix(acc = (total_acc/len(test_data)).item())


