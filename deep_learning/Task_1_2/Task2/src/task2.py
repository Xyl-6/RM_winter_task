import torch.optim
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from tqdm import *
from models import *

trans = torchvision.transforms.ToTensor()
train_data = torchvision.datasets.CIFAR10("/home/alee/Desktop/deep_learning/Datasets", train=True, transform=trans,
                                          download=True)
test_data = torchvision.datasets.CIFAR10("/home/alee/Desktop/deep_learning/Datasets", train=False, transform=trans,
                                         download=True)

batch_size = 256
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True)





my_nn = My_nn()
my_nn = my_nn.cuda()
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
learning_rate = 0.01
optimizer = torch.optim.SGD(my_nn.parameters(), lr=learning_rate)

epoch = 100

train_step = 0
for i in range(epoch):
    train_bar = tqdm(train_loader, desc=f"Train_Epoch[{i + 1}/{epoch}]")
    train_loss = 0
    for data in train_bar:
        images, targets = data
        images = images.cuda()
        targets = targets.cuda()
        outputs = my_nn(images)
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
            outputs = my_nn(images)
            loss = loss_fn(outputs, targets)
            acc = (outputs.argmax(1) == targets).sum().item()
            accuracy += acc

            test_bar.set_postfix(acc=accuracy / len(test_data))


