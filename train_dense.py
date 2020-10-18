import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms

from models.dense_net import *
from trainers.dense_trainer import DenseTrainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using gpu: %s ' % torch.cuda.is_available())

if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    # resnet50 = models.resnet50(pretrained=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = DenseNet(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.032, momentum=0.9)

    trainer = DenseTrainer(dataloader=trainloader, net=net, loss=criterion, optimizer=optimizer, device=device)
    trainer.train()

