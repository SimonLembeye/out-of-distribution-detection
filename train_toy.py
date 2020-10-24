import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms

from models.dense_net import DenseNet
from models.toy_net import ToyNet
from models.wide_res_net import WideResNet
from models.wideresnet import WideResNetFb
from trainers.toy_trainer import ToyTrainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using gpu: %s " % torch.cuda.is_available())

if __name__ == "__main__":

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=20, shuffle=True, num_workers=2
    )

    validationset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    validationloader = torch.utils.data.DataLoader(
        validationset, batch_size=20, shuffle=True, num_workers=2
    )

    # resnet50 = models.resnet50(pretrained=True)

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # net = DenseNet(num_classes=10, depth=10).to(device)
    # net = ToyNet(class_nb=10).to(device)
    net = WideResNetFb(10).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    trainer = ToyTrainer(
        dataloader=[trainloader, validationloader],
        net=net,
        loss=criterion,
        optimizer=optimizer,
        device=device,
    )
    trainer.train()
