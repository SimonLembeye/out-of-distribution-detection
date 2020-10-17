import torch
import torch.optim as optim
import os

from datasets.cifar10 import Cifar10Dataset
from loss import margin_loss
from models.toy_net import ToyNet
from trainers.cifar_trainer import Cifar10Trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using gpu: %s " % torch.cuda.is_available())

if __name__ == "__main__":
    train_dataset = Cifar10Dataset(
        data_dir=os.path.join("data", "cifar-10", "train"),
        id_class_list=["airplane", "automobile", "bird", "cat"],
        ood_class_list=["truck", "ship"],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4
    )

    validation_dataset = Cifar10Dataset(
        data_dir=os.path.join("data", "cifar-10", "test"),
        id_class_list=["airplane", "automobile", "bird", "cat"],
        ood_class_list=["truck", "ship"],
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=32, shuffle=True, num_workers=4
    )

    net = ToyNet(class_nb=4).to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    trainer = Cifar10Trainer(
        dataloader=[train_loader, validation_loader],
        net=net,
        loss=margin_loss,
        optimizer=optimizer,
        device=device,
    )
    trainer.train()
