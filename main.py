import torch
import os

from datasets.cifar10 import Cifar10Dataset
from loss import margin_loss
from models.toy_net import ToyNet
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    dataset = Cifar10Dataset(
        data_dir=os.path.join("data", "cifar-10", "train"),
        id_class_list=["airplane", "automobile", "bird", "cat"],
        ood_class_list=["truck", "ship"],
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=2
    )

    net = ToyNet(class_nb=4)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for e in range(100):

        e_loss = 0

        for i, data in enumerate(loader):
            id_images, ood_images, id_labels, ood_labels = (
                data["id_images"],
                data["ood_images"],
                data["id_labels"],
                data["ood_labels"],
            )

            id_outputs = []
            ood_outputs = []

            for j in range(len(id_labels)):
                outputs = net(id_images[:, j, :, :, :])
                id_outputs.append(outputs)

            for j in range(ood_images.size()[1]):
                outputs = net(ood_images[:, j, :, :, :])
                ood_outputs.append(outputs)

            optimizer.zero_grad()
            loss = margin_loss((id_outputs, ood_outputs), id_labels)
            loss.backward()
            optimizer.step()

            e_loss += loss.item()

        print(e_loss)

