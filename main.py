import torch
import os

from datasets.cifar10 import Cifar10Dataset
from models.toy_net import ToyNet
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    dataset = Cifar10Dataset(
        data_dir=os.path.join("data", "cifar-10", "train"),
        id_class_list=["airplane", "automobile", "bird", "cat"],
        ood_class_list=["truck", "ship"],
    )

    print("Yes")

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

            for j in range(len(id_labels)):
                optimizer.zero_grad()
                outputs = net(id_images[:, j, :, :, :])
                # print(id_labels[j])
                # print(outputs)

                loss = loss_fn(outputs, id_labels[j])
                loss.backward()
                optimizer.step()

                e_loss += loss.item()

        print(e_loss)

