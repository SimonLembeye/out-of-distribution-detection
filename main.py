import torch
import os

from datasets.cifar10 import Cifar10Dataset
from datasets.tiny_imagenet import TinyImagenetDataset
from loss import margin_loss
from models.toy_net import ToyNet
import torch.optim as optim
from trainers.cifar_trainer import Cifar10Trainer
from torch.distributions import Categorical
import torch.nn as nn
from torchvision.transforms import transforms
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using gpu: %s " % torch.cuda.is_available())

soft_max = nn.Softmax(dim=0)

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class Classifier:
    def __init__(self, class_to_id={}):
        ood_list = []
        id_list = []
        self.id_to_class = {}
        for class_name in CLASSES:
            if class_name in class_to_id.keys():
                id_list.append(class_name)
                self.id_to_class[class_to_id[class_name]] = class_name
            else:
                ood_list.append(class_name)

        train_dataset = Cifar10Dataset(
            data_dir=os.path.join("data", "cifar-10", "train"),
            id_class_list=id_list,
            ood_class_list=ood_list,
            class_to_id=class_to_id,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=256, shuffle=True, num_workers=3
        )

        validation_dataset = Cifar10Dataset(
            data_dir=os.path.join("data", "cifar-10", "test"),
            id_class_list=id_list,
            ood_class_list=ood_list,
            class_to_id=class_to_id,
        )

        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=256, shuffle=True, num_workers=3
        )

        self.net = ToyNet(class_nb=8).to(device)
        optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        self.trainer = Cifar10Trainer(
            dataloader=[train_loader, validation_loader],
            net=self.net,
            loss=margin_loss,
            optimizer=optimizer,
            device=device,
            max_epoch=5,
        )


def validation(classifiers, dataset):

    batch_size = 256
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    ood_sum = 0
    image_counter = 0

    for i, data in enumerate(loader, 0):

        images, labels = data

        scores = {
            "airplane": 0,
            "automobile": 0,
            "bird": 0,
            "cat": 0,
            "deer": 0,
            "dog": 0,
            "frog": 0,
            "horse": 0,
            "ship": 0,
            "truck": 0,
        }
        ood_scores = [0 for _ in range(batch_size)]

        for j in range(len(classifiers)):

            clf = classifiers[j]
            net = clf.net
            out = net(images.to(device))  # softmax function needs to be added

            for k in range(len(out)):
                res = out[k]
                for j in range(len(res)):
                    scores[clf.id_to_class[j]] += res[j].item()

                image_counter += 1
                sm = soft_max(res)
                entropy = Categorical(probs=sm).entropy()
                # TO DO: Add gradient over cross entropy loss step
                # TO DO: Add temperature scaling
                ood_scores[k] += (torch.max(sm) - entropy).item()

        ood_sum += sum(ood_scores)

    print()
    print(image_counter)
    print(ood_sum)
    print(ood_sum / image_counter)


if __name__ == "__main__":
    class_to_id_list = [
        {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "frog": 6,
            "horse": 7,
        },
        {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "ship": 6,
            "truck": 7,
        },
        {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "frog": 4,
            "horse": 5,
            "ship": 6,
            "truck": 7,
        },
        {
            "airplane": 0,
            "automobile": 1,
            "deer": 2,
            "dog": 3,
            "frog": 4,
            "horse": 5,
            "ship": 6,
            "truck": 7,
        },
        {
            "bird": 0,
            "cat": 1,
            "deer": 2,
            "dog": 3,
            "frog": 4,
            "horse": 5,
            "ship": 6,
            "truck": 7,
        },
    ]

    classifiers = [
        Classifier(class_to_id=class_to_id) for class_to_id in class_to_id_list
    ]

    for _ in range(200):

        print("Validation CIFAR10")
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)
        validation(classifiers, cifar_dataset)

        print("Validation Tinyimagenet")
        tiny_dataset = TinyImagenetDataset(
            data_dir=os.path.join("data", "tiny-imagenet-200", "val", "images"),
        )
        validation(classifiers, tiny_dataset)

        for classifier in classifiers:
            print()
            print()
            print("## !")
            classifier.trainer.train()

