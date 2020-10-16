import torch
import os

from datasets.cifar10 import Cifar10Dataset
from datasets.tiny_imagenet import TinyImagenetDataset
from loss import margin_loss
from models.toy_net import ToyNet
import torch.optim as optim
from trainers.cifar_trainer import Cifar10Trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using gpu: %s " % torch.cuda.is_available())

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

        dataset = Cifar10Dataset(
            data_dir=os.path.join("data", "cifar-10", "train"),
            id_class_list=id_list,
            ood_class_list=ood_list,
            class_to_id=class_to_id,
        )

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=256, shuffle=True, num_workers=3
        )

        self.net = ToyNet(class_nb=8).to(device)
        optimizer = optim.SGD(self.net.parameters(), lr=0.0005, momentum=0.9)
        self.trainer = Cifar10Trainer(
            dataloader=loader,
            net=self.net,
            loss=margin_loss,
            optimizer=optimizer,
            device=device,
            max_epoch=10,
        )


def validation(classifiers):
    dataset = TinyImagenetDataset(
        data_dir=os.path.join("data", "tiny-imagenet-200", "val", "images"),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=3, shuffle=True, num_workers=4
    )

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

    for i, data in enumerate(loader, 0):

        clf = classifiers[0]
        net = clf.net
        out = net(data.to(device)) # softmax function needs to be added



        for res in out:
            for j in range(len(res)):
                # print(res)
                scores[clf.id_to_class[j]] += res[j].item()

        print(scores)


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

    validation(classifiers)

    """

    for classifier in classifiers:
        print()
        print("## !")
        classifier.trainer.train()


    """
