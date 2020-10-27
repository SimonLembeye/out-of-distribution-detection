import torch
import os

from class_to_id_lists import cifar_10_class_to_id_list_5
from classifier import Classifier
from datasets.ood import OodDataset
from models.dense_net import DenseNet
from models.toy_net import ToyNet
from models.wide_res_net import WideResNet

from ood_validation import get_validation_metrics
import torch.nn as nn
from torchvision.transforms import transforms
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using gpu: %s " % torch.cuda.is_available())

soft_max = nn.Softmax(dim=0)


if __name__ == "__main__":

    net_architecture = "WideResNet"
    class_to_id_list = cifar_10_class_to_id_list_5
    train_name = "wide_train_102501"

    learning_rate = 0.05
    weight_decay = 0.0005
    momentum = 0.9
    batch_size = 25

    if net_architecture == "DenseNet":
        net = DenseNet(num_classes=8, depth=100).to(device)
    elif net_architecture == "WideResNet":
        net = WideResNet(8).to(device)
    else:
        net = ToyNet(class_nb=8).to(device)

    classifiers = [
        Classifier(
            net,
            class_to_id=class_to_id_list[k],
            train_name=train_name,
            id=k,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        for k in range(len(class_to_id_list))
    ]

    for _ in range(200):
        print()
        for classifier in classifiers:
            print()
            print(f"## Train classifier {classifier.id}!")
            print()
            trainer = classifier.get_and_update_current_trainer()
            trainer.train()
            torch.save(trainer.net.state_dict(), classifier.best_weights_path)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        cifar_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        ood_dataset = OodDataset(
            data_dir=os.path.join("data", "iSUN", "iSUN_patches"),
            image_extension="jpeg",
        )

        get_validation_metrics(
            cifar_dataset,
            ood_dataset,
            net_architecture=net_architecture,
            train_name=train_name,
            class_to_id_list=class_to_id_list,
            temperature=100,
            epsilon=0.002,
            batch_size=25,
            num_epoch=25,
        )
