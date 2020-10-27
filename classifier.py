import os
import pathlib

import torch
from torch import optim as optim

from datasets.cifar10 import Cifar10Dataset
from loss import margin_loss
from trainers.cifar_trainer import Cifar10Trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    def __init__(
        self,
        net_architecture,
        train_name="toy_train",
        id=0,
        class_to_id={},
        batch_size=20,
        learning_rate=0.05,
        momentum=0.9,
        weight_decay=0.005,
    ):
        self.net = net_architecture
        ood_list = []
        id_list = []
        self.id_to_class = {}
        for class_name in CLASSES:
            if class_name in class_to_id.keys():
                id_list.append(class_name)
                self.id_to_class[class_to_id[class_name]] = class_name
            else:
                ood_list.append(class_name)

        dir_path = os.path.join("train_models_weights", train_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.id = id
        self.train_name = train_name
        self.best_weights_path = os.path.join(pathlib.Path(__file__).parent.absolute(), os.path.join(
            "train_models_weights", train_name, f"{id}.pth"
        ))

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.train_dataset = Cifar10Dataset(
            data_dir=os.path.join("data", "cifar-10", "train"),
            id_class_list=id_list,
            ood_class_list=ood_list,
            class_to_id=class_to_id,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=3
        )

        self.validation_dataset = Cifar10Dataset(
            data_dir=os.path.join("data", "cifar-10", "test"),
            id_class_list=id_list,
            ood_class_list=ood_list,
            class_to_id=class_to_id,
        )

        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset, batch_size=batch_size, shuffle=True, num_workers=3
        )

    def get_and_update_current_trainer(self):

        if os.path.exists(self.best_weights_path):
            self.net.load_state_dict(torch.load(self.best_weights_path))

        self.net.train()

        optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.weight_decay,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        trainer = Cifar10Trainer(
            dataloader=[self.train_loader, self.validation_loader],
            net=self.net,
            loss=margin_loss,
            optimizer=optimizer,
            device=device,
            max_epoch=1,
        )
        self.trainer = trainer
        return trainer
