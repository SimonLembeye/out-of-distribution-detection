import torch
import os

from datasets.cifar10 import Cifar10Dataset
from datasets.tiny_imagenet import TinyImagenetDataset
from loss import margin_loss
from models.dense_net import DenseNet
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

# NET = ToyNet(class_nb=8).to(device)
NET = DenseNet(num_classes=8, depth=50).to(device)


class Classifier:
    def __init__(self, train_name="toy_train", id=0, class_to_id={}):
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
        self.best_weights_path = os.path.join(
            "train_models_weights", train_name, f"{id}.pth"
        )

        self.train_dataset = Cifar10Dataset(
            data_dir=os.path.join("data", "cifar-10", "train"),
            id_class_list=id_list,
            ood_class_list=ood_list,
            class_to_id=class_to_id,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=2, shuffle=True, num_workers=3
        )

        self.validation_dataset = Cifar10Dataset(
            data_dir=os.path.join("data", "cifar-10", "test"),
            id_class_list=id_list,
            ood_class_list=ood_list,
            class_to_id=class_to_id,
        )

        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset, batch_size=2, shuffle=True, num_workers=3
        )

    def get_and_update_current_trainer(self):

        if os.path.exists(self.best_weights_path):
            NET.load_state_dict(torch.load(self.best_weights_path))

        optimizer = optim.SGD(NET.parameters(), lr=0.005, momentum=0.9)
        trainer = Cifar10Trainer(
            dataloader=[self.train_loader, self.validation_loader],
            net=NET,
            loss=margin_loss,
            optimizer=optimizer,
            device=device,
            max_epoch=1,
        )
        self.trainer = trainer
        return trainer


def validation(classifiers, dataset):

    batch_size = 2
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    ood_sum = 0
    image_counter = 0

    temperature = 1

    ood_scores = [0 for _ in range(batch_size)]

    scores = {
        "airplane": [0 for _ in range(batch_size)],
        "automobile": [0 for _ in range(batch_size)],
        "bird": [0 for _ in range(batch_size)],
        "cat": [0 for _ in range(batch_size)],
        "deer": [0 for _ in range(batch_size)],
        "dog": [0 for _ in range(batch_size)],
        "frog": [0 for _ in range(batch_size)],
        "horse": [0 for _ in range(batch_size)],
        "ship": [0 for _ in range(batch_size)],
        "truck": [0 for _ in range(batch_size)],
    }

    for j in range(len(classifiers)):

        clf = classifiers[j]
        if os.path.exists(clf.best_weights_path):
            NET.load_state_dict(torch.load(clf.best_weights_path))

        image_counter = 0

        for i, data in enumerate(loader, 0):

            if i > 0:
                break

            images, labels = data

            out = NET(images.to(device))  # softmax function needs to be added

            for k in range(len(out)):
                res = out[k]
                for j in range(len(res)):
                    scores[clf.id_to_class[j]][k] += res[j].item()

                image_counter += 1

                # print()
                # print(res)
                # print(res * temperature)
                sm = soft_max(res * temperature)
                # print(sm)
                entropy = Categorical(probs=sm * temperature).entropy()
                # print(entropy)
                # TO DO: Add gradient over cross entropy loss step
                ood_scores[k] += (torch.max(sm) - entropy).item()


    # print(ood_scores)
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
        Classifier(
            class_to_id=class_to_id_list[k], train_name="dense_train_1021202002", id=k
        )
        for k in range(len(class_to_id_list))
    ]

    for _ in range(200):

        print("Validation CIFAR10")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        cifar_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        validation(classifiers, cifar_dataset)

        print("Validation Tinyimagenet")
        tiny_dataset = TinyImagenetDataset(
            data_dir=os.path.join("data", "tiny-imagenet-200", "mini_val", "images"),
        )
        validation(classifiers, tiny_dataset)

        print()

        for classifier in classifiers:
            print()
            print(f"## Train classifier {classifier.id}!")
            trainer = classifier.get_and_update_current_trainer()
            trainer.train()
            torch.save(trainer.net.state_dict(), classifier.best_weights_path)










