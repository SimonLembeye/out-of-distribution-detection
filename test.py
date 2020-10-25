import os

import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms

from class_to_id_lists import cifar_10_class_to_id_list_5
from classifier import Classifier
from datasets.tiny_imagenet import TinyImagenetDataset
from metrics import detection_error, fpr95, auroc
from models.dense_net import DenseNet
from models.toy_net import ToyNet
from models.wide_res_net import WideResNet
from models.wideresnet import WideResNetFb
from ood_validation import validation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using gpu: %s " % torch.cuda.is_available())

soft_max = torch.nn.Softmax(dim=0)

NET = ToyNet(class_nb=8).to(device)
# NET = DenseNet(num_classes=8, depth=50).to(device)
# NET = WideResNet(8).to(device)
# NET = WideResNetFb(8).to(device)

if __name__ == "__main__":

    classifiers = [
        Classifier(
            NET, class_to_id=cifar_10_class_to_id_list_5[k], train_name="toy_train_102501", id=k
        )
        for k in range(len(cifar_10_class_to_id_list_5))
    ]


    print()
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
    ood_scores_cifar = np.array(validation(NET, classifiers, cifar_dataset, w_label=True))

    print()
    print("Validation Tinyimagenet")
    tiny_dataset = TinyImagenetDataset(
        data_dir=os.path.join("data", "tiny-imagenet-200", "val", "images"),
    )
    ood_scores_tiny = np.array(validation(NET, classifiers, tiny_dataset))

    labels = np.concatenate((np.ones_like(ood_scores_cifar), np.zeros_like(ood_scores_tiny)), axis=0)
    ood_scores = np.concatenate((ood_scores_cifar, ood_scores_tiny), axis=0)

    print("detection_error", detection_error(labels, ood_scores))
    print("fpr95", fpr95(labels, ood_scores))
    print("auroc", auroc(labels, ood_scores))


