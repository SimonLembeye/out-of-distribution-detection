import torch
import os

from class_to_id_lists import cifar_10_class_to_id_list_5
from classifier import Classifier
from datasets.tiny_imagenet import TinyImagenetDataset
from models.dense_net import DenseNet
from models.toy_net import ToyNet
from models.wide_res_net import WideResNet

from models.wideresnet import WideResNetFb
from ood_validation import validation
import torch.nn as nn
from torchvision.transforms import transforms
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using gpu: %s " % torch.cuda.is_available())

soft_max = nn.Softmax(dim=0)

# NET = ToyNet(class_nb=8).to(device)
# NET = DenseNet(num_classes=8, depth=50).to(device)
NET = WideResNet(8).to(device)
# NET = WideResNetFb(8).to(device)

if __name__ == "__main__":

    classifiers = [
        Classifier(
            NET, class_to_id=cifar_10_class_to_id_list_5[k], train_name="wide_train_102501", id=k
        )
        for k in range(len(cifar_10_class_to_id_list_5))
    ]

    for _ in range(200):
        print()
        for classifier in classifiers:
            print()
            print(f"## Train classifier {classifier.id}!")
            trainer = classifier.get_and_update_current_trainer()

            trainer.train()
            torch.save(trainer.net.state_dict(), classifier.best_weights_path)

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
        ood_scores_cifar = validation(NET, classifiers, cifar_dataset, w_label=True)

        print()
        print("Validation Tinyimagenet")
        tiny_dataset = TinyImagenetDataset(
            data_dir=os.path.join("data", "tiny-imagenet-200", "val", "images"),
        )
        ood_scores_tiny = validation(NET, classifiers, tiny_dataset)










