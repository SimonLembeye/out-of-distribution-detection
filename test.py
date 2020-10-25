import os
import torch
import torchvision
from torchvision.transforms import transforms

from class_to_id_lists import cifar_10_class_to_id_list_5
from datasets.tiny_imagenet import TinyImagenetDataset
from ood_validation import get_validation_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using gpu: %s " % torch.cuda.is_available())

soft_max = torch.nn.Softmax(dim=0)

if __name__ == "__main__":
    net_architecture = "ToyNet"
    train_name = "toy_train_102501"
    class_to_id_list = cifar_10_class_to_id_list_5

    temperature = 100
    epsilon = 0.002
    batch_size = 25
    num_epoch_validation = 50

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    cifar_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    tiny_dataset = TinyImagenetDataset(
        data_dir=os.path.join("data", "tiny-imagenet-200", "val", "images"),
    )

    get_validation_metrics(
        cifar_dataset,
        tiny_dataset,
        net_architecture=net_architecture,
        train_name=train_name,
        class_to_id_list=class_to_id_list,
        temperature=temperature,
        epsilon=epsilon,
        batch_size=batch_size,
        num_epoch=num_epoch_validation,
    )



