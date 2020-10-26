import os
import torch
import torchvision
from torchvision.transforms import transforms

from class_to_id_lists import cifar_10_class_to_id_list_5
from datasets.ood import OodDataset
from ood_validation import get_validation_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using gpu: %s " % torch.cuda.is_available())


def test_basic(
    ood_dataset,
    net_architecture="ToyNet",
    train_name="toy_train_102501",
    class_to_id_list=cifar_10_class_to_id_list_5,
    temperature=100,
    epsilon=0.002,
    batch_size=25,
    num_epoch_validation=4,
):
    print(f"{net_architecture} | {train_name} | temperature: {temperature} | epsilon: {epsilon} | batch_size: {batch_size} | num_epoch: {num_epoch_validation}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    cifar_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    get_validation_metrics(
        cifar_dataset,
        ood_dataset,
        net_architecture=net_architecture,
        train_name=train_name,
        class_to_id_list=class_to_id_list,
        temperature=temperature,
        epsilon=epsilon,
        batch_size=batch_size,
        num_epoch=num_epoch_validation,
    )


if __name__ == "__main__":
    net_architecture = "ToyNet"
    train_name = "toy_train_102501"
    class_to_id_list = cifar_10_class_to_id_list_5

    temperatures = [100, 1000]
    epsilons = [0, 0.002]
    batch_size = 25
    num_epoch_validation = 1

    ood_datasets = [
        OodDataset(
            data_dir=os.path.join("data", "Imagenet_resize", "Imagenet_resize"), image_extension="jpg"
        ),
        OodDataset(
            data_dir=os.path.join("data", "LSUN_resize", "LSUN_resize"), image_extension="jpg"
        ),
        OodDataset(
            data_dir=os.path.join("data","gaussian_noise"), image_extension="jpg"
        ),
        OodDataset(
            data_dir=os.path.join("data", "uniform_noise"), image_extension="jpg"
        ),
    ]

    for ood_dataset in ood_datasets:
        test_basic(
            ood_dataset,
            net_architecture=net_architecture,
            train_name=train_name,
            class_to_id_list=cifar_10_class_to_id_list_5,
            temperature=100,
            epsilon=0.002,
            batch_size=batch_size,
            num_epoch_validation=num_epoch_validation,
        )


    print()
    print("## Params valdation ...")
    print()

    ood_dataset = OodDataset(
        data_dir=os.path.join("data", "iSUN", "iSUN_patches"), image_extension="jpeg"
    )

    for temperature in temperatures:
        for epsilon in epsilons:
            print()
            test_basic(
                ood_dataset,
                net_architecture=net_architecture,
                train_name=train_name,
                class_to_id_list=cifar_10_class_to_id_list_5,
                temperature=temperature,
                epsilon=epsilon,
                batch_size=batch_size,
                num_epoch_validation=num_epoch_validation,
            )
