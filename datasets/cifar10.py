import os
import cv2
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset


class Cifar10Dataset(Dataset):
    def __init__(self, data_dir, id_class_list, ood_class_list, class_to_id = {}, transform=None):
        self.data_dir = data_dir
        self.id_class_list = id_class_list
        self.ood_class_list = ood_class_list
        self.class_to_id = {}
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

    def __len__(self):
        return len(os.listdir(os.path.join(self.data_dir, self.id_class_list[0])))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx += 1  # image names go from 1 to 5000

        if not len(self.class_to_id):
            for i in range(len(self.id_class_list)):
                self.class_to_id[self.id_class_list[i]] = i

        id_images = []
        id_labels = []

        for class_name in self.id_class_list:
            image_path = os.path.join(
                self.data_dir, class_name, f"{str(idx).zfill(4)}.png"
            )
            id_labels.append(self.class_to_id[class_name])
            image = cv2.imread(image_path)
            image = self.transform(image) if self.transform else image
            id_images.append(image)

        ood_images = []
        ood_labels = []

        for class_name in self.ood_class_list:
            image_path = os.path.join(
                self.data_dir, class_name, f"{str(idx).zfill(4)}.png"
            )
            ood_labels.append(class_name)
            image = cv2.imread(image_path)
            image = self.transform(image) if self.transform else image
            ood_images.append(image)

        sample = {
            "id_images": torch.stack(id_images),
            "ood_images": torch.stack(ood_images),
            "id_labels": id_labels,
            "ood_labels": ood_labels,
        }

        return sample
