import os
import cv2
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset


class OodDataset(Dataset):
    def __init__(self, data_dir, image_extension="jpg", transform=None):
        self.data_dir = data_dir
        self.image_extension = image_extension
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
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.data_dir, f"{idx}.{self.image_extension}")
        image = cv2.resize(cv2.imread(image_path), (32, 32))
        image = self.transform(image)

        return image, "label" # read label in txt file if needed