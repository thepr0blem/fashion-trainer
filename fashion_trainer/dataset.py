from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class FashionDataset(Dataset):
    def __init__(
        self, data: pd.DataFrame, transform: Optional[transforms.Compose] = None
    ):

        self.fashion_MNIST = list(data.values)
        self.transform = transform

        label = []
        image = []

        for item in self.fashion_MNIST:
            label.append(item[0])
            image.append(item[1:])

        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape((-1, 28, 28, 1)).astype("float32")

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)


def get_label_name(label):
    output_mapping = {
        0: "T-shirt/Top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    label = label.item() if type(label) == torch.Tensor else label
    return output_mapping[label]
