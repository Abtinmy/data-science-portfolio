import os
import cv2
import random
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, root, train: bool = True):
        self.root = root
        coef = 10 if train else 1
        self.gp1 = [x for x in os.listdir(self.root) if int(x[:-4]) < 45 * coef]
        self.gp2 = [x for x in os.listdir(self.root) if int(x[:-4]) >= 45 * coef]

    def __len__(self):
        return len(os.listdir(self.root))

    def __getitem__(self, idx):
        image_gp1 = random.sample(self.gp1, 1)
        image_gp2 = random.sample(self.gp2, 1)

        img1 = cv2.imread(self.root + image_gp1[0])
        img2 = cv2.imread(self.root + image_gp2[0])
        avg_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)

        transformer = transforms.ToTensor()

        return transformer(avg_img), transformer(img1), transformer(img2)
