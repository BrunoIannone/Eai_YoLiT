from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2


class YoLoDataset(Dataset):
    def __init__(self, samples, stage):
        super().__init__()

        self.samples = samples
        self.stage = stage
        self.bbox_labels = ["face", "l_eye", "r_eye"]
        self.transform = A.Compose([
            A.RandomCrop(width=450, height=450),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2()

        ], bbox_params=A.BboxParams(format='coco', label_fields=['bbox_labels']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]
        image = sample["img"]
        bbox = sample["bbox"]
        labels = sample["labels"]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(
            image=image, bboxes=bbox, bbox_labels=self.bbox_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']

        return transformed_image, transformed_bboxes, labels
