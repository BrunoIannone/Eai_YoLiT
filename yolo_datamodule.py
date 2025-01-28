from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import yolo_dataset

class YoloDatamodule(LightningDataModule):
    def __init__(self,training_path,test_path):
        super().__init__()

        self.training_path = training_path
        self.test_path = test_path
