from HEM.dataloaders.hem_dataset import HEMDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class HEMLoader(LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_set = None

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_set = HEMDataset(self.cfg)


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, drop_last=True)
