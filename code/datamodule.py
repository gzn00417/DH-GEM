import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split


class TalentTrendDataModule(pl.LightningDataModule):

    def __init__(self, joint_dataset, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore='joint_dataset')
        self.dataset = joint_dataset
        n = len(self.dataset)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [int(0.7 * n), n - int(0.7 * n)], generator=torch.Generator().manual_seed(6081))
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.train_batch_size, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.val_batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.test_batch_size, pin_memory=True)
