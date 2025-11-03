import pytorch_lightning as pl
import os
import sys
import pickle
import numpy as np

import kornia.metrics.psnr as PSNR
import kornia.metrics.ssim as SSIM
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from loss.L1 import L1_loss
from loss.Perceptual_our import PerceptualLoss, L_color

from argparse import Namespace
from data.dataloader import Haze4kdataset, Val4kdataset
from pytorch_lightning import seed_everything
from core.monitor import Monitor as mo

from core.network  import DIE
from core.DA_text import perception
from core.SIEM import process_batch

seed = 42
seed_everything(seed)

class CoolSystem(pl.LightningModule):

    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams
        # train/val/test datasets
        self.train_datasets = self.params.train_datasets
        self.train_batchsize = self.params.train_bs
        self.test_datasets = self.params.test_datasets
        self.test_batchsize = self.params.test_bs
        self.validation_datasets = self.params.val_datasets
        self.val_batchsize = self.params.val_bs


        # Train setting
        self.initlr = self.params.initlr
        self.weight_decay = self.params.weight_decay
        self.crop_size = self.params.crop_size
        self.num_workers = self.params.num_workers

        # loss_function
        self.loss_L1 = L1_loss()
        self.loss_Pe = PerceptualLoss()
        self.model = DIE(decoder=True)

        self.wwww = mo('./log')

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'weak_method' in state:
            del state['weak_method']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def forward(self, x, ref):

        Degradation_feature = perception(x, text_features, model)

        Tags_feature = process_batch(x, RAM, model, clip.tokenize,
                                       device)
        y = self.model(x, Degradation_feature, Tags_feature)

        return y


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.initlr, betas=[0.9, 0.999],
                                      weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.initlr, max_lr=1.5 * self.initlr,
                                                      cycle_momentum=False)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y, ref = batch
        x = x.to(self.device)
        y = y.to(self.device)
        ref = ref.to(self.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = torch.clip(x1, 0, 1)
        x2 = torch.clip(x2, 0, 1)

        y = torch.clip(y, 0, 1)
        y2 = self.forward(x1, ref)

        loss = self.loss_L1(y2, y) + 0.2 * self.loss_Pe(y2, y)

        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y, ref = batch
        y_hat = self.forward(x, ref)
        loss = self.loss_L1(y_hat, y) + 0.2 * self.loss_Pe(y_hat, y)

        ssim = SSIM(y_hat, y, 5).mean().item()
        psnr = PSNR(y_hat, y, 1).item()

        self.wwww.imageWriter(batch_idx, im=x.to('cpu'), tag='raw')
        self.wwww.imageWriter(batch_idx, im=y_hat.to('cpu'), tag='pred')
        self.wwww.imageWriter(batch_idx, im=y.to('cpu'), tag='gt')

        self.log('val_loss', loss)
        self.log('psnr', psnr)
        self.log('ssim', ssim)
        self.trainer.checkpoint_callback.best_model_score

        return {'val_loss': loss, 'psnr': psnr, 'ssim': ssim}

    def train_dataloader(self):
        # REQUIRED
        train_set = Haze4kdataset(self.train_datasets, train=True, size=self.crop_size)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.train_batchsize, shuffle=True,
                                                   num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_set = Val4kdataset(self.validation_datasets, train=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.val_batchsize, shuffle=False,
                                                 num_workers=self.num_workers)
        return val_loader


def main():
    RESUME = False
    resume_checkpoint_path = r''
    if len(sys.argv) > 1:
        device = [int(x) for x in str(sys.argv[1]).split(',')]
    else:
        print("No device argument provided. Using default device [0].")
        device = [6]
    print(device)
    args = {
        'epochs': 500,
        'train_datasets': r'/dataset/UIEBD/train',
        'test_datasets': None,
        # Randomly select 100 from the training set
        'val_datasets': r'/dataset/UIEBD/val',
        'train_bs':4, # 4
        'test_bs': 1,
        'val_bs': 8,
        'initlr': 0.0001,
        'weight_decay': 0.001,
        'crop_size': 256,
        'num_workers': 16,
        'model_blocks': 5,
        'chns': 64
    }

    hparams = Namespace(**args)
    model = CoolSystem(hparams)

    try:
        print("Model initialized successfully")
    except pickle.PicklingError as e:
        print(f"Pickling error: {e}")

    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        monitor='psnr',
        filename='epoch{epoch:02d}-psnr{psnr:.3f}-ssim{ssim:.3f}-train_loss{train_loss:.3f}-val_loss{val_loss:.3f}',
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_top_k=3,
        mode="max"
    )

    if RESUME:
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            devices=[0],
            resume_from_checkpoint=resume_checkpoint_path,
            logger=logger,
            precision=16,
            callbacks=[checkpoint_callback],

        )
    else:
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            devices=[0],
            logger=logger,
            precision=16,
            callbacks=[checkpoint_callback],
        )

    trainer.fit(model)


if __name__ == '__main__':
    main()
