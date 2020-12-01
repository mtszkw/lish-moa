import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from loss_functions import SmoothBCEwLogits


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


class MoANeuralNetwork(pl.LightningModule):
    def __init__(self, input_size: int, output_size: int, cfg: argparse.Namespace):
        super().__init__()
        self.cfg = cfg
        self.net = self.get_model(input_size, output_size)
        self.net.apply(weights_init_uniform_rule)

    def get_model(self, input_size, output_size):        
        return nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.utils.weight_norm(nn.Linear(input_size, cfg.n_units)),
            nn.ReLU(),
            
            nn.BatchNorm1d(cfg.n_units),
            nn.Dropout(p=0.4),
            nn.utils.weight_norm(nn.Linear(cfg.n_units, cfg.n_units)),
            nn.ReLU(),
            
            nn.BatchNorm1d(cfg.n_units),
            nn.Dropout(p=0.3),
            nn.utils.weight_norm(nn.Linear(cfg.n_units, output_size)))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam( self.net.parameters(), lr=1e-2, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=cfg.lr_sched_factor, patience=10, verbose=False)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = SmoothBCEwLogits(smoothing=0.001)(y_hat, y)

        self.log_dict({'train/loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = torch.nn.BCEWithLogitsLoss()(y_hat, y)

        self.log_dict({'val/loss': val_loss})
        return val_loss