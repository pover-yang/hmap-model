import cv2
import lightning as L
import torch

from model.loss import FocalLoss
from model.unet import UNet
from utils import visualize_batch_hmaps, warmup_lr


class LitHMapModel(L.LightningModule):
    def __init__(self, init_lr, gamma, alpha, predict_dataloader=None, **model_conf):
        super().__init__()
        self.generator = UNet(**model_conf)
        self.loss = FocalLoss(gamma, alpha)

        self.init_lr = float(init_lr)
        self.predict_dataloader = predict_dataloader
        self.save_hyperparameters(ignore=['predict_dataloader'])

    def forward(self, x):
        return self.generator.forward(x)

    def training_step(self, batch, batch_idx):
        ins_tensor, gts_tensor = batch
        hmaps_tensor = self.generator.forward(ins_tensor)
        loss = self.loss(hmaps_tensor, gts_tensor)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=False, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ins_tensor, gts_tensor = batch
        hmaps_tensor = self.generator.forward(ins_tensor)
        loss = self.loss(hmaps_tensor, gts_tensor)
        self.log('val_loss', loss, on_step=False,
                 on_epoch=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        self.inference()

    def inference(self):
        if self.predict_dataloader is None or self.logger is None:
            return
        batch = next(iter(self.predict_dataloader()))
        ins_tensor, imgs_tensor = batch
        ins_tensor = ins_tensor.to(self.device)
        hmaps_tensor = self.generator.forward(ins_tensor)
        hmaps_tensor = torch.sigmoid(hmaps_tensor)
        visualize_hmaps = visualize_batch_hmaps(hmaps_tensor, imgs_tensor)
        cv2.imwrite(
            f'{self.logger.log_dir}/hmap_{self.current_epoch}.png', visualize_hmaps)

    def configure_optimizers(self):
        lr_lambda = warmup_lr(max_epochs=self.trainer.max_epochs,
                              warmup_epochs=None, warmup_factor=0.1)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [lr_scheduler]
