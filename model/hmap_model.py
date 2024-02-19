import cv2
import torch
from lightning import LightningModule

from model.loss import FocalLoss
from model.unet import UNet
from utils import visualize_batch_hmaps, warmup_lr


class HMapLitModel(LightningModule):
    def __init__(self, init_lr, gamma, alpha, **model_conf):
        super().__init__()
        self.generator = UNet(**model_conf)
        self.loss = FocalLoss(gamma, alpha)

        self.init_lr = float(init_lr)
        self.predict_dataloader = None
        self.save_hyperparameters(ignore=['predict_dataloader'])

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        batch_in_tensor, batch_gt_tensor = batch
        batch_hmap_tensor = self(batch_in_tensor)
        loss = self.loss(batch_hmap_tensor, batch_gt_tensor)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_in_tensor, batch_gt_tensor = batch
        batch_hmap_tensor = self(batch_in_tensor)
        loss = self.loss(batch_hmap_tensor, batch_gt_tensor)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)

        # calculate precision and recall
        batch_hmap_tensor = torch.sigmoid(batch_hmap_tensor)
        tp, fp, fn = 0, 0, 0
        for hmap_tensor, gt_tensor in zip(batch_hmap_tensor, batch_gt_tensor):
            hmap_bboxes = hmap_to_bboxes(hmap_tensor)
            gt_bboxes = hmap_to_bboxes(gt_tensor)
            tp_, fp_, fn_ = calc_confusion_matrix(hmap_bboxes, gt_bboxes)
            tp += tp_
            fp += fp_
            fn += fn_
        self.test_step_outputs.append(torch.tensor([tp, fp, fn], dtype=torch.float32))
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
        lr_lambda = warmup_lr(max_epochs=self.trainer.max_epochs, warmup_factor=0.1)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [lr_scheduler]

    def set_predict_dataloader(self, dataloader):
        self.predict_dataloader = dataloader
