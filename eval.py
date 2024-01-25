from lightning import Trainer

from dataset.hmap_dataset import HMapDataModule
from model.hmap_model import HMapLitModel
from utils import load_configs, load_pl_model


def main(exp_name, model_path=None):
    """
    This is the main function to start evaluation.

    Args:
        exp_name: experiment name
        model_path: the path of trained model

    Returns: None

    """

    # Load experiment configurations
    configs = load_configs(f'configs/{exp_name}.yaml')

    # Set up data module
    datamodule = HMapDataModule(**configs['data'])
    datamodule.setup()

    # Initialize model
    hmap_model = load_pl_model(HMapLitModel, model_path, **configs['model'])

    # Evaluate model
    configs['trainer']['devices'] = 1
    trainer = Trainer(**configs['trainer'])
    trainer.test(hmap_model, datamodule=datamodule)


if __name__ == '__main__':
    # Start training with experiment name
    main('hmap-v2', "test/ckpt/hmap-v2-epoch=499-val_loss=3.828e-04.ckpt")
