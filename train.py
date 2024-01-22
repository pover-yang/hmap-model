import lightning as L
import torch

from dataset.hmap_dataset import HeatMapDataModule
from model.hmap_model import LitHMapModel
from utils import load_configs, set_callbacks


def main(exp_name, pretrained_path=None, resume_path=None):
	"""
	This is the main function to start training.

	Args:
		exp_name: experiment name
		pretrained_path: the path of pretrained model
		resume_path: the path of checkpoint to resume training

	Returns: None

	"""

	# Load experiment configurations
	configs = load_configs(f'configs/{exp_name}.yaml')

	# Set up data module
	datamodule = HeatMapDataModule(**configs['data'])
	datamodule.setup()

	# Initialize model
	hmap_model = LitHMapModel(**configs['model'], predict_dataloader=datamodule.predict_dataloader())

	# Load checkpoint if provided
	if pretrained_path is not None:
		pl_state_dict = torch.load(
			pretrained_path, map_location=torch.device('cpu'))
		hmap_model.load_state_dict(pl_state_dict['state_dict'])

	# Initialize trainer with callbacks
	callbacks = set_callbacks(exp_name)
	trainer = L.Trainer(**configs['trainer'], callbacks=callbacks)

	# Start training
	trainer.fit(hmap_model, datamodule=datamodule, ckpt_path=resume_path)


if __name__ == '__main__':
	# Start training with experiment name
	main('hmap-v3')
