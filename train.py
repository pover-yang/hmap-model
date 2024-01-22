import lightning as L
import torch

from dataset.hmap_dataset import HeatMapDataModule
from model.hmap_model import LitHMapModel
from utils import load_configs, set_callbacks


def main(exp_name, pretrained_path=None, resume_path=None):
	# Load experiment configurations from YAML file
	configs = load_configs(f'configs/{exp_name}.yaml')

	# Initialize data module and model
	datamodule = HeatMapDataModule(**configs['data'])
	datamodule.setup()
	hmap_model = LitHMapModel(
		**configs['model'], predict_dataloader=datamodule.predict_dataloader())

	# Load checkpoint if provided
	if pretrained_path is not None:
		pl_state_dict = torch.load(
			pretrained_path, map_location=torch.device('cpu'))
		hmap_model.load_state_dict(pl_state_dict['state_dict'])

	# Initialize trainer with callbacks
	callbacks = set_callbacks(exp_name)
	trainer = L.Trainer(**configs['trainer'],
	                    callbacks=callbacks,
	                    )

	# Start training
	trainer.fit(hmap_model,
	            datamodule=datamodule,
	            ckpt_path=resume_path
	            )


def profile_model():
	from thop import profile
	from thop import clever_format
	from model.unet import UNet

	model = UNet(in_channels=1, n_classes=3, inc_channels=16)
	in_tensor = torch.randn(1, 1, 960, 640)
	flops, params, ret_dict = profile(model, inputs=(in_tensor,), ret_layer_info=True)
	flops, params = clever_format([flops, params], "%.3f")
	ret_dict = {k: clever_format([v[0], v[1]], "%.3f") for k, v in ret_dict.items()}
	print("|{:-^15}|{:-^15}|{:-^15}|".format("Layer", "FLOPS", "Params"))
	print("|{:^15}|{:^15}|{:^15}|".format("Total", flops, params))
	for k, v in ret_dict.items():
		print("|{:^15}|{:^15}|{:^15}|".format(k, v[0], v[1]))


def export_onnx(exp_name, ckpt_path):
	configs = load_configs(f'configs/{exp_name}.yaml')

	dummy_input = torch.randn(1, 1, 400, 640)

	hmap_model = LitHMapModel(**configs['model'])
	pl_state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
	hmap_model.load_state_dict(pl_state_dict['state_dict'])
	hmap_model.to_onnx(ckpt_path.replace('.ckpt', '.onnx'), input_sample=dummy_input)


if __name__ == '__main__':
	# Start main function with experiment name
	# main('hmap-v3')

	# profile_model()

	export_onnx('hmap-v2', './test/ckpt/hmap-v2-epoch=499-val_loss=3.828e-04.ckpt')
