import torch

from model.hmap_model import LitHMapModel
from .misc import load_configs


def export_onnx(exp_name, ckpt_path):
    configs = load_configs(f'configs/{exp_name}.yaml')

    dummy_input = torch.randn(1, 1, 400, 640)

    hmap_model = LitHMapModel(**configs['model'])
    pl_state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    hmap_model.load_state_dict(pl_state_dict['state_dict'])
    hmap_model.to_onnx(ckpt_path.replace('.ckpt', '.onnx'), input_sample=dummy_input)
