import glob

import cv2
import matplotlib.pyplot as plt
import torch

from dataset.hmap_transform import HeatMapTransform
from model.hmap_model import LitHMapModel
from utils import load_configs, visualize_single_hmap


# Inference on a single image
def infer_single_image(image_path):
    # preprocess
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    in_tensor, image_tensor = hmap_transform(image)
    in_tensor = in_tensor.unsqueeze(0)
    image_tensor = image_tensor.unsqueeze(0)

    # inference
    hmap_tensor = hmap_model(in_tensor)
    hmap_tensor = torch.sigmoid(hmap_tensor)

    # visualize
    fig = visualize_single_hmap(hmap_tensor, image_tensor)
    plt.show()


if __name__ == '__main__':
    configs = load_configs(f'configs/hmap-v2.yaml')
    model_path = "./test/ckpt/hmap-v2-epoch=499-val_loss=3.828e-04.ckpt"
    hmap_model = LitHMapModel(**configs['model'])
    pl_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    hmap_model.load_state_dict(pl_state_dict['state_dict'])
    hmap_model.eval()

    hmap_transform = HeatMapTransform(
        input_size=(400, 640), img_aug=False, geo_aug=False)

    image_paths = glob.glob('./test/image/test4.png')
    for f in image_paths:
        infer_single_image(f)
