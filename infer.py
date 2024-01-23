import glob

import cv2
import matplotlib.pyplot as plt
import torch

from dataset.hmap_transform import HeatMapTransform
from model.hmap_model import HMapLitModel
from utils import load_configs, load_pl_model, visualize_single_hmap


def infer_single_image(image_path, hmap_model):
    """
    Inference a single image.

    Args:
        image_path: the path of image
        hmap_model: the heatmap model

    Returns: heatmap tensor

    """
    # preprocess
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hmap_transform = HeatMapTransform(
        input_size=(400, 640), img_aug=False, geo_aug=False)
    in_tensor, image_tensor = hmap_transform(image)
    in_tensor = in_tensor.unsqueeze(0)
    image_tensor = image_tensor.unsqueeze(0)

    # inference
    hmap_tensor = hmap_model(in_tensor)
    hmap_tensor = torch.sigmoid(hmap_tensor)

    # visualize
    visualize_single_hmap(hmap_tensor, image_tensor)
    plt.show()

    return hmap_tensor


def infer_batch_images():
    """
    Inference a batch of images.
    """
    raise NotImplementedError


def main():
    configs = load_configs(f'configs/hmap-v2.yaml')
    model_path = "./test/ckpt/hmap-v2-epoch=499-val_loss=3.828e-04.ckpt"

    hmap_model = load_pl_model(HMapLitModel, model_path, **configs['model'])
    hmap_model.eval()

    image_paths = glob.glob('./test/image/test4.png')
    for f in image_paths:
        infer_single_image(f, hmap_model)


if __name__ == '__main__':
    main()
