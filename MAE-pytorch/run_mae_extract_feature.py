# --------------------------------------------------------
# Modified from https://github.com/pengzhiliang/MAE-pytorch
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from PIL import Image
import cv2

from timm.models import create_model
from datasets import DataAugmentationForMAE
import modeling_pretrain
from modeling_pretrain import pretrain_mae_base_patch16_224


def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    parser.add_argument('img_path', type=str, help='