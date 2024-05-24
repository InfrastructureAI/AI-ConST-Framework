# [MAE]((https://arxiv.org/abs/2111.06377)) implementation

The implementation of MAE is adopted from [here](https://github.com/pengzhiliang/MAE-pytorch), appreciate their work!

## Extract features from spot patches
1. Crop the whole histology image to spot patches according to the spatial coordinates, and stack them in `npy` file. Please note that the size of cropped patches should be 224*224.

2. Run MAE to extract features
```bash
# Set the path to save images
OUTPUT_DIR='output/'
# Path to image for visualization
PATCHES_PATH='/path/to/cropped/patches.npy'
# Path to pretrain model
MODEL_PATH='/path/to/pretrain/checkpoint.pth'


python run_mae_extract_feature.py ${PA