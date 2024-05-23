# [MAE]((https://arxiv.org/abs/2111.06377)) implementation

The implementation of MAE is adopted from [here](https://github.com/pengzhiliang/MAE-pytorch), appreciate their work!

## Extract features from spot patches
1. Crop the whole histology image to spot patches according to the spatial coordinates, and stack them in `npy` file. Please note that the size