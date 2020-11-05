# [conST: an interpretable multi-modal contrastive learning framework for spatial transcriptomics](https://www.biorxiv.org/content/10.1101/2022.01.14.476408v1)
We are actively updating this repository! More features/examples/experiments coming soon!

## Overview

![Framework overview](imgs/conST-workflow.png)
Framework of conST. conST models the ST data as a graph by treating gene expression and
morphology as node attributes and constructing edges by spatial coordinates. The training is divided into
two stages: pretraining and major training stage. Pretraining stage initializes the weights of the encoder E
by reconstruction loss. In major training stage, data augmentation is applied and then contrastive learning
in three levels, i.e., local-local, local-global, local-context, are used to learn a low-dimensional embedding by
minimize or maximize the mutual information (MI) between different embeddings. The learned embedding
can be used for various downstream tasks, which, when analyzed together, can shed light on the widely
concerned tumour microenvironment and cell-to-cell interaction. GNNExplainer helps to provide more con-
vincing predictions with interpretability.

## Dependencies
- Python=3.7.11
- torch=1.8.0
- torchvision=0.9.2
- torch-geometric=2.0.1
- torch-scatter=2.0.8
- torch-sparse=0.6.12
- scikit-learn=0.24.2
- umap-learn=0.5.1
- scanpy=1.8.1
- seaborn=0.11.2
- scipy=1.7.1
- networkx=2.6.3
- pandas=1.3.3
- anndata=0.7.6
- timm=0.4.12
- leidenalg=0.8.7


## Usage
Run `co