# [conST: an interpretable multi-modal contrastive learning framework for spatial transcriptomics](https://www.biorxiv.org/content/10.1101/2022.01.14.476408v1)
We are actively updating this repository! More features/examples/experiments coming soon!

## Overview

![Framework overview](imgs/conST-workflow.png)
Framework of conST. conST models the ST data as a graph by treating gene expression and
morphology as node attributes and constructing edges by spatial coordinates. The training is divided into
two stages: pretraining and major training stage. Pretraining stage initializes the weights of the encoder E
by reconstruction loss. In major training stage, data augmentation is applied and then contrastive learning
in three levels, i.e., local-local, local-global, local-context, are used to learn a low-dimensional embedding by
minimize or maximize the mutual information (MI) between different embeddings. The le