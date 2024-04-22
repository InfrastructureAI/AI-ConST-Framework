# some of the code is from https://github.com/JinmiaoChenLab/SEDR
import os
import scanpy as sc
import pandas as pd
from pathlib import Path
from scanpy.readwrite import read_visium
from scanpy._utils import check_presence_download
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_clustering(adata, colors, savepath = None):
    adata.obs['x_pixel'] = adata.obsm['spatial'][:, 0]
    adata.obs['y_pixel'] = adata.obsm['spatial'][:, 1]

    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(111)
    sc.pl.scatter(adata, alpha=1, x="x_pixel", y="y_pixel", color=colors, title='Clustering of 151673 slice',
                  palette=sns.color_palette('plasma', 7), show=False, ax=ax1)

    ax1.set_aspect('equal', 'box')
    ax1.axis('off')
    ax1.ax