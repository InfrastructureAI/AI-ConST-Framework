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
    ax1.axes.invert_yaxis()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')


def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]
        
        return:
            resolution[i