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
            resolution[int]
    '''
    for res in sorted(list(np.arange(0.01, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res


def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path


def adata_preprocess(i_adata, min_cells=3, pca_n_comps=300):
    print('===== Preprocessing Data ')
    sc.pp.filter_genes(i_adata, min_cells=min_cells)
    adata_X = sc.pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = sc.pp.scale(adata_X)
    adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
    return adata_X


def load_ST_file(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True