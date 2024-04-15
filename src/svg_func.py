
# code from https://github.com/jianhuupenn/SpaGCN
import scanpy as sc
import pandas as pd
import numpy as np
import scipy
from scipy.sparse import issparse
import numba


@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1,t2):
    sum=0
    for i in range(t1.shape[0]):
        sum+=(t1[i]-t2[i])**2
    return np.sqrt(sum)


@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
    n=X.shape[0]
    adj=np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j]=euclid_dist(X[i], X[j])
    return adj


def calculate_adj_matrix(x, y):

    X=np.array([x, y]).T.astype(np.float32)
    return pairwise_distance(X)


def count_nbr(target_cluster,cell_id, x, y, pred, adj_2d, radius):
    # adj_2d=calculate_adj_matrix(x=x,y=y, histology=False)
    cluster_num = dict()
    df = {'cell_id': cell_id, 'x': x, "y":y, "pred":pred}
    df = pd.DataFrame(data=df)
    df.index=df['cell_id']
    target_df=df[df["pred"]==target_cluster]
    row_index=0
    num_nbr=[]
    for index, row in target_df.iterrows():
        x=row["x"]
        y=row["y"]
        tmp_nbr=df[((df["x"]-x)**2+(df["y"]-y)**2)<=(radius**2)]
        num_nbr.append(tmp_nbr.shape[0])
    return np.mean(num_nbr)


def search_radius(target_cluster,cell_id, x, y, pred, adj_2d, start, end, num_min=8, num_max=15,  max_run=100):
    run=0
    num_low=count_nbr(target_cluster,cell_id, x, y, pred, adj_2d, start)
    num_high=count_nbr(target_cluster,cell_id, x, y, pred, adj_2d, end)
    if num_min<=num_low<=num_max:
        print("recommended radius = ", str(start))
        return start
    elif num_min<=num_high<=num_max:
        print("recommended radius = ", str(end))
        return end
    elif num_low>num_max:
        print("Try smaller start.")
        return None
    elif num_high<num_min:
        print("Try bigger end.")
        return None
    while (num_low<num_min) and (num_high>num_min):
        run+=1
        print("Run "+str(run)+": radius ["+str(start)+", "+str(end)+"], num_nbr ["+str(num_low)+", "+str(num_high)+"]")
        if run >max_run:
            print("Exact radius not found, closest values are:\n"+"radius="+str(start)+": "+"num_nbr="+str(num_low)+"\nradius="+str(end)+": "+"num_nbr="+str(num_high))
            return None
        mid=(start+end)/2
        num_mid=count_nbr(target_cluster,cell_id, x, y, pred, adj_2d, mid)
        if num_min<=num_mid<=num_max:
            print("recommended radius = ", str(mid), "num_nbr="+str(num_mid))
            return mid
        if num_mid<num_min:
            start=mid
            num_low=num_mid
        elif num_mid>num_max:
            end=mid
            num_high=num_mid


def rank_genes_groups(input_adata, target_cluster,nbr_list, label_col, adj_nbr=True, log=False):