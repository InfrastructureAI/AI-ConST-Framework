import os,csv,re
import pandas as pd
import numpy as np


def Moran_I(genes_exp, XYdistances, XYindices):
    
    W = np.zeros((genes_exp.shape[0],genes_exp.shape[0]))
    for i in range(0,genes_exp.shape[0]):
        W[i,XYindices[i,:]]=1
    for i in range(0,genes_exp.shape[0]):
        W[i,i]=0
    
    I = pd.Series(index=genes_exp.columns, d