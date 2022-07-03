# THANKS TO MATPONT (Github)

from itertools import groupby
from operator import itemgetter
import pickle
import os
from sklearn.preprocessing import StandardScaler
import numpy as np

path = "dataset"

def compute_sparse_correlation_matrix(A: object) -> object:
    scaler = StandardScaler(with_mean=False, with_std=False)
    scaled_A = scaler.fit_transform(A)  # Assuming A is a CSR or CSC matrix
    corr_matrix = (1/scaled_A.shape[0]) * (scaled_A.T @ scaled_A)
    return corr_matrix

def pre_processing(mat, mat_file):

    shape = str(mat.shape[0])+"_"+str(mat.shape[1])
    bu_index_file = mat_file+"_bu_index_"+shape+".data"
    bi_index_file = mat_file+"_bi_index_"+shape+".data"

    if not (os.path.isfile(bu_index_file) and os.path.isfile(bi_index_file)):

        print("---- Pre-processing ----")
        mat_nonzero = mat.nonzero()

        print("   making bi indexes   ")
        bi_index = []
        for k, g in groupby(zip(mat_nonzero[0], mat_nonzero[1]), itemgetter(0)):
          to_add = list(map(lambda x:int(x[1]), list(g)))
          bi_index.append(to_add)

        print("   making bu indexes   ")
        bu_index = []
        indexes = np.argsort(mat_nonzero[1])
        for k, g in groupby(zip(mat_nonzero[1][indexes], mat_nonzero[0][indexes]), itemgetter(0)):
          to_add = list(map(lambda x:int(x[1]), list(g)))
          bu_index.append(to_add)

        with open(bi_index_file, "wb") as fp:
            pickle.dump(bi_index, fp)
        with open(bu_index_file, "wb") as fp:
            pickle.dump(bu_index, fp)
    else:
        with open(bi_index_file, "rb") as fp:
            bi_index = pickle.load(fp)
        with open(bu_index_file, "rb") as fp:
            bu_index = pickle.load(fp)

    print("Pre-processing done.")
    return bu_index, bi_index