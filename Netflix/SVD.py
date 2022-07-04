# SVD
import random

from scipy import io
from math import sqrt

import numpy as np

from utils import path, pre_processing


def predict_r_ui(mat, u, i, mu, Bu, Bi, Q, P):

    return mu + Bu[u] + Bi[i] + np.dot(P[u], Q[i].T)

def compute_e_ui(mat, u, i, mu, Bu, Bi, Q, P):

    predict = predict_r_ui(mat, u, i, mu, Bu, Bi, Q, P)

    return mat[u, i] - predict


def _svd(mat, mat_file, f, gamma, l_reg6 = 0.005, l_reg7 = 0.015):

    rmse = 0
    cnt = 0

    # subsample
    mat = mat[0:mat.shape[0] // 128, 0:mat.shape[1] // 128]
    mat = mat[mat.getnnz(1) > 0][:, mat.getnnz(0) > 0]

    no_users = mat.shape[0]
    no_movies = mat.shape[1]

    bu_index, bi_index = pre_processing(mat, mat_file)

    # Init parameters
    Bu = np.random.standard_normal(no_users)
    Bi = np.random.standard_normal(no_movies)
    Q = np.random.standard_normal((no_movies, f))
    P = np.random.standard_normal((no_users, f))

    mu = mat.data[:].mean()

    n_iter = 30
    cx = mat.tocoo()


    print("--- Optimizing ---")
    for it in range(n_iter):
        for u, i, v in zip(cx.row, cx.col, cx.data):

            e_ui = compute_e_ui(mat, u, i, mu, Bu, Bi, Q, P)

            Bu[u] = Bu[u] + gamma * (e_ui - l_reg6 * Bu[u])
            Bi[i] = Bi[i] + gamma * (e_ui - l_reg6 * Bi[i])
            Q[i] = Q[i] + gamma * (e_ui * P[u] - l_reg7 * Q[i])
            P[u] = P[i] + gamma * (e_ui * Q[i] - l_reg7 * P[u])
        gamma *= 0.9
        rmse = 0
        cnt = 0
        for u, i, v in zip(cx.row, cx.col, cx.data):
            e_ui = compute_e_ui(mat, u, i, mu, Bu, Bi, Q, P)
            rmse = rmse + (e_ui*e_ui)
            cnt = cnt + 1
        print("iter : ", it, " / RMSE : ", sqrt(rmse / cnt))

    return (sqrt(rmse / cnt))






def svd_based():

    mat_file = path + "/T.mat"
    mat = io.loadmat(mat_file)['X']
    print(_svd(mat, mat_file, 40, 0.007))



if __name__ == "__main__":
    svd_based()
