# SVD++
from scipy import io
from math import sqrt

import numpy as np

from utils import path, pre_processing


def predict_r_ui(mat, u, i, mu, bu, bi, qi, pu, N_u, yj):
    N_u_sum = yj[N_u].sum(0)
    return mu + bu[u] + bi[0, i] + np.dot(qi[i], (pu[u] + N_u_sum / sqrt(len(N_u))))


def compute_error_ui(mat, u, i, mu, bu, bi, qi, pu, N_u, yj):
    return mat[u, i] - predict_r_ui(mat, u, i, mu, bu, bi, qi, pu, N_u, yj)


def svd_more_more(mat, mat_file, f, gamma1=0.007, gamma2=0.007, l_reg6=0.005, l_reg7=0.015):

    # subsample the matrix to make computation faster
    mat = mat[0:mat.shape[0] // 128, 0:mat.shape[1] // 128]
    mat = mat[mat.getnnz(1) > 0][:, mat.getnnz(0) > 0]

    no_users = mat.shape[0]
    no_movies = mat.shape[1]

    bu_index, bi_index = pre_processing(mat, mat_file)

    # Init parameters
    bu = np.random.rand(no_users, 1) * 2 - 1
    bi = np.random.rand(1, no_movies) * 2 - 1
    qi = np.random.rand(no_movies, f) * 2 - 1
    pu = np.random.rand(no_users, f) * 2 - 1
    yj = np.random.rand(no_movies, f) * 2 - 1

    mu = mat.data[:].mean()

    print("--- Optimizing ---")
    n_iter = 100

    cx = mat.tocoo()

    for it in range(n_iter):
        for ind in range(len(cx.row)):
            u, i = cx.row[ind], cx.col[ind]
            N_u = bi_index[u]
            e_ui = compute_error_ui(mat, u, i, mu, bu, bi, qi, pu, N_u, yj)

            bu[u] += gamma1 * (e_ui - l_reg6 * bu[u])
            bi[0, i] += gamma1 * (e_ui - l_reg6 * bi[0, i])
            qi[i] += gamma2 * (e_ui * (pu[u] + 1 / sqrt(len(N_u)) * yj[N_u].sum(0)) - l_reg7 * qi[i])
            pu[u] += gamma2 * (e_ui * qi[i] - l_reg7 * pu[u])
            yj[N_u] += gamma2 * (e_ui * 1 / sqrt(len(N_u)) * qi[i] - l_reg7 * yj[N_u])

        gamma1 *= 0.9
        gamma2 *= 0.9


    rmse = 0
    cnt = 0


    no_users = mat.shape[0]
    no_movies = mat.shape[1]
    for i in range(no_users):
        for j in range(no_movies):
            if (mat[i, j] != 0):
                cnt += 1
                N_u = bi_index[i]
                rmse += ((mat[i, j] - predict_r_ui(mat, i, j, mu, bu, bi, qi, pu, N_u, yj)) ** 2)
#                if (i+j)%10 == 0:
#                    print(mat[i,j], predict_r_ui(mat, i, j, mu, bu, bi, qi, pu, N_u, yj), ((mat[i, j] - predict_r_ui(mat, i, j, mu, bu, bi, qi, pu,N_u, yj)) ** 2))

    return (sqrt(rmse / cnt))



def svd_mm():
    mat_file = path + "/T.mat"
    mat = io.loadmat(mat_file)['X']
    rmse_list = []
    for i in [25,50,100] :
        rmse_list.append(svd_more_more(mat, mat_file, i))

    print(rmse_list)


if __name__ == "__main__":
    svd_mm()