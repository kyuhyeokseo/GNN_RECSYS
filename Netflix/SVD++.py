# SVD++
from scipy import io
from math import sqrt

import numpy as np

from utils import path, pre_processing


def predict_r_ui(mat, u, i, mu, Bu, Bi, Q, P, N_u, Y):

    p = P[u] + Y[N_u].sum(0) / np.sqrt(len(N_u))
    Factor_part = np.dot(p, Q[i].T)

    return mu + Bu[u] + Bi[i] + Factor_part


def compute_e_ui(mat, u, i, mu, Bu, Bi, Q, P, N_u, Y):

    predict = predict_r_ui(mat, u, i, mu, Bu, Bi, Q, P, N_u, Y)

    return mat[u, i] - predict


def svd_more_more(mat, mat_file, f, gamma1=0.007, gamma2=0.007, l_reg6=0.005, l_reg7=0.015):

    rmse = 0
    cnt = 0
    # subsample the matrix to make computation faster
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
    Y = np.random.standard_normal((no_movies, f))

    mu = mat.data[:].mean()

    print("--- Optimizing ---")
    n_iter = 30

    cx = mat.tocoo()

    for it in range(n_iter):
        for ind in range(len(cx.row)):
            u, i = cx.row[ind], cx.col[ind]

            N_u = bi_index[u]

            e_ui = compute_e_ui(mat, u, i, mu, Bu, Bi, Q, P, N_u, Y)

            Bu[u] = Bu[u] + gamma1 * (e_ui - l_reg6 * Bu[u])
            Bi[i] = Bi[i] + gamma1 * (e_ui - l_reg6 * Bi[i])
            Q[i] = Q[i] + gamma2 * ( e_ui * ( P[u] + (np.sum(Y[N_u], axis=0) )/np.sqrt(len(N_u)) ) - l_reg7 * Q[i] )
            P[u] = P[i] + gamma2 * (e_ui * Q[i] - l_reg7 * P[u])
            Y[N_u] = Y[N_u] + gamma2 * ( ((e_ui * Q[i]) / np.sqrt(len(N_u))).reshape(1, -1) - l_reg7 * Y[N_u] )

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
                    e_ui = compute_e_ui(mat, i, j, mu, Bu, Bi, Q, P, N_u, Y)

                    rmse = rmse + (e_ui ** 2)
    #                if (i+j)%10 == 0:
    #                    print(mat[i,j], predict_r_ui(mat, i, j, mu, bu, bi, qi, pu, N_u, yj), ((mat[i, j] - predict_r_ui(mat, i, j, mu, bu, bi, qi, pu,N_u, yj)) ** 2))
        print("iter : ", it, " / RMSE : ", sqrt(rmse / cnt))


    return (sqrt(rmse / cnt))



def svd_mm():

    rmse_list = []
    for i in (10,20,40,80):

        mat_file = path + "/T.mat"
        mat = io.loadmat(mat_file)['X']
        rmse_list.append(svd_more_more(mat, mat_file, i))

    print(rmse_list)


if __name__ == "__main__":
    svd_mm()