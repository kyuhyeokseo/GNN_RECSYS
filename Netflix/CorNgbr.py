# correlation based neighborhood model
import random

import numpy as np
from scipy import io, sparse
from math import sqrt
from time import time

from utils import path, compute_sparse_correlation_matrix


def predict_r_ui(mat, u, i, mu, S, Sk_iu, baseline_bu, baseline_bi):
    bui = mu + baseline_bu[u] + baseline_bi[0, i]
    sum = 0
    for j in Sk_iu:
        buj = mu + baseline_bu[u] + baseline_bi[0, j]
        sum += (S[1, j] * (mat[u, j] - buj))
    return bui + sum / S[i, Sk_iu].sum()


def correlation_based_neighbourhood_model(mat, mat_file, itr, bu, bi, l_reg2=100.0, k=10):


    no_users = mat.shape[0]
    no_movies = mat.shape[1]

    baseline_bu, baseline_bi = bu, bi
    mu = mat.data[:].mean()

    # Compute similarity matrix
    N = sparse.csr_matrix(mat).copy()

    N.data[:] = 1
    S = sparse.csr_matrix.dot(N.T, N) # Count common users for item i,j
    S.data[:] = S.data[:] / (S.data[:] + l_reg2)
    S = S * compute_sparse_correlation_matrix(mat)
    S = S.toarray()

    # Computation
    print("---Computation ---")
    n_iter = itr
    cx = mat.tocoo()
    r_ui_mat = []
    for u,i,v in zip(cx.row, cx.col, cx.data):
        Sk_iu = np.flip(np.argsort(S[i,])).ravel()[:k]
        r_ui = predict_r_ui(mat, u, i, mu, S, Sk_iu, baseline_bu, baseline_bi)
        r_ui_mat.append((u, i, r_ui[0]))


    data = list(map(lambda x: x[2], r_ui_mat))
    col = list(map(lambda x: x[1], r_ui_mat))
    row = list(map(lambda x: x[0], r_ui_mat))
    cnt = len(data)

    r_ui_pred = sparse.csr_matrix((data, (row, col)), shape=mat.shape)

    return mat, r_ui_pred, cnt



def rmse_2(mat,mat_file, itr, bu, bi):
    baseline_bu, baseline_bi = bu, bi
    mat, r_ui_pred, cnt = correlation_based_neighbourhood_model(mat, mat_file, itr, baseline_bu, baseline_bi)
    rmse = 0
    no_users = mat.shape[0]
    no_movies = mat.shape[1]
    for i in range(no_users):
        for j in range(no_movies):
            if (mat[i,j]!=0):
                rmse += ((mat[i,j]-r_ui_pred[i,j])**2)
    return (sqrt(rmse/cnt))


def CorNgbr():
    mat_file = path + "/T.mat"
    mat = io.loadmat(mat_file)['X']
    mat = mat[0:mat.shape[0] // 128, 0:mat.shape[1] // 128]
    mat = mat[mat.getnnz(1) > 0][:, mat.getnnz(0) > 0]

    rmse_list = []
    no_users = mat.shape[0]
    no_movies = mat.shape[1]
    baseline_bu, baseline_bi = np.random.rand(no_users, 1) * 2 - 1, np.random.rand(1, no_movies) * 2 - 1

    for i in [100, 200, 400, 800, 1500]:
        rmse_list.append(rmse_2(mat, mat_file, i, baseline_bu, baseline_bi))

    print(rmse_list)


def CorNgbr_top_k():
    mat_file = path + "/T.mat"
    mat = io.loadmat(mat_file)['X']
    mat = mat[0:mat.shape[0] // 128, 0:mat.shape[1] // 128]
    mat = mat[mat.getnnz(1) > 0][:, mat.getnnz(0) > 0]

    no_users = mat.shape[0]
    no_movies = mat.shape[1]
    baseline_bu, baseline_bi = np.random.rand(no_users, 1) * 2 - 1, np.random.rand(1, no_movies) * 2 - 1
    mat, r_pred, cnt = correlation_based_neighbourhood_model(mat, mat_file, 400, baseline_bu, baseline_bi)

    # Top K Recommendation
    rank_list = []
    cx = mat.tocoo()
    no_users = mat.shape[0]
    no_movies = mat.shape[1]

    dense_mat = r_pred.toarray()

    for u, i, v in zip(cx.row, cx.col, cx.data):
        if v == 5:
            nums = set()  # 중복을 제거하기 위해 집합사용
            rank_cnt = 0
            while len(nums) != 20:  # M개가 될때까지 순회하면서
                m_id = random.randint(0, no_movies - 1)
                if i != m_id:
                    nums.add(m_id)

            for x in nums:

                if dense_mat[u][x] > 5:
                    rank_cnt += 1

            rank_list.append(rank_cnt)

#    print(rank_list)


if __name__ == "__main__":
    CorNgbr()
