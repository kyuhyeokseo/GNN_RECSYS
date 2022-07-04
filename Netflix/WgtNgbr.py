# Integrated Model
import random
import sys
import time


sys.setrecursionlimit(10**9)
from utils import pre_processing, compute_sparse_correlation_matrix, path

import numpy as np
from scipy import io, sparse
from math import sqrt


def predict_r_ui(mat, u, i, mu, Bu, Bi, W, R_u):

    if len(R_u)>0 :
        bias_uj = mat[u,R_u] - (mu + Bu[u] + Bi[R_u])
        Neigh_part = np.dot(bias_uj, W[i,R_u]) / np.sqrt(len(R_u))
    else :
        bias_uj = 0
        Neigh_part = 0

    return mu + Bu[u] + Bi[i] + Neigh_part, bias_uj



def compute_e_ui(mat, u, i, mu, Bu, Bi, W, R_u):

    predict, bias = predict_r_ui(mat, u, i, mu, Bu, Bi, W, R_u)

    return mat[u, i] - predict, bias



def WgtNgbr_model(mat, mat_file, gamma1=0.005, l_reg2=100, l_reg6=0.002):

    # subsample
    mat = mat[0:mat.shape[0] // 128, 0:mat.shape[1] // 128]
    mat = mat[mat.getnnz(1) > 0][:, mat.getnnz(0) > 0]

    no_users = mat.shape[0]
    no_movies = mat.shape[1]

    # baseline_bu, baseline_bi = baseline_estimator(mat)
    # We should call baseline_estimator but we can init at random for test
    baseline_bu, baseline_bi = np.random.rand(no_users, 1) * 2 - 1, np.random.rand(1, no_movies) * 2 - 1

    bu_index, bi_index = pre_processing(mat, mat_file)


    # Initializing
    Bu = np.random.standard_normal(no_users)
    Bi = np.random.standard_normal(no_movies)
    W = np.random.standard_normal((no_movies, no_movies))


    mu = mat.data[:].mean()

    N = sparse.csr_matrix(mat).copy()
    N.data[:] = 1
    S = sparse.csr_matrix.dot(N.T, N)
    S.data[:] = S.data[:] / (S.data[:] + l_reg2)
    S = S * compute_sparse_correlation_matrix(mat)


    # Train
    print("--- Optimizing ---")
    n_iter = 100
    # to penalize overflow
    cx = mat.tocoo()
    time_sum = 0

    for it in range(n_iter):
        start = time.time()  # 시간측정 시작

        for u, i, v in zip(cx.row, cx.col, cx.data):

            R_u = bi_index[u]

            e_ui, bias = compute_e_ui(mat, u, i, mu, Bu, Bi, W, R_u)

            Bu[u] = Bu[u] + gamma1 * (e_ui - l_reg6 * Bu[u])
            Bi[i] = Bi[i] + gamma1 * (e_ui - l_reg6 * Bi[i])

            if len(R_u)>0 :
                W[i][R_u] = W[i][R_u] + gamma1 * ( e_ui * bias / np.sqrt(len(R_u)) - l_reg6 * W[i][R_u] )

        gamma1 *= 0.9

        end = time.time()  # 시간측정 종료
        time_sum += end-start

        rmse = 0
        cnt = 0

        no_users = mat.shape[0]
        no_movies = mat.shape[1]
        for i in range(no_users):
            for j in range(no_movies):
                if (mat[i, j] != 0):
                    cnt += 1
                    R_u = bi_index[i]

                    e_ui, bias = compute_e_ui(mat, i, j, mu, Bu, Bi, W, R_u)

                    rmse += ( e_ui ** 2)
        #                if (i+j)%10 == 0:
        #                    print(mat[i,j], predict_r_ui(mat, i, j, mu, bu, bi, qi, pu, N_u, yj), ((mat[i, j] - predict_r_ui(mat, i, j, mu, bu, bi, qi, pu,N_u, yj)) ** 2))

        print("iter : ",it, " / RMSE : ", sqrt(rmse/cnt))

    time_avg = time_sum/n_iter
    t = str(round(time_avg * (10 ** 3), 2)) + "ms"  # 걸린 시간 ms 단위로 변환
    print(t)



    # Top K Recommendation
    rank_list = []

    for u, i, v in zip(cx.row, cx.col, cx.data):
        if u==741 and v==5:
            nums = set()  # 중복을 제거하기 위해 집합사용
            rank_cnt = 0
            while len(nums) != 40:  # M개가 될때까지 순회하면서
                m_id = random.randint(0, no_movies-1)
                if i!=m_id :
                    nums.add(m_id)

            for x in nums:
                R_u = bi_index[u]

                predict, bias = predict_r_ui(mat, u, x, mu, Bu, Bi, W, R_u)

                if predict > 5:
                    rank_cnt += 1

            rank_list.append(rank_cnt)


    return [(sqrt(rmse / cnt)), rank_list]



#################################################

def Wgt_model():
    mat_file = path + "/T.mat"
    mat = io.loadmat(mat_file)['X']

    print(WgtNgbr_model(mat, mat_file)[0])




def Inte_top_k():
    mat_file = path + "/T.mat"
    mat = io.loadmat(mat_file)['X']
    rank_list = WgtNgbr_model(mat, mat_file)[1]

    print(rank_list)

    sum = 0
    cum_list = []
    for i in range(41):
        sum = sum + rank_list.count(i)
        cum_list.append(sum)

    print(cum_list)



if __name__ == "__main__":
    Wgt_model()

