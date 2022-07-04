# Integrated Model
import random
import sys
import time


sys.setrecursionlimit(10**9)
from utils import pre_processing, compute_sparse_correlation_matrix, path

import numpy as np
from scipy import io, sparse
from math import sqrt


def predict_r_ui(mat, u, i, mu, Bu, Bi, Rk_iu, W, Nk_iu, C, baseline_bu, baseline_bi, Q, P, N_u, Y):

    p = P[u] + Y[N_u].sum(0)/np.sqrt(len(N_u))
    Factor_part = np.dot(p,Q[i].T)

    if len(Rk_iu)>0 :
        bias_uj = mat[u,Rk_iu] - (mu + Bu[u] + Bi[Rk_iu])
        Neigh_part = np.dot(bias_uj, W[i,Rk_iu]) / np.sqrt(len(Rk_iu))
    else :
        bias_uj = 0
        Neigh_part = 0

    if len(Nk_iu)>0 :
        Neigh_part = Neigh_part + np.sum(C[i,Nk_iu]) / np.sqrt(len(Nk_iu))

    return mu + Bu[u] + Bi[i] + Factor_part + Neigh_part, bias_uj



def compute_e_ui(mat, u, i, mu, Bu, Bi, Rk_iu, W, Nk_iu, C, baseline_bu, baseline_bi, Q, P, N_u, Y):
    predict, bias = predict_r_ui(mat, u, i, mu, Bu, Bi, Rk_iu, W, Nk_iu, C, baseline_bu, baseline_bi, Q, P, N_u, Y)

    return mat[u, i] - predict, bias



def integrated_model(mat, mat_file, f, gamma1=0.007, gamma2=0.007, gamma3=0.001, l_reg2=100, l_reg6=0.005, l_reg7=0.015,
                     l_reg8=0.015, k=30):

    # subsample
    mat = mat[0:mat.shape[0] // 128, 0:mat.shape[1] // 128]
    mat = mat[mat.getnnz(1) > 0][:, mat.getnnz(0) > 0]

    no_users = mat.shape[0]
    no_movies = mat.shape[1]


    baseline_bu, baseline_bi = np.random.rand(no_users, 1) * 2 - 1, np.random.rand(1, no_movies) * 2 - 1

    bu_index, bi_index = pre_processing(mat, mat_file)


    # Initializing
    Bu = np.random.standard_normal(no_users)
    Bi = np.random.standard_normal(no_movies)
    W = np.random.standard_normal((no_movies, no_movies))
    C = np.random.standard_normal((no_movies, no_movies))
    Q = np.random.standard_normal((no_movies, f))
    P = np.random.standard_normal((no_users, f))
    Y = np.random.standard_normal((no_movies, f))

    mu = mat.data[:].mean()

    N = sparse.csr_matrix(mat).copy()
    N.data[:] = 1
    S = sparse.csr_matrix.dot(N.T, N)
    S.data[:] = S.data[:] / (S.data[:] + l_reg2)
    S = S * compute_sparse_correlation_matrix(mat)


    # Train
    print("--- Optimizing ---")
    n_iter = 20
    # to penalize overflow
    cx = mat.tocoo()
    time_sum = 0
    rmse = 0
    cnt =0

    for it in range(n_iter):
        start = time.time()  # 시간측정 시작
        for u, i, v in zip(cx.row, cx.col, cx.data):
            N_u = bi_index[u]
            Nk_iu = np.flip(np.argsort(S[i,].toarray()))[:k].ravel()
            Rk_iu = Nk_iu

            e_ui, bias = compute_e_ui(mat, u, i, mu, Bu, Bi, Rk_iu, W, Nk_iu, C, baseline_bu, baseline_bi, Q, P, N_u, Y)

            Bu[u] = Bu[u] + gamma1 * (e_ui - l_reg6 * Bu[u])
            Bi[i] = Bi[i] + gamma1 * (e_ui - l_reg6 * Bi[i])

            Q[i] = Q[i] + gamma2 * (e_ui * ( P[u] + ( np.sum(Y[N_u], axis=0 ) )/np.sqrt(len(N_u)) ) - l_reg7 * Q[i])
            P[u] = P[i] + gamma2 * (e_ui * Q[i] - l_reg7 * P[u])

            Y[N_u] = Y[N_u] + gamma2 * ( ((e_ui * Q[i])/np.sqrt(len(N_u))).reshape(1,-1) - l_reg7 * Y[N_u])

            if len(Rk_iu)>0 :
                W[i][Rk_iu] = W[i][Rk_iu] + gamma3 * ( e_ui * bias / np.sqrt(len(Rk_iu)) - l_reg8 * W[i][Rk_iu] )
            if len(Nk_iu)>0 :
                C[i][Nk_iu] = C[i][Nk_iu] + gamma3 * ( e_ui / np.sqrt(len(Nk_iu)) - l_reg8 * C[i][Nk_iu] )
        gamma1 *= 0.9
        gamma2 *= 0.9
        gamma3 *= 0.9

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
                    N_u = bi_index[i]
                    Nk_iu = np.flip(np.argsort(S[j,].toarray()))[:k].ravel()
                    Rk_iu = Nk_iu

                    e_ui, bias = compute_e_ui(mat, i, j, mu, Bu, Bi, Rk_iu, W, Nk_iu, C, baseline_bu, baseline_bi, Q, P,N_u, Y)

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
                N_u = bi_index[u]
                Nk_iu = np.flip(np.argsort(S[x,].toarray()))[:k].ravel()
                Rk_iu = Nk_iu
                predict, bias = predict_r_ui(mat, u, x, mu, Bu, Bi, Rk_iu, W, Nk_iu, C, baseline_bu, baseline_bi, Q, P, N_u, Y)

                if predict > 5:
                    rank_cnt += 1

            rank_list.append(rank_cnt)


    return [(sqrt(rmse / cnt)), rank_list]



#################################################

def int_model():
    mat_file = path + "/T.mat"
    mat = io.loadmat(mat_file)['X']
    rmse_list = []
    for i in [10,20,40,80]:
        rmse_list.append(integrated_model(mat, mat_file, i)[0])

    print(rmse_list)


def Inte_top_k():
    mat_file = path + "/T.mat"
    mat = io.loadmat(mat_file)['X']
    rank_list = integrated_model(mat, mat_file, 100)[1]

    print(rank_list)

    sum = 0
    cum_list = []
    for i in range(41):
        sum = sum + rank_list.count(i)
        cum_list.append(sum)

    print(cum_list)



if __name__ == "__main__":
    int_model()
    #Inte_top_k()
