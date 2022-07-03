# Integrated Model
import random
import sys
import time


sys.setrecursionlimit(10**9)
from utils import pre_processing, compute_sparse_correlation_matrix, path

import numpy as np
from scipy import io, sparse
from math import sqrt


def predict_r_ui(mat, u, i, mu, bu, bi, Rk_iu, w_ij, Nk_iu, c_ij, baseline_bu, baseline_bi, qi, pu, N_u, yj):

    Rk_iu_sum = 0
    for j in Rk_iu:
        buj = mu + bu[u] + bi[0,j]
        Rk_iu_sum += (mat[u,j]-buj) * w_ij[i][j]

    Nk_iu_sum = c_ij[i][Rk_iu].sum()
    N_u_sum = yj[N_u].sum(0)

    return mu + bu[u] + bi[0, i] \
           + np.dot(qi[i], (pu[u] + N_u_sum / sqrt(len(N_u)))) + Rk_iu_sum / sqrt(len(Rk_iu)) + Nk_iu_sum / sqrt(len(Nk_iu))


def compute_e_ui(mat, u, i, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi, qi, pu, N_u, yj):
    return mat[u, i] - predict_r_ui(mat, u, i, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi, qi, pu, N_u, yj)


def integrated_model(mat, mat_file, f, gamma1=0.007, gamma2=0.007, gamma3=0.001, l_reg2=100, l_reg6=0.005, l_reg7=0.015,
                     l_reg8=0.015, k=300):

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
    bu = np.random.rand(no_users, 1) * 2 - 1
    bi = np.random.rand(1, no_movies) * 2 - 1
    wij = np.random.rand(no_movies, no_movies) * 2 - 1
    cij = np.random.rand(no_movies, no_movies) * 2 - 1
    qi = np.random.rand(no_movies, f) * 2 - 1
    pu = np.random.rand(no_users, f) * 2 - 1
    yj = np.random.rand(no_movies, f) * 2 - 1


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

    for it in range(n_iter):
        start = time.time()  # 시간측정 시작

        for u, i, v in zip(cx.row, cx.col, cx.data):

            N_u = bi_index[u]
            Nk_iu = np.flip(np.argsort(S[i,].toarray()))[:k].ravel()
            Rk_iu = Nk_iu

            e_ui = compute_e_ui(mat, u, i, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi, qi, pu, N_u, yj)
            bu[u] += gamma1 * (e_ui - l_reg6 * bu[u])
            bi[0,i] += gamma1 * (e_ui - l_reg6 * bi[0, i])

            qi[i] += gamma2 * (e_ui * (pu[u] + 1 / sqrt(len(N_u)) * yj[N_u].sum(0)) - l_reg7 * qi[i])
            pu[u] += gamma2 * (e_ui * qi[i] - l_reg7 * pu[u])
            yj[N_u] += gamma2 * (e_ui * 1 / sqrt(len(N_u)) * qi[i] - l_reg7 * yj[N_u])

            buj = []
            for j in Rk_iu :
                buj.append((mu+bu[u]+bi[0,j])[0])

            wij[i][Rk_iu] += gamma3 * ( 1/sqrt(len(Rk_iu)) * e_ui * (mat[u, Rk_iu].toarray().ravel() - buj) - l_reg8 * wij[i][Rk_iu])
            cij[i][Nk_iu] += gamma3 * ( 1/sqrt(len(Nk_iu)) * e_ui - l_reg8 * cij[i][Nk_iu])


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
                    rmse += ((compute_e_ui(mat, i, j, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi, qi, pu, N_u, yj)) ** 2)
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
                print(predict_r_ui(mat, u, x, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi, qi, pu, N_u, yj))
                if predict_r_ui(mat, u, x, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi, qi, pu, N_u, yj) > 5:
                    rank_cnt += 1
            print("------")

            rank_list.append(rank_cnt)


    return [(sqrt(rmse / cnt)), rank_list]



#################################################

def int_model():
    mat_file = path + "/T.mat"
    mat = io.loadmat(mat_file)['X']
    rmse_list = []
    for i in [25,50,100]:
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
