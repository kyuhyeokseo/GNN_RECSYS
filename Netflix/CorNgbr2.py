# correlation based neighborhood model > Calculate rank
import random

import numpy as np
from scipy import io, sparse


from utils import path, compute_sparse_correlation_matrix


def predict_r_ui(mat, u, i, mu, S, Sk_iu, baseline_bu, baseline_bi, k):
    bui = mu + baseline_bu[u] + baseline_bi[0, i]
    sum = 0
    cnt = 0
    divide = 0
    for j in Sk_iu:
        if mat[u,j]!=0:
            buj = mu + baseline_bu[u] + baseline_bi[0, j]
            sum += (S[i,j]*(mat[u,j]-buj[0]))
            divide += S[i,j]
            cnt +=1

        if cnt == k:
            break

    if divide==0 :
        divide =1

    return bui + sum / divide



def correlation_based_neighbourhood_model(mat, mat_file, l_reg2=100.0, k=20):
    # subsample
    mat = mat[0:mat.shape[0]//128, 0:mat.shape[1]//128]
    mat = mat[mat.getnnz(1)>0][:, mat.getnnz(0)>0]


    no_users = mat.shape[0]
    no_movies = mat.shape[1]

    baseline_bu, baseline_bi = np.random.rand(no_users, 1)  * 2 - 1, np.random.rand(1, no_movies) * 2 - 1


    mu = mat.data[:].mean()

    # Compute similarity matrix
    N = sparse.csr_matrix(mat).copy()
    N.data[:] = 1
    S = sparse.csr_matrix.dot(N.T, N)

    S.data[:] = S.data[:] / (S.data[:] + l_reg2)


    S = S * compute_sparse_correlation_matrix(mat)
    S = S.toarray()

    # Computation
    print("Computation...")
    n_iter = 200
    cx = mat.tocoo()
    r_ui_mat = []
    for u in range(no_users):
        for i in range(no_movies):
            Sk_iu = (np.flip(np.argsort(S[i,]))).ravel()[:]
            r_ui = predict_r_ui(mat, u, i, mu, S, Sk_iu, baseline_bu, baseline_bi, k)
            r_ui_mat.append((u, i, r_ui[0]))


    data = list(map(lambda x: x[2], r_ui_mat))
    col = list(map(lambda x: x[1], r_ui_mat))
    row = list(map(lambda x: x[0], r_ui_mat))
    r_ui_pred = sparse.csr_matrix((data, (row, col)), shape=mat.shape)

    return [mat, r_ui_pred.toarray()]








if __name__ == "__main__":
    mat_file = path+"/T.mat"
    mat = io.loadmat(mat_file)['X']
    RESULT = correlation_based_neighbourhood_model(mat, mat_file)
    mat = RESULT[0]
    pred = RESULT[1]

    # Top K Recommendation
    rank_list = []
    no_movies = mat.shape[1]
    cx = mat.tocoo()




    for u, i, v in zip(cx.row, cx.col, cx.data):
        if v == 5:
            nums = set()  # 중복을 제거하기 위해 집합사용
            rank_cnt = 0
            while len(nums) != 20:  # M개가 될때까지 순회하면서
                m_id = random.randint(0, no_movies - 1)
                if i != m_id:
                    nums.add(m_id)

            for x in nums:

                if pred[u,x] >= 5:
                    rank_cnt += 1

            rank_list.append(rank_cnt)

    sum = 0
    cum_list = []
    for i in range(21):
        sum = sum + rank_list.count(i)
        cum_list.append(sum)

    print(cum_list)