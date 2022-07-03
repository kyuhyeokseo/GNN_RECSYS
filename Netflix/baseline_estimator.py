# baseline estimator
from math import sqrt

from scipy import io, sparse
import numpy as np
import matplotlib.pyplot as plt

from utils import pre_processing, path


def compute_loss(mat, mu, bu, bi, l_reg=0.002):
    loss = 0

    no_users_entries = np.array((mat != 0).sum(1)).T.ravel()
    bu_rep = np.repeat(bu.ravel(), no_users_entries)

    no_movies_entries = np.array((mat != 0).sum(0)).ravel()
    bi_rep = np.repeat(bi.ravel(), no_movies_entries)

    temp_mat = sparse.csc_matrix(mat).copy()
    temp_mat.data[:] -= bi_rep
    temp_mat.data[:] -= mu
    temp_mat = sparse.coo_matrix(temp_mat)
    temp_mat = sparse.csr_matrix(temp_mat)
    temp_mat.data[:] -= bu_rep

    loss = (temp_mat.data[:] ** 2).sum()

    loss_reg = l_reg * ((bu ** 2).sum() + (bi ** 2).sum())
    # loss += loss_reg

    return loss + loss_reg


def baseline_estimator(mat, mat_file, itr, l_reg=0.002, learning_rate=0.00005):
    # subsample the matrix to make computation faster
    mat = mat[0:mat.shape[0] // 128, 0:mat.shape[1] // 128]
    mat = mat[mat.getnnz(1) > 0][:, mat.getnnz(0) > 0]

    no_users = mat.shape[0]
    no_movies = mat.shape[1]

    bu_index, bi_index = pre_processing(mat, mat_file)

    bu = np.random.rand(no_users, 1) * 2 - 1
    bi = np.random.rand(1, no_movies) * 2 - 1
    # bu = np.zeros((no_users,1))
    # bi = np.zeros((1,no_movies))

    mu = mat.data[:].mean()
    mat_sum1 = mat.sum(1)
    mat_sum0 = mat.sum(0)
    n = mat.data[:].shape[0]

    no_users_entries = np.array((mat != 0).sum(1))
    no_movies_entries = np.array((mat != 0).sum(0))

    # Train
    print("Train")
    n_iter = itr
    for it in range(n_iter):

        bi_sum = np.array(list(map(lambda x: bi.ravel()[x].sum(), bi_index))).reshape((no_users, 1))
        bu_sum = np.array(list(map(lambda x: bu.ravel()[x].sum(), bu_index))).reshape((1, no_movies))

        # Vectorized operations
        bu_gradient = - 1.0 * (mat_sum1 - no_users_entries * mu - no_users_entries * bu - bi_sum) + 1.0 * l_reg * bu
        bu -= learning_rate * bu_gradient

        bi_gradient = - 1.0 * (mat_sum0 - no_movies_entries * mu - no_movies_entries * bi - bu_sum) + 1.0 * l_reg * bi
        bi -= learning_rate * bi_gradient

    return bu, bi, mat, mu


def rmse_1(mat,mat_file, itr):
    bu, bi, mat, mu = baseline_estimator(mat, mat_file, itr)
    rmse = 0
    cnt = 0
    no_users = mat.shape[0]
    no_movies = mat.shape[1]
    for i in range(no_users):
        for j in range(no_movies):
            if (mat[i,j]!=0):
                cnt += 1
                rmse += int((mat[i,j]- mu - bu[i,0] - bi[0,j])**2)
    print(cnt)
    return (sqrt(rmse/cnt))

def be():
    mat_file = path + "/T.mat"
    mat = io.loadmat(mat_file)['X']
    rmse_list = []
    for i in [100, 200, 400, 800, 1500]:
        rmse_list.append(rmse_1(mat, mat_file, i))
    print(rmse_list)

if __name__ == "__main__":
    be()