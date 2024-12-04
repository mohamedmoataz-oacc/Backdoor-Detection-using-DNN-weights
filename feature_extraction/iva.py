# import numpy as np
# from independent_vector_analysis import iva_g, consistent_iva
# from independent_vector_analysis.data_generation import MGGD_generation


# N = 3
# K = 4
# T = 10000
# rho = 0.7
# S = np.zeros((N, T, K))
# for idx in range(N):
#     S[idx, :, :] = MGGD_generation(T, K, 'ar', rho, 1)[0].T
# A = np.random.randn(N,N,K)
# X = np.einsum('MNK, NTK -> MTK', A, S)
# W, cost, Sigma_n, isi = iva_g(X, A=A, jdiag_initW=False)