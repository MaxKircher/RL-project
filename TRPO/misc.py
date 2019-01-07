import numpy as np
from conjugate_gradient import *

row1 = []
row2 = []

for i in range(15):
    if i == 14:
        row1 += [0]
        row2 += [1]
    else:
        row1 += [i + 1]
        row2 += [0]

J = np.matrix((row1, row2))
M = np.matrix([[2, 0], [0, 1]])

Matrix = J.T * M * J
g = np.ones((15, 1))
g_lstsq = np.ones(15)

#print(J.shape)
#print(Matrix)
#print(np.linalg.matrix_rank(Matrix))

# cg = ConjugateGradient(10)
# x0 = .5 * np.ones((15, 1))
#
# x_cg = cg.cg(g, J, M, x0)
# x_lstsq = np.linalg.lstsq(Matrix, g_lstsq, rcond=None)[0]

#print("x_cg: ", x_cg)
#print("x_lstsq: ", np.matrix(x_lstsq).T)

# print(np.abs(x_cg - np.matrix(x_lstsq).T))

'''
    Abgewandelte CG implementierung:
    Einfach Matrix A statt, etwas der Form J.T @ M @ J
    Solve Ax = g

    x = x0
'''
def conjugate_gradient(g, A, x):
    Ax = A @ x

    r = g - Ax
    d = r

    r_norm_2 = r.T @ r

    for i in range(10):
        z = A @ d
        dz = d.T @ z
        assert dz != 0
        alpha = r_norm_2 / dz

        x += alpha[0,0] * d
        r = r - alpha[0,0] * z

        r_kplus1_norm_2 = r.T @ r
        beta = r_kplus1_norm_2 / r_norm_2

        r_norm_2 = r_kplus1_norm_2

        d = r + beta[0,0] * d

        if r_norm_2[0,0] < 1e-10:
            break

    return x

x0 = 2 * np.ones((15,1))
print(conjugate_gradient(g, np.eye(15,15), x0))
