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

Matrix = J.T @ M @ J
b = np.ones(15)

print(Matrix)
print(np.linalg.matrix_rank(Matrix))
