import numpy as np

def calculate_eigenvalues_2x2(matrix):
    A = np.array(matrix)
    tr_A = np.sum(np.diag(A)) # np.trace(A)
    det_A = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    sqrt_disc = np.sqrt(tr_A**2 - 4 * det_A)
    l1 = (tr_A + sqrt_disc) / 2
    l2 = (tr_A - sqrt_disc) / 2
    return sorted([float(l1), float(l2)], reverse=True)