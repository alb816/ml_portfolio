import numpy as np
from vector_spaces.rref_matrix import rref

def get_colA_basis(A):
    """Находит базис пространства столбцов матрицы А."""

	# column space of A = span A = image A = all possible linear combinations of the columns in A
    R = rref(A)
    
    # Находим позиции первых ненулевых элементов в каждой строке
    pivot_indices = []
    for r in range(R.shape[0]):
        # Находим все ненулевые элементы в строке
        non_zero = np.where(np.abs(R[r, :]) > 0)[0]
        if len(non_zero) > 0:
            pivot_indices.append(non_zero[0])  # Берем первый
    
    return A[:, sorted(set(pivot_indices))] if pivot_indices else np.zeros((A.shape[0], 0))


if __name__ == "__main__":
    A = np.array(
        [
            [1, 2, 3],
            [2, 4, 6],
            [1, 1, 1]
        ]    
    )  

    print(get_colA_basis(A))
