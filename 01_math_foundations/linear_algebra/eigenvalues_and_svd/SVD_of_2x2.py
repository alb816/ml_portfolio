import numpy as np



def EVD(A):
    eig_vals, eig_vecs = np.linalg.eigh(A)
    # D = np.diag(eig_vals)
    # evd_A = eig_vecs @ D @ eig_vecs.T
    return eig_vals, eig_vecs


def svd_2x2(A: np.ndarray) -> tuple:
    # 1. Сначала ищем V -- ортогональную матрицу правосингулярных векторов A
    S = A.T @ A

    # применяем собственное разложение PDP^T над S и находим её собств векторы и значения
    eig_vals, P = EVD(S)
    
    # 2. Найденные собственные значения S - это квадраты сингулярных значений в сингулярном разложении 
    sigma = np.sqrt(np.maximum(eig_vals, 0)) # maximum заменяет возможные отрицательные числа (ошибки округления) на ноль

    # сортируем сингулярные значения по убыванию
    idx = sigma.argsort()[::-1]
    sigma = sigma[idx]

    # правосингулярных векторы также сортируем по убыванию соответствующих им сингулярных значений
    V = P[:, idx]

    # 3. Поиск левосингулярной матрицы U
    sigma_inv = np.where(sigma > 1e-10, 1.0 / sigma, 0) 
    sigma_inv_mat = np.diag(sigma_inv)
    U = A @ V @ sigma_inv_mat

    # 4. Итоговое разложение
    # sigma_mat = np.diag(sigma)
    # svd_A = U @ sigma_mat @ V.T

    return U, sigma, V.T



if __name__ == "__main__":
    A = np.array([[2, 1], [1, 2]])
    U, sigma, Vt = svd_2x2(A)
    Sigma = np.diag(sigma)
    assert np.allclose(U @ Sigma @ Vt, A)
    print(svd_2x2(A))