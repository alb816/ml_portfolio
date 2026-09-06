import numpy as np 


def orthogonal_projection(v, L):
    """
    Вычисляет ортогональную проекцию вектора v на L.

	:param v: Проецируемый вектор
	:param L: Вектор задающий направление проекции
	:return: Вектор - проекция v на L
	"""
    v = np.array(v, dtype=float)
    L = np.array(L, dtype=float)

    # скалярное произведение v и L
    dot_vL = np.dot(v, L)

    # квадрат длины L
    sq_len_L = np.dot(L, L) # эквивалентно np.linalg.norm(L)**2

    if sq_len_L == 0:
        raise ValueError("Vector L is zero vector.")
    
    proj_v = dot_vL / sq_len_L * L # проекция вектора v на прямую вдоль вектора L

    return proj_v


if __name__ == "__main__":
    v = [3, 4]
    L = [1, 0]
    print(np.round(orthogonal_projection(v, L), 3))

