import numpy as np



def get_ONB(vectors: list[list[float]]) -> list[np.ndarray]:
    """
    Конвертирует исходный базис в ортонормированный методом Грама-Шмидта.
    """
    onb = []
    for v in vectors:
        v = np.asarray(v, dtype=float).copy()
        for u in onb:
            v -= np.dot(v, u) * u 
            
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            onb.append(v / norm)

    return onb


if __name__ == "__main__":
    b = [[1.0,0.0], [1.8, 1.0]]
    print(get_ONB(b))


