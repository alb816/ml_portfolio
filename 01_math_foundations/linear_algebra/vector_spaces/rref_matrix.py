import numpy as np

def rref(matrix:np.ndarray):
    """
    Функция находит RREF(Reduced Row Echelon Form) - приведенную ступенчатую форму исходной матрицы.
    Такая форма позволяет находить канонический базис.
    
    Аргументы:
        matrix (np.ndarray): Исходная матрица.
        
    Возвращает:
        np.ndarray: Матрица в RREF форме.
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Входной параметр должен быть массивом numpy.")
    if len(matrix.shape) != 2:
        raise ValueError("Матрица должна быть двумерной.")
        
    A = matrix.astype(float)
    rows, cols = A.shape
    pivot = 0 # индекс текущего исследуемого столбца для поиска опорного элемента
    for r in range(0, rows):
        if pivot >= cols:
            return A
        if np.all(A[r, :] == 0):
            continue
            
        i = r
        while np.abs(A[i, pivot]) <= 1e-8: # пока элемент на позиции ведущего равен нулю:
            i += 1 
            if i == rows:
                i = r 
                pivot += 1
                if pivot == cols:
                    return A
        A[[i, r]] = A[[r, i]] # меняем местами строки
        A[r] = A[r] / A[r, pivot] # нормирование строки: делаем опорный элемент равным единице
        for i in range(rows):
            if i != r:
                A[i] = A[i] - A[i, pivot]*A[r] # обнуляем элемент на ведущем столбце
        pivot += 1
    return A


if __name__ == "__main__":
    A = np.array(
        [
            [1, 2, 3],
            [2, 4, 6],
            [1, 1, 1]
        ]    
    )  
    result = rref(A)
    print(f"Исходная форма:\n {A} \n Приведенная к RREF:\n {result}")
