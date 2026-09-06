import numpy as np



def jacobi_method(A, b, max_iter=1000, eps=1e-10):
    """
    Функция для решения систем линейных уравнений Ax = b методом якоби
    """

    diag_is_ok = 0 if 0 in np.diag(A) else 1  # для проверки возможности применить метод

    if (
        A.shape[0] == A.shape[1]
        and A.shape[0] == b.shape[0]
        and diag_is_ok  # матрица коэффициентов должна быть квадратной, а диагональ не должна иметь нулей
    ):
        # определяю основные матрицы для итерационого расчета вектора решений x
        x = np.zeros_like(
            b, dtype=np.float64
        )  # начальное приближение для вектора решений
        D = np.diag(np.diag(A))  # диагональная матрица с диагональю от A
        R = (
            A - D
        )  # матрица объединеных вместе верхней и нижней треугольных матриц с нулевой диагональю

        # начало итерационной части метода
        for i in range(max_iter):
            x_new  = np.dot(np.linalg.inv(D), (b - np.dot(R, x)))
            if np.linalg.norm(x_new - x, ord=np.inf) < eps:
                return x_new
            x = x_new
        print("Не удалось достичь заданной точности")
        return x
    else:
        print("Введены неподходящие данные!")


def seidel_method(A, b, max_iter=1000, eps=1e-10):
    """
    Функция для решения систем Ax = b методом зейделя
    """

    diag_is_ok = 0 if 0 in np.diag(A) else 1  # проверка наличия нулей на диагонали

    if A.shape[0] == A.shape[1] and A.shape[0] == b.shape[0] and diag_is_ok:
        x = np.zeros_like(b, dtype=np.float64)  # начальное приближение
        L_D = np.tril(A)  # нижняя треугольная матрица от А с исходной диагональю
        U = A - L_D  # строго верхняя треугольная часть А

        for i in range(max_iter):
            x_new = np.dot(np.linalg.inv(L_D), (b - np.dot(U, x)))
            if np.linalg.norm(x_new - x, ord=np.inf) < eps:
                return x_new
            x = x_new
        print("Не удалось достичь заданной точности")
        return x
    else:
        print("Введены неподходящие данные!")


def gaussian_elimination(A: np.ndarray, b: np.ndarray):
    n = b.shape[0]

    # оптимизация решения через выбор главного элмента на к-ой итерации
    for i in range(n):
        max_row = i + np.argmax(abs(A[i:, i]))  # A[i:, i] возвращает i-й столбец-срез [0:n-i]
        if i != max_row:
            # меняем строки местами
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]

        if np.isclose(A[i, i], 0, atol=1e-12):
            raise ValueError("Матрица вырожденная или почти вырожденная")

        # Обнуление элементов ниже диагонали
        for k in range(i + 1, n):
            factor = (
                A[k, i] / A[i, i]
            )  # factor - это коэффициент элементарного преобразования строки, обнуляющий A[k, i]. Выводится из уравнения A[k, i] - factor*A[i, i] = 0
            A[k, i:] -= (
                factor * A[i, i:]
            )  # применяем элементарное преобразование над строкой A[k, i:], чтобы обнулить A[k, i]
            b[k] -= (
                factor * b[i]
            )  # операция проводится над расширенной матрицей A|b, поэтому делаем то же с b[k]
    x = np.zeros(shape=n)

    # находим вектор x, подставляя с конца
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1 :], x[i + 1 :])) / A[i, i]
    return x

if __name__ == "__main__":
    A = np.array(
        [[4, -1, 0, 1], [-1, 4, -1, 0], [0, -1, 4, -1], [1, 0, -1, 4]], dtype=float
    )
    b = np.array([15, 10, 10, 15], dtype=float)

    print(np.round(gaussian_elimination(A, b), 4))
    print(np.round(seidel_method(A, b), 4))
    print(np.round(jacobi_method(A, b), 4))
