import numpy as np


def corr(x: np.ndarray, y: np.ndarray) -> float:
    """Вычисляет коэффициент корреляции Пирсона между двумя векторами.

    Args:
        x: Первый одномерный массив данных.
        y: Второй одномерный массив данных.

    Returns:
        float: Значение корреляции в диапазоне [-1.0, 1.0].
               Возвращает 0.0, если один из векторов константный.
    """
    # Центрируем векторы (вычитаем математическое ожидание)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    # Находим ковариацию через среднее произведение центрированных значений
    covariance = np.mean(x_centered * y_centered)

    std_x = np.std(x)
    std_y = np.std(y)

    # Защита от деления на ноль (ZeroDivisionError) для константных признаков
    if std_x == 0 or std_y == 0:
        return 0.0

    return float(covariance / (std_x * std_y))


def correlation_matrix_iterative(X: np.ndarray) -> np.ndarray:
    """Вычисляет корреляционную матрицу с использованием циклов (покомпонентно).

    Args:
        X: Двумерный массив размера (n_features, n_samples),
           где каждая строка представляет собой отдельную переменную.

    Returns:
        np.ndarray: Квадратная матрица корреляции размера (n_features,
        n_features).
    """
    n_feats = X.shape[0]
    cor_matrix = np.zeros((n_feats, n_feats))

    for i in range(n_feats):
        for j in range(n_feats):
            cor_matrix[i, j] = corr(X[i], X[j])

    return cor_matrix


def correlation_matrix_vectorized(X: np.ndarray) -> np.ndarray:
    """Оптимизированное (векторизованное) вычисление корреляционной матрицы.

    Данный метод работает без явных циклов `for` с помощью матричных операций
    и векторизации NumPy, что критически важно для работы с Big Data.

    Args:
        X: Двумерный массив размера (n_features, n_samples).

    Returns:
        np.ndarray: Квадратная матрица корреляции.
    """
    # 1. Центрируем данные по строкам (вычитаем среднее каждой строки)
    X_centered = X - np.mean(X, axis=1, keepdims=True)

    # 2. Вычисляем матрицу ковариации через матричное умножение
    # Делим на количество колонок (samples), чтобы получить среднее
    n_samples = X.shape[1]
    covariance_matrix = np.dot(X_centered, X_centered.T) / n_samples

    # 3. Вычисляем СКО для каждой строки
    stds = np.std(X, axis=1, keepdims=True)

    # Матрица произведения СКО (знаменатель формулы Пирсона)
    std_outer = np.dot(stds, stds.T)

    # Заменяем нули на 1, чтобы избежать деления на ноль (в местах, где std == 0)
    # В итоговой матрице эти значения останутся нулевыми, так как ковариация там тоже 0
    std_outer[std_outer == 0] = 1.0

    return covariance_matrix / std_outer


if __name__ == "__main__":
    # Тестовый набор данных (демонстрация сильной прямой и обратной корреляции)
    A = np.array(
        [
            [1, -1, -1, -1],
            [-1, 1, 1, 1],
            [-1, 1, 1, 1],
            [-1, 1, 1, 1],
        ],
        dtype=float,
    )

    print("=== Тестирование кастомных реализаций корреляции ===")

    # Тест 1: Итеративный подход
    res_iterative = correlation_matrix_iterative(A)
    print("\n1. Результат итеративного метода (через циклы for):")
    print(res_iterative)

    # Тест 2: Векторизованный подход
    res_vectorized = correlation_matrix_vectorized(A)
    print("\n2. Результат векторизованного метода (матричный вид):")
    print(res_vectorized)

    # Тест 3: Проверка эталоном NumPy
    res_numpy = np.corrcoef(A)
    print("\n3. Эталонный результат библиотеки NumPy (np.corrcoef):")
    print(res_numpy)

    # Автоматический тест (Assert) на совпадение матриц
    assert np.allclose(
        res_vectorized, res_numpy
    ), "Векторизованный метод считает неверно!"
    assert np.allclose(
        res_iterative, res_numpy
    ), "Итеративный метод считает неверно!"
    print("\n[УСПЕХ] Все тесты пройдены! Результаты полностью совпадают.")
