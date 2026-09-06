import numpy as np

try:
    from .dual_numbers import gradient, DualNumber
except ImportError:
    from dual_numbers import gradient, DualNumber



def emp_risk(params, X, y):
    """
    Вычислить среднеквадратичную ошибку (MSE) с помощью AD в прямом режиме.

    Параметры
    ----------
    params : список объектов DualNumber
        Параметры модели: [w_1, w_2, ..., w_d, b], где w_i — весовые коэффициенты
        и b — смещение. Каждый параметр представляет собой объект DualNumber, содержащий как значение, так и производную.
    X : np.ndarray, форма (n_samples, n_features)
        Входные признаки.
    y : np.ndarray, форма (n_samples,)
        Целевые метки.

    Возвращается
    -------
    DualNumber
        Значение функции потерь и градиент по всем параметрам.
        .val: скалярное среднеквадратическое отклонение, .der: вектор градиента формы (n_params,)
    """
    w_dual = params[:-1]
    b_dual = params[-1]
    
    # Собираем значения весов
    w_vals = np.array([p.val for p in w_dual])
    b_val = b_dual.val
    
    # Предсказания через матричное умножение (быстро)
    y_pred_vals = X @ w_vals + b_val
    
    # Теперь считаем производную. 
    # В Forward Mode AD: der_y = X @ der_w + der_b
    w_ders = np.array([p.der for p in w_dual])
    b_der = b_dual.der
    
    y_pred_ders = X @ w_ders + b_der
    
    # Упаковываем в DualNumber
    predictions = DualNumber(y_pred_vals, y_pred_ders)
    
    # MSE
    errors = predictions - y
    loss = (errors ** 2).sum() / len(y)
    return loss


def gradient_descent(X, y, rate=0.001, n_iter=600):
    """
    Оптимизация методом градиентного спуска с использованием автоматического дифференцирования в прямом режиме.

    Параметры
    ----------
    X : np.ndarray, форма (n_samples, n_features)
        Входная матрица признаков.
    y : np.ndarray, форма (n_samples,)
        Целевой вектор (непрерывные значения для регрессии).
    rate : float, по умолчанию 0.001
        Скорость обучения для обновления параметров.
    n_iter : int, по умолчанию 600
        Количество итераций градиентного спуска.

    Возвращается
    -------
    w : np.ndarray, форма (n_features,)
        Обученные весовые коэффициенты.
    b : float
        Обученная смещенная оценка.
    final_mse : float
        Среднеквадратическая ошибка по конечным параметрам.
    """
    num_of_feats = X.shape[1]
    params = np.random.randn(num_of_feats + 1) * 0.01

    for t in range(1, n_iter + 1):
        objective = lambda p: emp_risk(p, X, y)
        grad_obj = gradient(objective, params.tolist())
        grad_values = grad_obj.der

        # Обновление параметров
        params = params - rate * grad_values

    w, b = params[:-1], params[-1]
    final_mse = np.mean((X @ w + b - y) ** 2)
    return w, b, final_mse



def predict(x, w, b):
    """
    Делает прогноз, используя обученные веса и смещение.

    Параметры
    ----------
    x : массив, форма (n_features,)
        Входная выборка.
    w : np.ndarray, форма (n_features,)
        Обученные веса.
    b : float
        Обученная смещенная оценка.

    Возвращается
    -------    
    pred : float
        Линейное предсказание (без пороговых значений для регрессии).
    """
    return np.dot(w, x) + b


if __name__ == "__main__":
    # Пример: линейная регрессия и градиентный спуск 
    # с использованием дуальных чисел для расчета градиентов
    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0]
    ])
    # Целевая переменная: y = 2*x1 + 3*x2 + noise
    y = np.array([8.0, 13.0, 18.0, 23.0, 28.0])

    w, b, mse = gradient_descent(X, y, rate=0.01, n_iter=100)
    print(f"Веса: {w}")
    print(f"Смещение: {b}")
    print(f"MSE: {mse:.6f}")

    # Предсказание на новой выборке
    test_sample = np.array([2.5, 3.5])
    pred = predict(test_sample, w, b)
    print(f"Предсказание для {test_sample}: {pred:.4f}")
