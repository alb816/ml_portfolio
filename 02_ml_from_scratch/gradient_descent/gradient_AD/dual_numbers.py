from __future__ import annotations
from typing import Union
import numpy as np


ArrayLike = Union[float, int, np.ndarray]


class DualNumber:
    """
    Класс дуальных чисел для демонстрации forward-mode автоматического
    дифференцирования (AD).

    Концепция:
        дуальное число представляет собой a + b * eps, где eps^2 = 0.
        val хранит значение функции a, der - тензор/вектор производных b.

    Контракт по формам (важно для векторизованных операций):
    - val может быть скаляром (ndim == 0) или 1D numpy-массивом формы (m,),
      где m - число примеров/элементов.
    - der для скалярного val обычно 1D-вектор формы (n_params,),
      для векторного val - 2D-массив формы (m, n_params).

    Методы перегружают арифметику (+, -, *, /, **, exp, sum) и поддерживают
    корректное broadcast-поведение для комбинаций скаляр/вектор.

    Пример использования (скалярная функция):
        x = DualNumber(2.0, der=1.0)
        y = x ** 2       # val = 4.0, der = 4.0

    Пример использования (для получения градиента функции нескольких
    переменных): см. gradient в этом модуле - каждая входная переменная
    представляется как DualNumber(val, der = единичный вектор).
    """

    val: np.ndarray
    der: np.ndarray

    def __init__(self, val: ArrayLike, der: ArrayLike = 0.0) -> None:
        self.val = np.asarray(val, dtype=float)
        self.der = np.asarray(der, dtype=float)

    @staticmethod
    def _ensure_dual(other: ArrayLike | DualNumber) -> DualNumber:
        if isinstance(other, DualNumber):
            return other

        return DualNumber(other)

    def __add__(self, other: ArrayLike | DualNumber) -> DualNumber:
        other = self._ensure_dual(other)
        return DualNumber(self.val + other.val, self.der + other.der)

    def __radd__(self, other: ArrayLike | DualNumber) -> DualNumber:
        return self.__add__(other)

    def __sub__(self, other: ArrayLike | DualNumber) -> DualNumber:
        other = self._ensure_dual(other)
        return DualNumber(self.val - other.val, self.der - other.der)

    def __rsub__(self, other: ArrayLike | DualNumber) -> DualNumber:
        other = self._ensure_dual(other)
        return other.__sub__(self)

    def __mul__(self, other: ArrayLike | DualNumber) -> DualNumber:
        other = self._ensure_dual(other)

        value = self.val * other.val

        # vector case: bring values to column shape (m,1) so (m,n) * (m,1) broadcasts to (m,n)
        if np.ndim(value) > 0:
            v1 = np.expand_dims(
                self.val, axis=-1
            )  # expand_dims добавляет размерность в указанную позицию. В данном случае в последнюю позицию т.е. -1
            v2 = np.expand_dims(other.val, axis=-1)
            derivative = self.der * v2 + v1 * other.der
        else:
            # scalar case
            derivative = self.der * other.val + self.val * other.der

        return DualNumber(value, derivative)

    def __rmul__(self, other: ArrayLike | DualNumber) -> DualNumber:
        return self.__mul__(other)

    def __pow__(self, n) -> DualNumber:
        # (x^n)' = n * x^(n-1) * x'
        value = self.val**n
        coeff = n * (self.val ** (n - 1))
        if np.ndim(self.val) > 0:
            coeff = np.expand_dims(coeff, axis=-1)
        derivative = coeff * self.der
        return DualNumber(value, derivative)

    def __truediv__(self, other: ArrayLike | DualNumber) -> DualNumber:
        if not isinstance(other, DualNumber):
            return self * (1 / other)

        value = self.val / other.val
        if np.ndim(value) > 0:
            v1 = np.expand_dims(self.val, axis=-1)
            v2 = np.expand_dims(other.val, axis=-1)
            derivative = (self.der * v2 - v1 * other.der) / (v2**2)
        else:
            derivative = (self.der * other.val - self.val * other.der) / (other.val**2)

        return DualNumber(value, derivative)

    def exp(self) -> DualNumber:
        value = np.exp(self.val)
        if np.ndim(self.val) > 0:
            coeff = np.expand_dims(value, axis=-1)
        else:
            coeff = value
        derivative = coeff * self.der
        return DualNumber(value, derivative)

    def sum(self) -> DualNumber:
        return DualNumber(np.sum(self.val), np.sum(self.der, axis=0))


def gradient(func, params: list):
    """
    Вычислить функцию с использованием дуальных чисел для автоматического дифференцирования.

    Параметры
    ----------    
    func : callable
        Функция, принимающая список объектов DualNumber и возвращающая DualNumber.
    params : list of float
        Исходные вещественные значения параметров модели.

    Возвращается
    -------
    DualNumber
        Объект дуального числа, содержащий значение функции (.val) и её градиент (.der).
    """

    n_params = len(params)

    dual_params = [
        DualNumber(val, der=np.eye(n_params)[i]) for i, val in enumerate(params)
    ]  # создается массив

    return func(dual_params)


if __name__ == "__main__":
    # Неболшая демонстрация
    def f_list(params):
        x, y = params
        return x * y + x**2

    grad = gradient(f_list, [2.0, 3.0])
    print("Градиент от f(x,y)=x*y + x^2 в точке (2,3):", grad.der)
