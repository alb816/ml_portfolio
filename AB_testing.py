import pandas as pd
import numpy as np
from scipy import special

# Настраиваем генератор случайных чисел для воспроизводимости
np.random.seed(42)

# Генерируем данные для Группы А (Контрольная)
# Средняя выручка 10$, стандартное отклонение 2$ group_a = np.random.normal(loc=10.0, scale=2.0, size=1000)
group_a = np.random.normal(loc=10.0, scale=2.0, size=1000)
# Генерируем данные для Группы Б (Тестовая) 
# Средняя выручка 10.5$ (наша система должна дать прирост!)
group_b = np.random.normal(loc=10.2, scale=2.0, size=1000)

# Собираем в DataFrame
df = pd.DataFrame({
    'group': ['A'] * 1000 + ['B'] * 1000,
    'revenue': np.concatenate([group_a, group_b])
})

# Основные статистики
n_a = group_a.size
n_b = group_b.size

mean_a = group_a.mean()
mean_b = group_b.mean()

var_a = np.var(group_a, ddof=1)
var_b = np.var(group_b, ddof=1)

error_a = var_a / n_a
error_b = var_b / n_b

# Вычисляем T-статистику
s_e = np.sqrt(error_a + error_b)

t_s = (mean_b - mean_a) / s_e 

if t_s > 1.98:
    print(f"T-статистика: {t_s} > 1.98 =>\n разница значима => отвергаем гипотезу 0\n")
    # Если в выборке по 1000, порог = 1.96
    # Если в выборке по 100, порог = 1.984
    # Если в выборке по 20, порог = 2.042
    # Если в выборке по 5, порог = 2.776
    # Поэтому p-value имеет преимущество: 
    # не нужно искать табличное значение порога для конкретного размера выборки при принятии решения.

# Находим число степеней свободы (DF) для расчета p_value
n = (error_a + error_b)**2
d = (error_a ** 2 / (n_a - 1)) + (error_b ** 2 / (n_b - 1))

df = n / d

# Далее считаем p_value

# Берем модуль T-статистики
abs_t = np.abs(t_s)

# Считаем интеграл нормального распределения (CDF) математически через erf
cdf = 0.5 * (1 + special.erf(abs_t / np.sqrt(2)))

# Двухсторонний p-value
p_value = 2 * (1 - cdf)

if p_value < 0.05:
    print(f"p-value (через нормальное приближение): {p_value:.16f} < 0.05 =>\n отвергаем гипотезу 0. Разница в 0.5 между A и B не случайна")