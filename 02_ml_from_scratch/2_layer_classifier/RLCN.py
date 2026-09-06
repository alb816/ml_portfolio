import os

# 1. Отключаем оптимизацию oneDNN (как просит само сообщение)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. Понижаем уровень логов самого TensorFlow (скрывает INFO и WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 3. Скрываем системные логи Abseil / gRPC
os.environ['GRPC_VERBOSITY'] = 'NONE'

import numpy as np
import pandas as pd



class RLCN:
    """Row-wise Logical Conjunctive Network"""

    def __init__(self, n_feats, n_iter=1000, alpha=0.01):
        self.w = np.random.randn(n_feats + 1) * 0.1
        self.s = np.random.randn(6) * 0.1
        self.n_iter = n_iter
        self.alpha = alpha

    def sigmoid(self, v: np.ndarray, w: np.ndarray):
        b = w[0]
        w = w[1:]
        z = v @ w + b
        z = np.clip(z, -50, 50)
        return 1 / (1 + np.exp(-z))

    def bceloss(self, p, y):
        eps = 1e-10
        p = np.clip(p, eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    def get_probabilities(self, X: np.ndarray):
        z = X.reshape(-1, 28) @ self.w[1:] + self.w[0]
        z = np.clip(z, -50, 50)
        p = 1 / (1 + np.exp(-z)) 
        return p.reshape(-1, 28) if X.ndim == 3 else p

    def get_gradient(self, X, p, y_pred, y):
        eps = 1e-10
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        
        # 1. Производная лосса по выходу сети F
        dL_dF = (y_pred_clipped - y) / (y_pred_clipped - y_pred_clipped**2)
        
        grad_w = np.zeros_like(self.w)
        grad_s = np.zeros_like(self.s)
        
        # Массив для накопления ошибок, пришедших на каждую вероятность p[i]
        dL_dp = np.zeros_like(p)
        
        # 2. Повторяем логику КНФ для восстановления промежуточных l_i и r_i
        for i in range(p.shape[0] - 1): 
            v = np.array([p[i], p[i + 1]])
            
            # Считаем выходы сигмоид логического слоя
            l_i = self.sigmoid(v, self.s[:3])
            r_i = self.sigmoid(v, self.s[3:])
            
            # Производная F по выходу левой и правой сигмоиды текущего шага
            dF_dl_i = y_pred / (l_i + eps) * r_i
            dF_dr_i = y_pred / (r_i + eps) * l_i
            
            # Полная ошибка лосса, дошедшая до левой и правой сигмоиды
            dL_dl_i = dL_dF * dF_dl_i
            dL_dr_i = dL_dF * dF_dr_i
            
            # Производные по линейной части z = v @ w + b
            dl_dz_left = l_i * (1 - l_i)
            dr_dz_right = r_i * (1 - r_i)
            
            # Градиенты для весов s (включая bias на позиции 0)
            grad_s[:3] += dL_dl_i * dl_dz_left * np.append(1.0, v)
            grad_s[3:] += dL_dr_i * dr_dz_right * np.append(1.0, v)
            
            # Переносим ошибку дальше назад — на элементы вектора p
            # Отрезаем bias-составляющую весов, берем только веса при p[i] и p[i+1]
            w_left_inputs = self.s[1:3]
            w_right_inputs = self.s[4:6]
            
            # Влияние на p[i] (первый элемент пары v)
            dL_dp[i] += dL_dl_i * dl_dz_left * w_left_inputs[0] + dL_dr_i * dr_dz_right * w_right_inputs[0]
            # Влияние на p[i+1] (второй элемент пары v)
            dL_dp[i + 1] += dL_dl_i * dl_dz_left * w_left_inputs[1] + dL_dr_i * dr_dz_right * w_right_inputs[1]
            
        # 3. Считаем градиент для базовых весов w признаков пикселей
        for i in range(X.shape[0]):
            # dL_dp[i] — ошибка, пришедшая от логического слоя к i-й строке
            # Умножаем на производную базовой сигмоиды этой строки
            dp_dz = p[i] * (1 - p[i])
            
            # Накапливаем градиент весов w (на позиции 0 стоит bias, дальше — пиксели строки X[i])
            grad_w += dL_dp[i] * dp_dz * np.append(1.0, X[i])
            
        return grad_w, grad_s

    def cnf(self, P: np.ndarray):
        F = 1.0
        for i in range(P.shape[0] - 1):
            v = np.array([P[i], P[i + 1]])
            left_conjunct = self.sigmoid(v, self.s[:3])
            right_conjunct = self.sigmoid(v, self.s[3:])
            F *= left_conjunct * right_conjunct
        return F

    def forward(self, X: np.ndarray):
        p = self.get_probabilities(X)
        return self.cnf(p), p

    def predict(self, X: np.ndarray):
        f = self.forward(X)[0]
        return 1 if f >= 0.5 else 0

    def fit(self, X, y):
        n_samples = y.shape[0]
        batch_size = 64

        for epoch in range(self.n_iter):
            loss_sum = 0

            grad_w_batch = np.zeros_like(self.w)
            grad_s_batch = np.zeros_like(self.s)

            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]
                y_pred, p = self.forward(x_i)

                loss_sum += self.bceloss(y_pred, y_i)

                grad_w_x, grad_s_x = self.get_gradient(x_i, p,  y_pred, y_i)

                grad_w_batch += grad_w_x
                grad_s_batch += grad_s_x
                
                if (i + 1) % batch_size == 0 or i == n_samples - 1:
                    current_batch_size = batch_size if (i + 1) % batch_size == 0 else n_samples % batch_size

                    grad_w_batch /= current_batch_size
                    grad_s_batch /= current_batch_size

                    grad_w_batch = np.clip(grad_w_batch, -1, 1)
                    grad_s_batch = np.clip(grad_s_batch, -1, 1)

                    print(grad_w_batch)
                    print(grad_s_batch)

                    self.w -= self.alpha * grad_w_batch
                    self.s -= self.alpha * grad_s_batch
                
                    grad_w_batch = np.zeros_like(self.w)
                    grad_s_batch = np.zeros_like(self.s)
        print(self.w, self.s)


from keras.datasets import mnist
import numpy as np
import os

# Путь для сохранения
DATA_PATH = 'mnist_data.npz'

# Проверяем, есть ли уже сохраненные данные
if os.path.exists(DATA_PATH):
    print("Загрузка из локального файла...")
    data = np.load(DATA_PATH)
    x_train = data['x_train'][:500]
    y_train = data['y_train'][:500]
    x_test = data['x_test']
    y_test = data['y_test']
else:
    print("Скачивание MNIST (только один раз)...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Сохраняем для будущих запусков
    np.savez(DATA_PATH, 
             x_train=x_train, y_train=y_train,
             x_test=x_test, y_test=y_test)
    print(f"Данные сохранены в {DATA_PATH}")

# Для бинарной классификации (0 и 1)
mask_train = (y_train == 0) | (y_train == 1)
mask_test = (y_test == 0) | (y_test == 1)

X_train_binary = x_train[mask_train]
y_train_binary = y_train[mask_train]
X_test_binary = x_test[mask_test]
y_test_binary = y_test[mask_test]

# Нормализация (важно для сходимости)
X_train_binary = X_train_binary.astype(np.float32) / 255.0
X_test_binary = X_test_binary.astype(np.float32) / 255.0
print(X_train_binary.shape)
print(y_train_binary[1])


clf = RLCN(X_train_binary.shape[2], n_iter=5, alpha=0.1)
print(clf.w)
print(clf.s)
print(clf.predict(X_train_binary[1]))
clf.fit(X_train_binary, y_train_binary)
print(clf.predict(X_train_binary[1]))