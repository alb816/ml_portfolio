import numpy as np


def stable_hash(s):
    h = 0
    for c in str(s):
        h =(h * 31 + ord(c)) % 2**31
    return h # хэш детерминирован
        
def to_bipolar_hv(dim, seed):
    """
    Конвертирует значение в биполярный
    гипервектор размерности dim через seed.
    """
    rng = np.random.RandomState(seed % (2**32))
    return rng.choice([1, -1], size=dim)

def get_random_seeds(row:dict):
    seeds = [stable_hash(i) for i in row.keys()]
    random_seeds = dict(zip(row.keys(), seeds))
    return random_seeds

def row_to_hv(row, dim, random_seeds):
    """
    Переводит объект row обучающей выборки с n признаками 
    в гипервектор с кол-вом "признаков" равным dim.
    При этом сначала суммируются все "подвекторы" вида hv_feat * hv_val, 
    затем полученный вектор нормализуется к значениям 1 и -1.
    """
    bundled = np.zeros(dim, dtype=int)
    
    for feat, val in row.items():
        feat_seed = random_seeds[feat]
        hv_feat = to_bipolar_hv(dim, feat_seed)
        
        val_seed = (feat_seed + stable_hash(val)) % (2**31)
        hv_val = to_bipolar_hv(dim, val_seed)

        # делаем bind (*) и bundle (+)
        bundled += hv_feat * hv_val
    return np.where(bundled >= 0, 1, -1)


if __name__ == "__main__":
    dim = 10000
    row = {"name":"Alex", "age":30}
    random_seeds = get_random_seeds(row)

    print(row_to_hv(row, dim, random_seeds))