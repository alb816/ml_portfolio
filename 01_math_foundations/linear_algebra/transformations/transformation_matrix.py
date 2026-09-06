import numpy as np

def get_transform_mx(B, C):
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)
    
    # матрица перехода из B в C
    P = np.linalg.inv(C) @ B
    return P

if __name__ == "__main__":
    # базис B
    B = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    # базис С
    C = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1]
    ])

    x_B = np.array([2, 3, 4], float) # координаты х при базисе B

    P = get_transform_mx(B, C)

    x_C = P @ x_B # координаты х при базисе С

    print(f"При базисе \nB = \n{B}\nс координатами {x_B}\nвектор x равен {B@x_B}\n")
    print(f"При базисе \nC = \n{C}\nс координатами {x_C}\nвектор x также равен {C@x_C}")