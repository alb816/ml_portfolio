import numpy as np


def translate_object(points, tx, ty):

	points = np.array(points)

	T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
	
	n_points = len(points)
	homogeneous = np.ones((n_points, 3))
	homogeneous[:, :2] = points
	
	translated_points = (T @ homogeneous.T).T
	
	return translated_points[:, :2].tolist()


if __name__ == "__main__":
	triangle = np.array([[0, 0], [1, 0], [0.5, 1]])
	tx, ty = 2, 3
	print(translate_object(triangle, tx, ty))


