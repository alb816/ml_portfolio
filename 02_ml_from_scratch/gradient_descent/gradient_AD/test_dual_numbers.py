import os
import sys
import numpy as np

module_path = os.path.dirname(__file__)
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from dual_numbers import DualNumber, gradient


def test_exp_pow_shapes():
    x_vals = np.array([2.0, 3.0])
    der = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = DualNumber(x_vals, der)

    e = d.exp()
    assert e.val.shape == (2,)
    assert e.der.shape == (2, 2)

    p = d ** 2
    assert p.val.shape == (2,)
    assert p.der.shape == (2, 2)


def test_emp_risk_gradient_shape():
    gd_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if gd_dir not in sys.path:
        sys.path.insert(0, gd_dir)
    import gradient_descent as gd

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1.0, 2.0])
    n_params = X.shape[1] + 1

    params = [0.1] * n_params
    grad_dn = gradient(lambda p: gd.emp_risk(p, X, y), params)
    der = grad_dn.der if hasattr(grad_dn, 'der') else grad_dn
    assert np.array(der).shape[-1] == n_params
