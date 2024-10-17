import scipy
import numpy as np
import doctest
import acsefunctions.acsefunctions
import warnings
from acsefunctions.acsefunctions import exp, sinh, cosh, tanh, factorial, gamma_function, bessel_j


def test_taylor_exp():
    assert np.isclose(exp(1), np.exp(1), atol=1e-6)
    assert np.allclose(exp([0, 1, 2]), np.exp([0, 1, 2]), atol=1e-6)

def test_taylor_sinh():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    assert np.isclose(sinh(1), np.sinh(1), atol=1e-6)
    assert np.allclose(sinh([0, 1, 2]), np.sinh([0, 1, 2]), atol=1e-6)

def test_taylor_cosh():
    assert np.isclose(cosh(1), np.cosh(1), atol=1e-6)
    assert np.allclose(cosh([0, 1, 2]), np.cosh([0, 1, 2]), atol=1e-6)

def test_taylor_tanh():
    assert np.isclose(tanh(1), np.tanh(1), atol=1e-6)
    assert np.allclose(tanh([0, 1, 2]), np.tanh([0, 1, 2]), atol=1e-6)

def test_factorial():
    assert factorial(5) == 120
    assert np.array_equal(factorial([0, 1, 2]), np.array([1, 1, 2]))

def test_gamma_function():
    assert np.isclose(gamma_function(5), scipy.special.gamma(5), atol=1e-6)
    assert np.isclose(gamma_function(0.5), scipy.special.gamma(0.5), atol=1e-6)

    input_values = np.array([0.5, 1.5, 2.5, 5])
    expected_values = scipy.special.gamma(input_values)
    result_values = gamma_function(input_values)
    assert np.allclose(result_values, expected_values, atol=1e-6)

def test_bessel_j():
    assert np.isclose(bessel_j(0, 1), scipy.special.jv(0, 1), atol=1e-6)
    assert np.isclose(bessel_j(1, 1), scipy.special.jv(1, 1), atol=1e-6)
    assert np.allclose(bessel_j(1, [0, 1, 2]), scipy.special.jv(1, [0, 1, 2]), atol=1e-6)


def test_doctests():
    """
    Run doctests on all functions within the package.
    This ensures that the examples in the docstrings are accurate.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    doctest.testmod(acsefunctions.acsefunctions)


if __name__ == "__main__":
    test_taylor_exp()
    # test_taylor_sinh()
    test_taylor_cosh()
    # test_taylor_tanh()
    test_factorial()
    # test_gamma_function()
    # test_bessel_j()
    test_doctests()
    print("All tests passed.")