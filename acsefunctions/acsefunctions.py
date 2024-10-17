"""
This module contains all the functions used in this project
"""

import numpy as np
import math


def exp(x, N=20):
    """
    Approximate e^x using the Taylor series expansion.

    Parameters:
        x (float or array-like): The point(s) at which to evaluate e^x.
        N (int): Number of terms in the Taylor series.

    Returns:
        float or ndarray: The approximation of e^x.

    Examples:
    >>> taylor_exp(1)
            2.7182815255731922
    >>> taylor_exp([0, 1, 2])
            array([1.0, 2.71828153, 7.3890561 ])
    """
    x = np.asarray(x, dtype=float)
    result = np.ones_like(x)
    term = np.ones_like(x)
    for n in range(1, N + 1):
        term *= x / n
        result += term
    return result


def sinh(x, N=100):
    """
    Approximate sinh(x) using the Taylor series expansion.

    Parameters:
        x (float or array-like): The point(s) at which to evaluate sinh(x).
        N (int): Number of terms in the Taylor series.

    Returns:
        float or ndarray: The approximation of sinh(x).

    Examples:
    >>> sinh(1)
            1.175201177598364
    >>> sinh([0, 1, 2])
            array([0.0, 1.17520118, 3.62686041])
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    for n in range(N):
        term = (x ** (2 * n + 1)) / factorial(2 * n + 1)
        result += term
    return result


def cosh(x, N=20):
    """
    Approximate cosh(x) using the Taylor series expansion.

    Parameters:
        x (float or array-like): The point(s) at which to evaluate cosh(x).
        N (int): Number of terms in the Taylor series.

    Returns:
        float or ndarray: The approximation of cosh(x).

    Examples:
    >>> taylor_cosh(1)
            1.5430806348152437
    >>> taylor_cosh([0, 1, 2])
            array([1.        , 1.54308063, 3.76219569])
    """
    x = np.asarray(x, dtype=float)
    result = np.ones_like(x)
    term = np.ones_like(x)
    for n in range(1, N + 1):
        term *= (x**2) / ((2 * n - 1) * (2 * n))
        result += term
    return result


def tanh(x, N=20):
    """
    Approximate tanh(x) as the ratio of sinh(x) to cosh(x).

    Parameters:
        x (float or array-like): The point(s) at which to evaluate tanh(x).
        N (int): Number of terms in the Taylor series.

    Returns:
        float or ndarray: The approximation of tanh(x).

    Examples:
    >>> taylor_tanh(1)
            0.7615941559557649
    >>> taylor_tanh([0, 1, 2])
            array([0.        , 0.76159416, 0.96402758])
    """
    return sinh(x, N) / cosh(x, N)


def factorial(n):
    """
    Compute the factorial of a non-negative integer n.
    Supports scalar and vector inputs.

    Parameters:
        n (int or array-like): The number(s) to compute the factorial of.

    Returns:
        int or ndarray: The factorial of n.

    Examples:
    >>> factorial(5)
            120
    >>> factorial([0, 1, 2, 3])
            array([ 1,  1,  2,  6])
    """
    n = np.asarray(n, dtype=np.int64)

    # Check for negative values
    if np.any(n < 0):
        raise ValueError("Factorial is not defined for negative values.")

    # Special case: return 1 for n == 0
    if np.isscalar(n):
        if n == 0:
            return 1
    else:
        result = np.ones_like(n, dtype=np.float64)
        result[n == 0] = 1

    # Calculate the factorial using a vectorized approach
    max_n = np.max(n)
    if max_n == 0:
        return result

    # Create an array of factorials up to max_n
    factorials = np.ones(max_n + 1, dtype=np.float64)
    for i in range(2, max_n + 1):
        factorials[i] = factorials[i - 1] * i

    # Map the computed factorial values to the input n
    if np.isscalar(n):
        return factorials[n]
    else:
        return factorials[n]


def gamma_function(num):
    """
    Compute the gamma function Γ(z) using a mix of recursive calculation and Stirling's approximation.

    Parameters:
        z (float or array-like): The input(s) to the gamma function (z > 0).

    Returns:
        float or ndarray: The approximate value(s) of Γ(z).

    Examples:
    >>> gamma_function(5)
    24.0
    >>> gamma_function([0.5, 1.5, 5])
    array([1.77245385, 0.88622693, 24.])

    Raises:
        ValueError: If z <= 0.
        OverflowError: If z > 171.5.
        NotImplementedError: If z is not an integer or half-integer.
    """
    num = np.asarray(num, dtype=np.float64)

    if np.any(num <= 0):
        raise ValueError("Gamma function is only defined for z > 0.")

    if np.any(num > 171.5):
        raise OverflowError("Gamma function input exceeds overflow limit.")

    result = np.empty_like(num, dtype=np.float64)

    is_integer = np.isclose(num % 1, 0)
    is_half_integer = np.isclose(num % 1, 0.5)

    # Handle scalar input (0-dimensional arrays)
    if np.ndim(num) == 0:
        if is_half_integer:
            return gamma_recursive(num)
        elif is_integer:
            return gamma_recursive(num)
        else:
            return stirling_approximation(num)

    # Loop for array input
    for idx in range(num.size):
        value = num[idx]
        if is_half_integer[idx]:
            result[idx] = gamma_recursive(value)
        elif is_integer[idx]:
            result[idx] = gamma_recursive(value)
        else:
            result[idx] = stirling_approximation(value)

    return result.item() if result.size == 1 else result


def stirling_approximation(z):
    """
    Stirling's approximation for the gamma function for non-integer, non-half-integer values.

    Parameters:
        z (float): The input value for approximation.

    Returns:
        float: The approximate value of the gamma function using Stirling's approximation.
    """
    return math.sqrt(2 * math.pi / z) * (z / math.e) ** z


def gamma_recursive(num):
    """
    Recursive calculation for the gamma function for integers and half-integers.

    Parameters:
        num (float): The input value (integer or half-integer).

    Returns:
        float: The gamma function value for the input.
    """
    if num == 0.5:
        return math.sqrt(math.pi)
    elif num == 1:
        return 1.0
    else:
        return (num - 1) * gamma_recursive(num - 1)


def bessel_j(alpha, x, N=20):
    """
    Compute the Bessel function J_alpha(x) using a truncated series expansion.
    Supports scalar and vector inputs.

    Parameters:
        alpha (float): The order of the Bessel function.
        x (float or array-like): The point(s) at which to evaluate the Bessel function.
        N (int): The number of terms in the series expansion (default is 20).

    Returns:
        float or ndarray: The approximate value of J_alpha(x).

    Examples:
    >>> bessel_j(0, 1)
            0.7651976865579666
    >>> bessel_j(1, [0, 1, 2])
            array([0.        , 0.44005059, 0.57672481])
    """
    x = np.asarray(x, dtype=np.float64)

    if np.any(x < 0):
        raise ValueError("Bessel function is defined for x >= 0.")

    result = np.zeros_like(x, dtype=np.float64)

    for m in range(N):
        term = (
            ((-1) ** m)
            / (factorial(m) * gamma_function(m + alpha + 1))
            * (x / 2) ** (2 * m + alpha)
        )
        result += term

    return result
