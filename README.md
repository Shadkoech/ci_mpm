# acsefunctions

`acsefunctions` is a Python package designed to compute transcendental and special functions using series expansions. This package provides implementations of functions such as exponential, hyperbolic sine, hyperbolic cosine, hyperbolic tangent, Bessel functions, and the gamma function, among others. With a focus on numerical accuracy and performance, this package is ideal for applications in physics and engineering.

## Features

- Compute \( e^x \) (exponential function) using Taylor series
- Compute \( \sinh(x) \), \( \cosh(x) \), and \( \tanh(x) \) (hyperbolic functions) using Taylor series
- Compute the Bessel function \( J_{\alpha}(x) \) from series expansion
- Compute the gamma function \( \Gamma(z) \)
- Support for vectorized inputs using NumPy

## Installation

To install the `acsefunctions` package, clone this repository and use the following instructions:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/acsefunctions.git
   cd acsefunctions

2. Create a conda environment
    `conda env create -f environment.yml`
    `conda activate your-env-name`

3. Install the package in editable mode
    `pip install -e .`

## Usage

### Importing Functions
 ``` from acsefunctions import exp, sinh, cosh, tanh, bessel_j, gamma ```

### Example 1: Calculate the Exponential Function
``` result = exp(1)  # e^1
print(result)  # Output: ~2.71828 
```

### Example 2: Calculate Hyperbolic Functions

``` x = 0.5
sinh_result = sinh(x)
cosh_result = cosh(x)
tanh_result = tanh(x)

print(f"sinh(0.5): {sinh_result}, cosh(0.5): {cosh_result}, tanh(0.5): {tanh_result}") 
```


### Example 3: Calculate Bessel Function

``` alpha = 0.5  # Order of the Bessel function
x = 1.0
bessel_result = bessel_j(alpha, x)
print(f"J_{alpha}(1.0): {bessel_result}")
```

## Testing
To run the test suite for the package, use:
    - `pytest tests/`


## Contributing
Contributions are welcome! Please open an issue or submit a pull request to discuss your changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
* Inspired by various numerical methods and special functions.
* Utilized NumPy for numerical operations.