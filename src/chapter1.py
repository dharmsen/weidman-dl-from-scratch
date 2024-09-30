import numpy as np
from typing import Callable, List

Array_Function = Callable[[ndarray], ndarray]
Chain = List[Array_Function]

def square(x: ndarray) -> ndarray:
    """
    Square each element in the input ndarray.
    """
    return np.power(x, 2)

def leaky_relu(x: ndarray) -> ndarray:
    """
    Apply 'Leaky ReLU' function to each element in ndarray.
    """
    return np.maximum(0.2 * x, x)

def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          delta: float = 0.001) -> ndarray:
    """
    Evaluates the derivative of a function 'func' at every element in the "input_" array.
    """
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

def chain_length_2(chain: Chain,
                   a: ndarray) -> ndarray:
    """
    Evaluates two functions in a row, in a 'Chain'.
    """
    assert len(chain) == 2, "Length of input 'chain' should be 2"

    f1 = chain[0]
    f2 = chain[1]
    return f2(f1(x))

def sigmoid(x: ndarray) -> ndarray:
    """
    Apply the sigmoid function to each element in the input ndarray.
    """
    return 1 / (1+ np.exp(-x))

def chain_deriv_2(chain: Chain,
                  input_range: ndarray) -> ndarray:
    """
    Uses the chain rule to copmute the derivative of two nested functions:
    (f2(f1(x))' = f2'(f1(x)) * f1'(x)
    """
    assert len(chain) == 2, "This function requires 'Chain' objects of length 2"
    assert input_range.ndim == 1, "This function requires a 1 dimensional ndarray as input_range"

    f1 = chain[0]
    f2 = chain[1]

    f1_of_x = f1(input_range)
    df1dx = deriv(f1, input_range)
    df2du = deriv(f2, f1_of_x)

    return df1dx * df2du

def chain_deriv_3(chain, Chain,
                  input_range: ndarray) -> ndarray:
    """
    Uses the chain rule to compute the derivative of three nested functions:
    (f3(f2(f1)))' = f3'(f2(f1(x))) * f2'(f1(x)) * f1'(x)
    """

    assert len(chain) == 3, "This function requires 'Chain' objects to have length 3"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    f1_of_x = f1(input_range)
    f2_of_x = f2(f1_of_x)
    df3du = deriv(f3, f2_of_x)
    df2du = deriv(f2, f1_of_x)
    df1dx = deriv(f1, input_range)

    return df3du * df2du * df1dx

def multiple_inputs_add(x: ndarray,
                        y: ndarray,
                        sigma: Array_Function) -> float:
    """
    Function with multiple inputs and addition, forward pass.
    """
    assert x.shape == y.shape

    a = x + y
    return sigma(a)

def multiple_inputs_add_backward(x: ndarray,
                                 y: ndarray,
                                 sigma: Array_Function) -> float:
    """
    Computes the derivative of this simple function with respect to both inputs.
    """
    a = x + y

    dsda = deriv(sigma, a)
    dadx, dady = 1, 1

    return dsda * dadx, dsda * dady

def multiple_inputs_multiply(x: ndarray,
                             y: ndarray,
                             sigma: Array_Function) -> float:
    """
    Function with multiple inputs and multiplication, forward pass.
    """
    a = x * y
    return sigma(a)

def multiple_inputs_multiply_backward(x: ndarray,
                                      y: ndarray,
                                      sigma: Array_Function) -> float:
    """
    Computes the derivative of this simple function with respect to both inputs.
    """
    a = x * y

    dsda = deriv(sigma, a)
    dadx = y
    dady = x

    return dsda * dadx, dsda * dady

def matmul_forward(X: ndarray,
                   W: ndarray) -> ndarray:
    """
    Computes the forward pass of a matrix multiplication.
    """
    assert X.shape[1] == W.shape[0], \
    """
    For matrix multiplication, the number of columns in the first array should
    match the number of rows in the second; instead the number of columns in the
    first array is {0} and the number of rows in the second array is {1}.
    """.format(X.shape[1], W.shape[0])

    N = np.dot(X, W)

    return N

def matmul_backward_first(X: ndarray,
                          W: ndarray) -> ndarray:
    """
    Computes the backward pass of a matrix multiplication with respect to the
    first argument.
    """
    dNdX = np.transpose(W, (1, 0))

    return dndX

def matrix_forward_extra(X: ndarray,
                         W: ndarray,
                         sigma: Array_Function) -> ndarray:
    """
    Computes the forward pass of a function involving matrix multiplication,
    one extra function.
    """
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    S = sigma(N)

    return S

def matrix_function_backward_1(X: ndarray,
                               W: ndarray,
                               sigma: Array_Function) -> ndarray:
    """
    Computes the derivative of our matrix function with respect to
    the first element.
    """
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    S = sigma(N)
    dSdN = deriv(sigma, N)
    dNdX = np.transpose(W, (1, 0))

    return np.dot(dSdN, dNdX)

def matrix_function_forward_sum(X: ndarray,
                                W: ndarray,
                                sigma: Array_Function) -> float:
    """
    Computing the result of the forward pass of this function with
    input ndarrays X and W and function sigma.
    """
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    S = sigma(N)
    L = np.sum(S)

    return L

def matrix_function_backward_sum_1(X: ndarray,
                                   W: ndarray,
                                   sigma: Array_Function) -> ndarray:
    """
    Compute derivative of matrix function with a sum with respect to the
    first matrix input.
    """
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    S = sigma(N)
    L = np.sum(S)

    dLdS = np.ones_like(S)
    dSdN = deriv(sigma, N)
    dLdN = dLdS * dSdN
    dNdX = np.transpose(W, (1, 0))
    dLdX = np.dot(dSdN, dNdX)

    return dLdX
