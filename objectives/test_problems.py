"""Test problems for AdaGrad optimization."""

import numpy as np
from typing import Tuple, Callable

def create_quadratic_problem(
    dimension: int,
    condition_number: float = 10.0
) -> Tuple[Callable, Callable, np.ndarray]:
    """Create a quadratic optimization problem.
    
    Parameters:
    -----------
    dimension : int
        Problem dimension
    condition_number : float
        Condition number of the Hessian matrix
        
    Returns:
    --------
    Tuple[Callable, Callable, np.ndarray]
        - loss_fn: Function that computes the loss
        - grad_fn: Function that computes the gradient
        - opt_point: Optimal point
    """
    # Generate random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(dimension, dimension))
    
    # Generate eigenvalues with specified condition number
    min_eig = 1.0
    max_eig = condition_number
    eigenvalues = np.linspace(min_eig, max_eig, dimension)
    
    # Construct Hessian matrix
    A = Q @ np.diag(eigenvalues) @ Q.T
    
    # Generate random optimal point
    x_opt = np.random.randn(dimension)
    
    def loss_fn(x: np.ndarray) -> float:
        """Compute quadratic loss."""
        diff = x - x_opt
        return 0.5 * diff.T @ A @ diff
    
    def grad_fn(x: np.ndarray) -> np.ndarray:
        """Compute gradient."""
        return A @ (x - x_opt)
    
    return loss_fn, grad_fn, x_opt

def create_ill_conditioned_problem(
    dimension: int,
    condition_number: float = 1000.0
) -> Tuple[Callable, Callable, np.ndarray]:
    """Create an ill-conditioned optimization problem.
    
    Parameters:
    -----------
    dimension : int
        Problem dimension
    condition_number : float
        Condition number of the Hessian matrix
        
    Returns:
    --------
    Tuple[Callable, Callable, np.ndarray]
        - loss_fn: Function that computes the loss
        - grad_fn: Function that computes the gradient
        - opt_point: Optimal point
    """
    # Generate random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(dimension, dimension))
    
    # Generate eigenvalues with specified condition number
    min_eig = 1.0
    max_eig = condition_number
    eigenvalues = np.logspace(np.log10(min_eig), np.log10(max_eig), dimension)
    
    # Construct Hessian matrix
    A = Q @ np.diag(eigenvalues) @ Q.T
    
    # Generate random optimal point
    x_opt = np.random.randn(dimension)
    
    def loss_fn(x: np.ndarray) -> float:
        """Compute quadratic loss."""
        diff = x - x_opt
        return 0.5 * diff.T @ A @ diff
    
    def grad_fn(x: np.ndarray) -> np.ndarray:
        """Compute gradient."""
        return A @ (x - x_opt)
    
    return loss_fn, grad_fn, x_opt 