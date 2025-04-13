"""AdaGrad optimization algorithms based on provided specifications."""

import numpy as np
import time
from typing import Tuple, List, Callable

def diagonal_adagrad(
    loss_fn: Callable,
    grad_fn: Callable,
    initial_params: np.ndarray,
    learning_rate: float = 0.1,
    epsilon: float = 1e-8, # Epsilon for diagonal accumulator stability
    max_iterations: int = 1000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, List[float], List[np.ndarray], float]:
    """Diagonal AdaGrad implementation.
    
    Updates parameters using: theta_t+1 = theta_t - eta * G_t^-1/2 * grad_t
    where G_t = diag(sum(g_k^2)) + epsilon * I.
    
    Args:
        loss_fn: Function that computes the loss.
        grad_fn: Function that computes the gradient.
        initial_params: Initial parameter values.
        learning_rate: Learning rate (eta).
        epsilon: Small constant added to the diagonal accumulator for stability.
        max_iterations: Maximum number of iterations.
        tol: Convergence tolerance for gradient norm.
        
    Returns:
        Tuple: (final parameters, loss history, parameter trajectory, runtime).
    """
    start_time = time.time()
    d = len(initial_params)
    theta = initial_params.copy()
    # G accumulates sum of squared gradients element-wise, initialized with epsilon
    G_diag_sum_sq = np.full(d, epsilon)
    
    loss_history = np.zeros(max_iterations + 1)
    trajectory = np.zeros((max_iterations + 1, d))
    loss_history[0] = loss_fn(theta)
    trajectory[0] = theta.copy()
    
    t = 0 # Iteration counter for accurate list slicing
    try:
        for t in range(1, max_iterations + 1):
            grad = grad_fn(theta)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < tol:
                print(f"Diagonal AdaGrad converged at iteration {t} (grad_norm={grad_norm:.2e})")
                break
                
            G_diag_sum_sq += grad**2
            # Compute G_t^-1/2 * grad_t (element-wise)
            adjusted_grad = grad / np.sqrt(G_diag_sum_sq) # np.sqrt handles epsilon implicitly now
            
            theta -= learning_rate * adjusted_grad
            
            loss_history[t] = loss_fn(theta)
            trajectory[t] = theta.copy()
    except Exception as e:
        print(f"Error in diagonal_adagrad at iteration {t}: {str(e)}")
    
    # Ensure loop variable `t` correctly reflects the last computed iteration
    if t < max_iterations and grad_norm >= tol: # Didn't converge or break early
         t += 1 # Include the last iteration if loop finished normally
         
    runtime = time.time() - start_time
    return theta, loss_history[:t].tolist(), trajectory[:t].tolist(), runtime

def full_matrix_adagrad(
    loss_fn: Callable,
    grad_fn: Callable,
    initial_params: np.ndarray,
    learning_rate: float = 0.1,
    delta: float = 1e-8,      # Delta for G_t initialization (matches description)
    epsilon_eig: float = 1e-8, # Epsilon for eigenvalue stability during inversion
    max_iterations: int = 1000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, List[float], List[np.ndarray], float]:
    """Full matrix AdaGrad implementation using G_t^-1/2.
    
    Updates parameters using: theta_t+1 = theta_t - eta * G_t^-1/2 * grad_t
    where G_t = delta*I + sum(g_k * g_k^T). G_t^-1/2 is computed via eigendecomposition.
    Note: This is computationally expensive (O(d^3) per iteration).

    Args:
        loss_fn: Function that computes the loss.
        grad_fn: Function that computes the gradient.
        initial_params: Initial parameter values.
        learning_rate: Learning rate (eta).
        delta: Small constant for G_t initialization.
        epsilon_eig: Small constant for eigenvalue stability in G_t^-1/2.
        max_iterations: Maximum number of iterations.
        tol: Convergence tolerance for gradient norm.
        
    Returns:
        Tuple: (final parameters, loss history, parameter trajectory, runtime).
    """
    start_time = time.time()
    d = len(initial_params)
    theta = initial_params.copy()
    # G accumulates sum of outer products, initialized with delta * I
    G = np.eye(d) * delta 
    
    loss_history = np.zeros(max_iterations + 1)
    trajectory = np.zeros((max_iterations + 1, d))
    loss_history[0] = loss_fn(theta)
    trajectory[0] = theta.copy()
    
    t = 0
    try:
        for t in range(1, max_iterations + 1):
            grad = grad_fn(theta)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < tol:
                print(f"Full Matrix AdaGrad converged at iteration {t} (grad_norm={grad_norm:.2e})")
                break
                
            # Accumulate outer product
            G += np.outer(grad, grad)
            
            # Ensure G remains symmetric (important for eigendecomposition)
            G = (G + G.T) / 2.0
            
            try:
                # Compute G_t^-1/2 via eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eigh(G) 
                
                # Add epsilon for stability before taking inverse sqrt
                # Use maximum to avoid issues with negative eigenvalues if G isn't perfectly PSD
                inv_sqrt_eigenvalues = 1.0 / np.sqrt(np.maximum(eigenvalues, 0) + epsilon_eig)
                
                # Construct G_t^-1/2 = Q * Lambda^-1/2 * Q^T
                G_inv_sqrt = eigenvectors @ np.diag(inv_sqrt_eigenvalues) @ eigenvectors.T
                
                # Calculate update direction
                update = G_inv_sqrt @ grad
                
                # Clip extreme updates for stability
                max_update_norm = 1e6 # Limit magnitude of update step
                update_norm = np.linalg.norm(update)
                if update_norm > max_update_norm:
                    update *= max_update_norm / update_norm
                
                theta -= learning_rate * update
                loss_history[t] = loss_fn(theta)
                trajectory[t] = theta.copy()
                
            except np.linalg.LinAlgError as e:
                print(f"Eigendecomposition failed at iteration {t}: {e}, stopping early")
                break
                
    except Exception as e:
        print(f"Error in full_matrix_adagrad at iteration {t}: {str(e)}")

    if t < max_iterations and grad_norm >= tol: t += 1
    runtime = time.time() - start_time
    return theta, loss_history[:t].tolist(), trajectory[:t].tolist(), runtime

def incremental_approx_full_matrix_adagrad(
    loss_fn: Callable,
    grad_fn: Callable,
    initial_params: np.ndarray,
    learning_rate: float = 0.1,
    delta: float = 1e-8, # Delta for G_t initialization and stability
    max_iterations: int = 1000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, List[float], List[np.ndarray], float]:
    """Incrementally Approximated Full Matrix AdaGrad.
    
    Uses Sherman-Morrison for G_t^-1 and a randomized approx A_t for G_t^1/2.
    Approximates update direction u_t = G_t^-1 * (A_t * grad_t) ~= G_t^-1/2 * grad_t.

    Args:
        loss_fn: Function that computes the loss.
        grad_fn: Function that computes the gradient.
        initial_params: Initial parameter values.
        learning_rate: Learning rate (eta).
        delta: Small constant for G_t initialization and stability.
        max_iterations: Maximum number of iterations.
        tol: Convergence tolerance for gradient norm.
        
    Returns:
        Tuple: (final parameters, loss history, parameter trajectory, runtime).
    """
    start_time = time.time()
    d = len(initial_params)
    theta = initial_params.copy()
    
    # Initialize matrices as per description
    G_inv = np.eye(d) / delta           # G_0^-1
    A = np.eye(d) * np.sqrt(delta)    # A_0
    
    loss_history = np.zeros(max_iterations + 1)
    trajectory = np.zeros((max_iterations + 1, d))
    loss_history[0] = loss_fn(theta)
    trajectory[0] = theta.copy()
    
    t = 0
    try:
        for t in range(1, max_iterations + 1):
            grad = grad_fn(theta)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < tol:
                print(f"Incremental Approx AdaGrad converged at iteration {t} (grad_norm={grad_norm:.2e})")
                break
                
            # Cap gradient norm for stability before using in updates
            max_grad_norm = 1e6 
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / grad_norm)
                
            # --- Update G_inv using Sherman-Morrison --- 
            G_inv_g = G_inv @ grad
            denominator = 1.0 + grad.dot(G_inv_g)
            
            # Handle numerical issues in denominator
            if denominator < delta: # Use delta for stability floor
                denominator = delta 
                
            G_inv = G_inv - np.outer(G_inv_g, G_inv_g) / denominator
            
            # Ensure G_inv remains symmetric
            G_inv = (G_inv + G_inv.T) / 2.0
            
            # --- Update A using random projection --- 
            z = np.random.randn(d) # Sample from N(0, I)
            z_norm = np.linalg.norm(z)
            if z_norm > 1e-9: # Avoid division by zero if z is effectively zero
                 z = z / z_norm # Normalize to unit sphere
            else:
                 z = np.zeros(d)
                 z[np.random.randint(d)] = 1.0 # Fallback: standard basis vector
                 
            A = A + np.outer(grad, z) # A_t = A_{t-1} + g_t * z_t^T
            
            # --- Compute Approximate Update Direction --- 
            v = A @ grad  # v_t = A_t * g_t
            u = G_inv @ v # u_t = G_t^-1 * v_t  (approximates G_t^-1/2 * g_t)
            
            # Clip extreme updates for stability
            max_update_norm = 1e6 
            update_norm = np.linalg.norm(u)
            if update_norm > max_update_norm:
                u *= max_update_norm / update_norm
                
            # --- Update parameters --- 
            theta -= learning_rate * u
            
            loss_history[t] = loss_fn(theta)
            trajectory[t] = theta.copy()
            
    except Exception as e:
        print(f"Error in incremental_approx_full_matrix_adagrad at iteration {t}: {str(e)}")

    if t < max_iterations and grad_norm >= tol: t += 1
    runtime = time.time() - start_time
    return theta, loss_history[:t].tolist(), trajectory[:t].tolist(), runtime 