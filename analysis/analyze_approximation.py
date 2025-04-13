"""Analyze the approximation quality of Incremental Approx Full Matrix AdaGrad.

Compares the approximate update direction u_approx = G_inv @ (A @ grad)
with the true update direction u_true = G^-1/2 @ grad at each iteration.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adagrad_benchmark.objectives.test_problems import create_quadratic_problem

def analyze_approximation(
    dimension: int,
    condition_number: float,
    learning_rate: float = 0.1,
    delta: float = 1e-8,
    epsilon_eig: float = 1e-8,
    max_iterations: int = 100,
    save_dir: str = './plots/analysis'
):
    """Runs the analysis comparing true and approximate update directions."""
    print(f"Analyzing approximation for dim={dimension}, cond={condition_number}")
    
    # --- Setup --- 
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42) # Reproducibility
    loss_fn, grad_fn, opt_point = create_quadratic_problem(dimension, condition_number)
    initial_params = np.random.randn(dimension)
    theta = initial_params.copy() # Use separate theta for simulation if needed
    d = dimension
    
    # Initialize matrices for approximation algorithm
    G_inv = np.eye(d) / delta           # G_0^-1
    A = np.eye(d) * np.sqrt(delta)    # A_0
    
    # Initialize matrix for true G_t accumulation
    G_true = np.eye(d) * delta
    
    # Storage for metrics
    cosine_similarities = []
    magnitude_ratios = []
    relative_errors = []    
    iterations = range(1, max_iterations + 1)
    
    # --- Simulation Loop --- 
    print("Starting simulation loop...")
    for t in iterations:
        try:
            grad = grad_fn(theta) # Use current theta for gradient
            grad_norm = np.linalg.norm(grad)
            
            # --- True G^-1/2 Calculation (Expensive) --- 
            G_true += np.outer(grad, grad)
            G_true = (G_true + G_true.T) / 2.0 # Ensure symmetry
            
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(G_true)
                inv_sqrt_eigenvalues = 1.0 / np.sqrt(np.maximum(eigenvalues, 0) + epsilon_eig)
                G_true_inv_sqrt = eigenvectors @ np.diag(inv_sqrt_eigenvalues) @ eigenvectors.T
                u_true = G_true_inv_sqrt @ grad
                u_true_norm = np.linalg.norm(u_true)
                if u_true_norm < 1e-12: # Avoid division by zero later
                     print(f"Warning: True update norm near zero at iter {t}. Skipping metrics.")
                     continue # Skip metrics for this iteration
            except np.linalg.LinAlgError:
                print(f"True G^-1/2 calculation failed at iter {t}. Stopping analysis.")
                break

            # --- Incremental Approximation Calculation --- 
            # Cap gradient before using in incremental updates
            max_grad_norm = 1e6 
            if grad_norm > max_grad_norm: grad = grad * (max_grad_norm / grad_norm)
                
            # Update G_inv (Sherman-Morrison)
            G_inv_g = G_inv @ grad
            denominator = 1.0 + grad.dot(G_inv_g)
            if denominator < delta: denominator = delta
            G_inv = G_inv - np.outer(G_inv_g, G_inv_g) / denominator
            G_inv = (G_inv + G_inv.T) / 2.0
            
            # Update A (Random Projection)
            z = np.random.randn(d)
            z_norm = np.linalg.norm(z)
            if z_norm > 1e-9: z = z / z_norm
            else: 
                z = np.zeros(d); z[0] = 1.0 # Simple fallback
            A = A + np.outer(grad, z)
            
            # Calculate Approx Update Direction
            v = A @ grad
            u_approx = G_inv @ v
            u_approx_norm = np.linalg.norm(u_approx)
            
            # --- Calculate Metrics --- 
            # Cosine Similarity
            dot_product = np.dot(u_approx, u_true)
            cos_sim = dot_product / (u_approx_norm * u_true_norm) if u_approx_norm > 1e-12 else 0.0
            cosine_similarities.append(cos_sim)
            
            # Magnitude Ratio
            mag_ratio = u_approx_norm / u_true_norm if u_true_norm > 1e-12 else np.nan
            magnitude_ratios.append(mag_ratio)
            
            # Relative Error
            rel_error = np.linalg.norm(u_approx - u_true) / u_true_norm if u_true_norm > 1e-12 else np.nan
            relative_errors.append(rel_error)
            
            # --- Update theta (optional - can simulate actual steps) --- 
            # For this analysis, we might not need to update theta, 
            # or update it using u_true to see how approx performs on the theoretical path
            # theta -= learning_rate * u_approx # Or use u_true?
            # Let's keep theta fixed for now to analyze approx based on same point
            
            if t % 10 == 0:
                print(f"Iter {t}: CosSim={cos_sim:.4f}, MagRatio={mag_ratio:.4f}, RelErr={rel_error:.4f}")
                
        except Exception as e:
            print(f"Error during simulation at iteration {t}: {str(e)}")
            traceback.print_exc()
            break
            
    # --- Plotting --- 
    print("Plotting results...")
    actual_iterations = iterations[:len(cosine_similarities)] # Adjust range if loop broke early
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f'Approximation Quality Analysis (Dim={dimension}, Cond={int(condition_number)})', fontsize=16)
    
    # Cosine Similarity Plot
    axes[0].plot(actual_iterations, cosine_similarities, label='Cosine Similarity', color='blue')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_ylim([-1.1, 1.1])
    axes[0].axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    axes[0].axhline(0.0, color='gray', linestyle=':', alpha=0.5)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    # Magnitude Ratio Plot
    axes[1].plot(actual_iterations, magnitude_ratios, label='Magnitude Ratio (||u_approx|| / ||u_true||)', color='red')
    axes[1].set_ylabel('Magnitude Ratio')
    axes[1].set_yscale('log') # Often useful for ratios
    axes[1].axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    axes[1].grid(True, which='both', linestyle='--', alpha=0.6)
    axes[1].legend()

    # Relative Error Plot
    axes[2].plot(actual_iterations, relative_errors, label='Relative Error (||u_approx - u_true|| / ||u_true||)', color='green')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Relative Error')
    axes[2].set_yscale('log')
    axes[2].grid(True, which='both', linestyle='--', alpha=0.6)
    axes[2].legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout for suptitle
    filename = f'approx_analysis_dim{dimension}_cond{int(condition_number)}.png'
    save_file = save_dir_path / filename
    plt.savefig(save_file, dpi=300)
    print(f"Analysis plot saved to {save_file}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Incremental Approx AdaGrad Approximation Quality')
    parser.add_argument('--dimension', type=int, default=10,
                       help='Problem dimension for analysis (low recommended)')
    parser.add_argument('--condition-number', type=float, default=100.0,
                       help='Condition number for test problem')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate (for potential theta updates, if added)')
    parser.add_argument('--delta', type=float, default=1e-8,
                       help='Initialization/stability constant delta')
    parser.add_argument('--epsilon-eig', type=float, default=1e-8,
                       help='Stability constant for true G^-1/2 calculation')
    parser.add_argument('--max-iterations', type=int, default=100,
                       help='Number of iterations to simulate')
    parser.add_argument('--save-dir', type=str, default='./plots/analysis',
                       help='Directory to save analysis plots')
    
    args = parser.parse_args()
    
    analyze_approximation(
        dimension=args.dimension,
        condition_number=args.condition_number,
        learning_rate=args.learning_rate,
        delta=args.delta,
        epsilon_eig=args.epsilon_eig,
        max_iterations=args.max_iterations,
        save_dir=args.save_dir
    ) 