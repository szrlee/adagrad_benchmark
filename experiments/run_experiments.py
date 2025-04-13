"""Run AdaGrad comparison experiments."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Callable
import numpy as np
import sys
import traceback

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adagrad_benchmark.optimizers.adagrad import (
    diagonal_adagrad,
    full_matrix_adagrad,
    incremental_approx_full_matrix_adagrad
)
from adagrad_benchmark.objectives.test_problems import (
    create_quadratic_problem,
    create_ill_conditioned_problem
)

def run_comparison(
    dimension: int,
    condition_number: float,
    learning_rate: float = 0.1,
    max_iterations: int = 1000,
    save_dir: str = './results'
) -> Dict[str, Any]:
    """Run comparison experiments.
    
    Args:
        dimension: Problem dimension
        condition_number: Condition number for test problems
        learning_rate: Learning rate for optimization
        max_iterations: Maximum number of iterations
        save_dir: Directory to save results
        
    Returns:
        Dictionary containing experiment results
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create test problems
    quadratic_loss, quadratic_grad, quadratic_opt = create_quadratic_problem(
        dimension=dimension,
        condition_number=condition_number
    )
    
    ill_loss, ill_grad, ill_opt = create_ill_conditioned_problem(
        dimension=dimension,
        condition_number=condition_number
    )
    
    # Initialize results dictionary
    results = {
        'dimension': dimension,
        'condition_number': condition_number,
        'quadratic': {
            'optimal_point': quadratic_opt.tolist()  # Save the optimal point
        },
        'ill_conditioned': {
            'optimal_point': ill_opt.tolist()  # Save the optimal point
        }
    }
    
    # Use same initial parameters for all optimizers on each problem
    # for fair comparison
    np.random.seed(42)  # For reproducibility
    quadratic_initial = np.random.randn(dimension)
    ill_initial = np.random.randn(dimension)
    
    # Run optimizers on quadratic problem
    print(f"\nRunning optimizers on quadratic problem (dim={dimension}, cond={condition_number})...")
    _run_optimizers_on_problem(
        problem_name="quadratic",
        loss_fn=quadratic_loss,
        grad_fn=quadratic_grad,
        initial_params=quadratic_initial,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        results=results
    )
    
    # Run optimizers on ill-conditioned problem
    print(f"\nRunning optimizers on ill-conditioned problem (dim={dimension}, cond={condition_number})...")
    _run_optimizers_on_problem(
        problem_name="ill_conditioned",
        loss_fn=ill_loss,
        grad_fn=ill_grad,
        initial_params=ill_initial,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        results=results
    )
    
    # Save results
    results_file = save_path / f'results_dim_{dimension}_cond_{condition_number}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    return results

def _run_optimizers_on_problem(
    problem_name: str,
    loss_fn: Callable,
    grad_fn: Callable,
    initial_params: np.ndarray,
    learning_rate: float,
    max_iterations: int,
    results: Dict[str, Any]
) -> None:
    """Run all optimizers on a specific problem.
    
    Args:
        problem_name: Name of the problem ("quadratic" or "ill_conditioned")
        loss_fn: Loss function
        grad_fn: Gradient function
        initial_params: Initial parameters
        learning_rate: Learning rate for optimization
        max_iterations: Maximum number of iterations
        results: Results dictionary to update
    """
    optimizers = [
        ('diagonal', diagonal_adagrad),
        ('full', full_matrix_adagrad),
        ('incremental', incremental_approx_full_matrix_adagrad)
    ]
    
    for name, optimizer in optimizers:
        print(f"  Running {name} AdaGrad...")
        try:
            theta, loss_history, trajectory, runtime = optimizer(
                loss_fn=loss_fn,
                grad_fn=grad_fn,
                initial_params=initial_params.copy(),  # Use copy to avoid modification
                learning_rate=learning_rate,
                max_iterations=max_iterations
            )
            
            results[problem_name][name] = {
                'final_loss': float(loss_history[-1]),  # Convert to float for JSON serialization
                'iterations': len(loss_history),
                'runtime': float(runtime),  # Convert to float for JSON serialization
                'trajectory': trajectory,
                'loss_history': loss_history,
                'initial_params': initial_params.tolist()  # Save initial params
            }
            print(f"    Final loss: {loss_history[-1]:.6e}, Runtime: {runtime:.3f}s")
        except Exception as e:
            print(f"    Error running {name} AdaGrad: {str(e)}")
            traceback.print_exc()
            results[problem_name][name] = {
                'error': str(e),
                'initial_params': initial_params.tolist()
            }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run AdaGrad comparison experiments')
    parser.add_argument('--dimensions', type=int, nargs='+', default=[10, 100, 1000],
                       help='Problem dimensions to test')
    parser.add_argument('--condition-numbers', type=float, nargs='+', default=[10.0, 100.0, 1000.0],
                       help='Condition numbers to test')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate for optimization')
    parser.add_argument('--max-iterations', type=int, default=1000,
                       help='Maximum number of iterations')
    parser.add_argument('--save-dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Run experiments for each dimension and condition number
    for dim in args.dimensions:
        for cond in args.condition_numbers:
            try:
                print(f"\nRunning experiments for dimension {dim}, condition number {cond}")
                run_comparison(
                    dimension=dim,
                    condition_number=cond,
                    learning_rate=args.learning_rate,
                    max_iterations=args.max_iterations,
                    save_dir=args.save_dir
                )
            except Exception as e:
                print(f"Error running experiments for dim={dim}, cond={cond}: {str(e)}")
                traceback.print_exc()

if __name__ == '__main__':
    main() 