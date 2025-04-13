"""AdaGrad benchmark package for comparing optimization algorithms.

This package implements and benchmarks several variants of the AdaGrad optimizer
on different optimization problems.
"""

__version__ = '0.1.0'

from adagrad_benchmark.optimizers.adagrad import (
    diagonal_adagrad,
    full_matrix_adagrad,
    incremental_approx_full_matrix_adagrad
)

from adagrad_benchmark.objectives.test_problems import (
    create_quadratic_problem,
    create_ill_conditioned_problem
)

__all__ = [
    'diagonal_adagrad',
    'full_matrix_adagrad',
    'incremental_approx_full_matrix_adagrad',
    'create_quadratic_problem',
    'create_ill_conditioned_problem',
] 