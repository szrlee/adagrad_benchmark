# AdaGrad Benchmark

A minimal Python package for benchmarking variants of the AdaGrad optimization algorithm 
(Diagonal, Full Matrix $G^{-1/2}$, and an Incremental Approximation) on synthetic test problems.

## Installation

Ensure you have Python 3.7+ installed.

```bash
# Clone the repository (if you haven't already)
# git clone <repository_url>
# cd adagrad-benchmark

# Install dependencies
pip install -r requirements.txt
```

## Features

- **AdaGrad Variants Implemented**:
  - Diagonal AdaGrad ($G_t = \text{diag}(\sum g_k^2)$)
  - Full Matrix AdaGrad ($G_t = \delta I + \sum g_k g_k^T$, using true $G_t^{-1/2}$ via Eigendecomposition)
  - Incremental Approximation Full Matrix AdaGrad (using Sherman-Morrison for $G_t^{-1}$ and random projection for $A_t \approx G_t^{1/2}$)

- **Test Problems**:
  - Quadratic functions with controlled condition numbers
  - Ill-conditioned test problems

- **Benchmarking & Analysis**:
  - Experiment runner script (`run_experiments.py`)
  - Result plotting script (`plotting.py`) generating separate plots for:
    - Convergence per configuration (Dim, Cond, Problem Type)
    - Runtime comparison per problem type
    - Trajectories per configuration (for low dimensions)
  - Approximation quality analysis script (`analyze_approximation.py`)

## Usage

All scripts are run as modules from the directory *containing* the `adagrad_benchmark` package folder.

### Running Experiments

This script runs the specified AdaGrad variants on the test problems for different dimensions and condition numbers, saving results as JSON files.

```bash
# Run with default parameters (dims=[10, 100, 1000], conds=[10, 100, 1000], iters=1000)
python -m experiments.run_experiments

# Run with custom parameters
python -m experiments.run_experiments \
  --dimensions 10 50 \
  --condition-numbers 10 1000 \
  --learning-rate 0.05 \
  --max-iterations 500 \
  --save-dir ./my_results 
```
Results are saved in the directory specified by `--save-dir` (default: `./results`).

### Visualizing Results

This script loads results saved by `run_experiments.py` and generates plots.

```bash
# Plot results from the default ./results directory, save plots to ./plots
python -m visualization.plotting

# Plot results from a custom directory, save plots to a custom directory
python -m visualization.plotting \
  --results-dir ./my_results \
  --save-dir ./my_plots
```
Plots are saved in the directory specified by `--save-dir` (default: `./plots`). 
Separate files are generated for each configuration's convergence and trajectory, 
and one summary plot per problem type is generated for runtime.

### Analyzing Approximation Quality (Incremental Approx Algorithm)

This script specifically analyzes how well the update direction of the `incremental_approx_full_matrix_adagrad` 
algorithm approximates the true `G^-1/2 @ grad` direction.

**Warning:** This analysis is computationally intensive as it requires calculating the true $G_t^{-1/2}$ via eigendecomposition ($O(d^3)$) at each step.

```bash
# Run analysis with default parameters (dim=10, cond=100, iters=100)
python -m analysis.analyze_approximation

# Run analysis with custom parameters
python -m analysis.analyze_approximation \
  --dimension 20 \
  --condition-number 500 \
  --max-iterations 150 \
  --delta 1e-7 \
  --save-dir ./plots/analysis
```
Metrics (Cosine Similarity, Magnitude Ratio, Relative Error) are plotted over iterations. 
The output plot is saved in the directory specified by `--save-dir` (default: `./plots/analysis`).

## Package Structure

```
adagrad_benchmark/
├── analysis/             # Approximation analysis scripts
│   ├── __init__.py
│   └── analyze_approximation.py
├── experiments/          # Experiment running scripts
│   ├── __init__.py
│   └── run_experiments.py
├── objectives/           # Optimization test problems
│   ├── __init__.py
│   └── test_problems.py
├── optimizers/           # AdaGrad implementations
│   ├── __init__.py
│   └── adagrad.py
├── visualization/        # Plotting utilities
│   ├── __init__.py
│   └── plotting.py
├── __init__.py           # Main package exports & version
└── requirements.txt      # Package dependencies
```

## Dependencies

- `numpy>=1.21.0,<2.0.0`
- `matplotlib>=3.4.0,<4.0.0`
- `pytest>=7.0.0` (For potential future tests)

## Contributing / Future Improvements

Contributions are welcome! Please focus on:
1.  Minimalism and clarity.
2.  Adding unit tests for new functionality.
3.  Maintaining consistency with the existing structure.

Potential future improvements:
*   Add more sophisticated test problems (e.g., non-convex).
*   Implement other adaptive gradient algorithms for comparison.
*   Add comprehensive unit and integration tests.
*   Refactor to use standard packaging practices (e.g., `setup.py` or `pyproject.toml`) instead of `sys.path` manipulation.
*   Develop a more unified command-line interface. 