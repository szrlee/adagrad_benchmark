# AdaGrad Benchmark

A minimal package for benchmarking AdaGrad optimization algorithm variants on different test problems.

## Installation

```bash
pip install -r requirements.txt
```

## Features

- **AdaGrad Variants**:
  - Diagonal AdaGrad
  - Full Matrix AdaGrad
  - Incremental Full Matrix AdaGrad

- **Test Problems**:
  - Quadratic functions with controlled condition numbers
  - Ill-conditioned test problems

- **Analysis**:
  - Convergence comparison
  - Runtime analysis
  - Optimization trajectory visualization

## Usage

### Running Experiments

```bash
# Run with default parameters
python -m experiments.run_experiments

# Run with custom parameters
python -m experiments.run_experiments \
  --dimensions 10 100 1000 \
  --condition-numbers 10 100 1000 \
  --learning-rate 0.1 \
  --max-iterations 1000 \
  --save-dir ./results
```

### Visualizing Results

```bash
# Plot all results 
python -m visualization.plotting \
  --results-dir ./results \
  --save-dir ./plots
  
# Plot specific problem type
python -m visualization.plotting \
  --results-dir ./results \
  --save-dir ./plots \
  --problem-type quadratic
```

## Package Structure

```
adagrad_benchmark/
├── experiments/          # Experiment setup
│   └── run_experiments.py
├── objectives/           # Test problems
│   └── test_problems.py
├── optimizers/           # Optimizer implementations
│   └── adagrad.py
├── visualization/        # Plotting utilities
│   └── plotting.py
├── __init__.py           # Package exports
└── requirements.txt      # Dependencies
```

## Contributing

1. Focus on minimalism and clarity
2. Add tests for new features
3. Follow the existing code structure for new contributions 