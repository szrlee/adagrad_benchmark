"""Plotting utilities for AdaGrad experiments."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import glob
import warnings

# Define common styles
OPTIMIZER_STYLES = {
    'diagonal': {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'label': 'Diagonal'},
    'full': {'color': 'red', 'marker': 's', 'linestyle': '--', 'label': 'Full Matrix'},
    'incremental': {'color': 'green', 'marker': '^', 'linestyle': ':', 'label': 'Incremental'}
}


def load_results(results_dir: str) -> Dict[Tuple[int, float], Dict[str, Any]]:
    """Load all results from the results directory.
    
    Organizes results by (dimension, condition_number) keys.
    
    Args:
        results_dir: Directory containing experiment results
        
    Returns:
        Dictionary containing results organized by (dimension, condition_number)
    """
    results = {}
    for file_path in glob.glob(f"{results_dir}/results_dim_*_cond_*.json"):
        file = Path(file_path)
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            dim = data['dimension']
            cond = data['condition_number']
            key = (dim, cond)
            results[key] = data
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON file {file.name}")
            continue
        except KeyError as e:
            print(f"Warning: Missing key {e} in {file.name}")
            continue
    
    if not results:
        print(f"Warning: No valid result files found in {results_dir}")
        
    return results

# --- Internal plotting functions for single figures --- 

def _plot_single_convergence(
    dim: int,
    cond: float,
    problem_type: str,
    problem_data: Dict[str, Any],
    save_dir_path: Path
) -> None:
    """Plots convergence for a single (dim, cond, problem_type) setting."""
    fig, ax = plt.subplots(figsize=(8, 6))
    line_added = False

    for name, optimizer_data in problem_data.items():
        if name == 'optimal_point': continue
        
        if 'loss_history' not in optimizer_data or 'iterations' not in optimizer_data:
            continue
            
        loss_history = np.array(optimizer_data['loss_history'])
        iterations = optimizer_data['iterations']
        final_loss = optimizer_data.get('final_loss', loss_history[-1] if len(loss_history) > 0 else np.nan)
        
        valid_mask = np.isfinite(loss_history)
        if not np.any(valid_mask): continue
        
        # Clip extreme values
        valid_loss = loss_history[valid_mask]
        if len(valid_loss) > 0:
            if np.max(valid_loss) > 1e10: loss_history = np.clip(loss_history, None, 1e10)
            if np.min(valid_loss) < 1e-10: loss_history = np.clip(loss_history, 1e-10, None)
        
        # Ensure style exists, fallback if needed
        style = OPTIMIZER_STYLES.get(name, {'color': 'gray', 'marker': '.', 'linestyle': '-.', 'label': name})
        ax.semilogy(range(iterations), loss_history, 
                     color=style['color'], 
                     marker=style['marker'], 
                     linestyle=style['linestyle'],
                     markersize=4, 
                     markevery=max(1, iterations//10),  
                     label=style['label'])
        line_added = True
        
        if np.isfinite(final_loss):
            # Adjust annotation placement based on loss value
            y_pos = final_loss
            va = 'bottom' if final_loss < 1e-5 else 'top' 
            offset_y = 5 if final_loss < 1e-5 else -5
            ax.annotate(f'{final_loss:.2e}', 
                      xy=(iterations-1, y_pos),
                      xytext=(5, offset_y), textcoords='offset points',
                      color=style['color'], fontsize=8, va=va)

    if line_added:
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss (log scale)', fontsize=12)
        ax.set_title(f'{problem_type.capitalize()} Convergence (Dim={dim}, Cond={int(cond)})', fontsize=14)
        ax.legend(fontsize='medium')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        
        # Adjust ylim based on plotted data
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(bottom=max(1e-12, y_min * 0.1), top=min(1e12, y_max * 10))
        
        filename = f'convergence_{problem_type}_dim{dim}_cond{int(cond)}.png'
        save_file = save_dir_path / filename
        plt.tight_layout()
        plt.savefig(save_file, dpi=300)
        print(f"Saved: {filename}")
    else:
        print(f"Skipped convergence plot (no data): {problem_type}_dim{dim}_cond{int(cond)}")
    plt.close(fig)

def _plot_single_trajectory(
    dim: int,
    cond: float,
    problem_type: str,
    problem_data: Dict[str, Any],
    save_dir_path: Path
) -> None:
    """Plots trajectory for a single (dim, cond, problem_type) setting."""
    if dim > 10: # Only plot trajectories for low dimensions
        return
        
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_added = False
    trajectories_present = False
    initial_point_plotted = False
    all_points = [] # For axis limits

    optimal_point = np.array(problem_data.get('optimal_point', [np.nan, np.nan]))[:2]
    if np.all(np.isfinite(optimal_point)):
        all_points.append(optimal_point)
        ax.plot(optimal_point[0], optimal_point[1], 'kx', markersize=12, 
                markeredgewidth=2.5, label='Optimal Point')
    else:
        optimal_point = None # Mark as not available

    # Collect points for limits and check presence
    initial_points = []
    for name, optimizer_data in problem_data.items():
        if name == 'optimal_point': continue
        if 'trajectory' in optimizer_data:
            traj = np.array(optimizer_data['trajectory'])[:, :2]
            valid_traj = traj[np.all(np.isfinite(traj), axis=1)]
            if valid_traj.shape[0] > 0:
                all_points.extend(valid_traj)
                trajectories_present = True
                if 'initial_params' in optimizer_data:
                     initial_points.append(np.array(optimizer_data['initial_params'])[:2])
                else:
                     initial_points.append(valid_traj[0]) # Fallback
            
    # Add initial points if valid
    valid_initial_points = [p for p in initial_points if np.all(np.isfinite(p))]
    if valid_initial_points:
        all_points.extend(valid_initial_points)
    
    # Set axis limits if points exist
    if all_points:
        all_points_arr = np.array(all_points)
        if all_points_arr.size > 0 and np.all(np.isfinite(all_points_arr)):
            x_min, x_max = np.min(all_points_arr[:, 0]), np.max(all_points_arr[:, 0])
            y_min, y_max = np.min(all_points_arr[:, 1]), np.max(all_points_arr[:, 1])
            x_range = max(x_max - x_min, 0.1)
            y_range = max(y_max - y_min, 0.1)
            margin = 0.1 
            ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
            ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        else:
            trajectories_present = False # Cannot set limits if points are invalid
    else:
        trajectories_present = False # No points to plot

    # Draw contours if possible
    if dim >= 2 and trajectories_present and optimal_point is not None:
        try:
            x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
            x_grid = np.linspace(x_lim[0], x_lim[1], 100)
            y_grid = np.linspace(y_lim[0], y_lim[1], 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            # Use a helper for contour calculation if needed, or inline
            if problem_type == 'quadratic':
                Z = (X - optimal_point[0])**2 + cond * (Y - optimal_point[1])**2
            else:
                Z = (X - optimal_point[0])**2 + cond * (Y - optimal_point[1])**2 + \
                    0.1 * cond * np.sin(3 * (X - optimal_point[0])) * np.cos(3 * (Y - optimal_point[1]))

            with warnings.catch_warnings(): # Suppress potential contour warnings
                warnings.simplefilter("ignore")
                Z_finite = Z[np.isfinite(Z)]
                if Z_finite.size > 0:
                    min_Z, max_Z = np.min(Z_finite), np.max(Z_finite)
                    if max_Z > min_Z and max_Z > 0:
                        levels = np.logspace(np.log10(max(1e-6, min_Z)), np.log10(max_Z), 10)
                        cont = ax.contour(X, Y, Z, levels=levels, colors='grey', alpha=0.4, linewidths=0.5)
                        # Optional: Add contour labels
                        # ax.clabel(cont, inline=True, fontsize=8, fmt='%.1e') 
        except Exception as e:
            print(f"Could not draw contours for dim={dim}, cond={cond}: {e}")

    # Plot trajectories
    for name, optimizer_data in problem_data.items():
        if name == 'optimal_point' or 'trajectory' not in optimizer_data: continue
            
        trajectory = np.array(optimizer_data['trajectory'])
        if trajectory.shape[1] > 2: trajectory = trajectory[:, :2]
        valid_traj = trajectory[np.all(np.isfinite(trajectory), axis=1)]
        if len(valid_traj) < 2: continue
            
        style = OPTIMIZER_STYLES.get(name, {'color': 'gray', 'marker': '.', 'linestyle': '-.', 'label': name})
        plot_added = True
        
        ax.plot(valid_traj[:, 0], valid_traj[:, 1], 
                color=style['color'], alpha=0.7, linewidth=1.2, 
                marker=style['marker'], markersize=3, linestyle=style['linestyle'], 
                label=style['label'])
        
        # Plot Initial Point (only once per plot)
        if not initial_point_plotted:
            ip = None
            if 'initial_params' in optimizer_data:
                ip_raw = np.array(optimizer_data['initial_params'])[:2]
                if np.all(np.isfinite(ip_raw)):
                    ip = ip_raw
            elif len(valid_traj) > 0 and np.all(np.isfinite(valid_traj[0])):
                ip = valid_traj[0]
            
            if ip is not None:
                ax.plot(ip[0], ip[1], 'o', markersize=7, markerfacecolor='yellow', markeredgecolor='black', label='Initial Pt')
                initial_point_plotted = True
        
        # Plot Final Point marker
        if np.all(np.isfinite(valid_traj[-1])):
            ax.plot(valid_traj[-1, 0], valid_traj[-1, 1], 'D', markersize=5, color=style['color'], markeredgecolor='black')

    if plot_added:
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.set_title(f'{problem_type.capitalize()} Trajectory (Dim={dim}, Cond={int(cond)})', fontsize=14)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize='medium', loc='best')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_aspect('equal', adjustable='box') # Ensure aspect ratio is equal
        
        filename = f'trajectory_{problem_type}_dim{dim}_cond{int(cond)}.png'
        save_file = save_dir_path / filename
        plt.tight_layout()
        plt.savefig(save_file, dpi=300)
        print(f"Saved: {filename}")
    else:
        print(f"Skipped trajectory plot (no data): {problem_type}_dim{dim}_cond{int(cond)}")
    plt.close(fig)

# --- Main plotting functions --- 

def plot_convergence(
    results: Dict[Tuple[int, float], Dict[str, Any]],
    save_dir: str
) -> None:
    """Generate individual convergence plots for each experiment setting."""
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    if not results:
        print("No results found to plot convergence.")
        return
        
    print("\nGenerating Convergence Plots...")
    for (dim, cond), data in results.items():
        for ptype in ['quadratic', 'ill_conditioned']:
            if ptype in data:
                _plot_single_convergence(dim, cond, ptype, data[ptype], save_dir_path)

def plot_runtime(
    results: Dict[Tuple[int, float], Dict[str, Any]],
    save_dir: str
) -> None:
    """Plot runtime comparison across dimensions, separate figure per problem type."""
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    if not results:
        print("No results found to plot runtime.")
        return

    print("\nGenerating Runtime Plots...")
    all_dims = sorted(list(set(dim for dim, cond in results.keys())))
    all_conds = sorted(list(set(cond for dim, cond in results.keys())))
    if not all_dims or not all_conds:
        print("Could not determine dimensions or conditions from results.")
        return
        
    n_dims = len(all_dims)
    n_conds = len(all_conds)
    n_optimizers = len(OPTIMIZER_STYLES)
    
    x_indices = np.arange(n_dims) # positions for dimensions
    total_width = 0.8
    single_bar_width = total_width / (n_conds * n_optimizers)
    
    for ptype in ['quadratic', 'ill_conditioned']:
        fig, ax = plt.subplots(figsize=(max(8, n_dims * 1.0), 6)) # Wider figure based on dims
        bar_added = False
        labels_added = set()
        all_plotted_runtimes = [] # Collect all valid runtimes plotted
        
        for k, cond in enumerate(all_conds):
            for j, (name, style) in enumerate(OPTIMIZER_STYLES.items()):
                runtimes = []
                
                for dim_idx, dim in enumerate(all_dims):
                    key = (dim, cond)
                    runtime = np.nan # Default to NaN
                    if key in results and ptype in results[key] and name in results[key][ptype]:
                        rt_data = results[key][ptype][name].get('runtime', np.nan)
                        if np.isfinite(rt_data):
                            runtime = rt_data
                            
                    # Store runtime (or NaN) corresponding to each dimension
                    runtimes.append(runtime)
                        
                # Get valid runtimes and their original indices for plotting
                valid_runtimes = [rt for rt in runtimes if np.isfinite(rt) and rt > 0] # Only positive for log
                corresponding_x_indices = [i for i, rt in enumerate(runtimes) if np.isfinite(rt) and rt > 0]

                if valid_runtimes:
                    offset = (k * n_optimizers + j) * single_bar_width - total_width / 2 + single_bar_width / 2
                    actual_positions = x_indices[corresponding_x_indices] + offset
                    
                    label_key = (name, cond)
                    current_label = None
                    if label_key not in labels_added:
                        current_label = f'{style["label"]} (C={int(cond)})' # Shorter label
                        labels_added.add(label_key)
                    
                    bars = ax.bar(actual_positions, valid_runtimes, single_bar_width, 
                                color=style['color'], alpha=0.8,
                                label=current_label)
                    bar_added = True
                    all_plotted_runtimes.extend(valid_runtimes) # Add to list for limit calculation
                    ax.bar_label(bars, fmt='%.2fs', padding=3, fontsize=7, rotation=90)

        if bar_added:
            ax.set_xlabel('Dimension', fontsize=12)
            ax.set_ylabel('Runtime (seconds, log scale)', fontsize=12)
            ax.set_title(f'{ptype.capitalize()} Runtime Comparison', fontsize=14)
            ax.set_yscale('log')
            ax.set_xticks(x_indices)
            ax.set_xticklabels(all_dims)
            ax.legend(title="Optimizer (Cond #)", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize='small')
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)
            
            # Adjust y-limits based on data
            if all_plotted_runtimes: # Check if any valid runtimes were plotted
                min_rt_data = min(all_plotted_runtimes)
                current_bottom, current_top = ax.get_ylim()
                # Set bottom slightly below min, ensuring it's positive and less than top
                new_bottom = max(min_rt_data * 0.5, 1e-6) # Use a small positive floor
                if new_bottom < current_top:
                     ax.set_ylim(bottom=new_bottom)
            
            filename = f'runtime_{ptype}.png'
            save_file = save_dir_path / filename
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect slightly for potentially wider legend
            plt.savefig(save_file, dpi=300)
            print(f"Saved: {filename}")
        else:
            print(f"Skipped runtime plot (no data): {ptype}")
        plt.close(fig)

def plot_trajectories(
    results: Dict[Tuple[int, float], Dict[str, Any]],
    save_dir: str
) -> None:
    """Generate individual trajectory plots for each low-dimensional setting."""
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    if not results:
        print("No results found to plot trajectories.")
        return

    print("\nGenerating Trajectory Plots (for dim <= 10)...")
    for (dim, cond), data in results.items():
        if dim > 10:
            continue
        for ptype in ['quadratic', 'ill_conditioned']:
            if ptype in data:
                _plot_single_trajectory(dim, cond, ptype, data[ptype], save_dir_path)

def plot_all_results(
    results_dir: str = './results',
    save_dir: str = './plots'
) -> None:
    """Load results and generate all standard plots.
    
    Args:
        results_dir: Directory containing experiment results.
        save_dir: Directory to save plots.
    """
    # Load results
    results = load_results(results_dir)
    
    if not results:
        print(f"No results found in {results_dir}. Cannot generate plots.")
        return
    
    # Generate plots
    plot_convergence(results, save_dir)
    plot_runtime(results, save_dir)
    plot_trajectories(results, save_dir) # Renamed from plot_trajectory
    
    print(f"\nAll plots generated in {Path(save_dir).resolve()}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot AdaGrad experiment results')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory containing experiment results')
    parser.add_argument('--save-dir', type=str, default='./plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    plot_all_results(args.results_dir, args.save_dir) 