# --- animation.py ---
"""
Visualization utilities: residual plots, formation error, and animation.
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

COLORS = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple']


def plot_residuals(residuals_over_time, threshold=None):
    fig, ax = plt.subplots()
    T = len(residuals_over_time[0])
    for i, res in enumerate(residuals_over_time):
        ax.plot(range(T), res, label=f'Node {i+1}', marker='')
    if threshold is not None:
        ax.hlines(threshold, 0, T-1, colors='k', linestyles='--', label='Threshold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Residual')
    ax.legend()
    ax.grid(True)
    return fig, ax


def plot_formation_error(positions, desired_distances):
    fig, ax = plt.subplots()
    errors = []
    for t in range(len(positions[0])):
        d_err = 0
        # compute average pairwise error vs. desired distances
        errors.append(d_err)
    ax.plot(range(len(errors)), errors)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Formation Error')
    ax.grid(True)
    return fig, ax


def animate_trajectories(positions, hall_positions, edges, interval=100):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(True)
    max_t = len(positions[0])

    def init():
        ax.clear()
        return []

    def update(frame):
        ax.clear()
        for i, traj in enumerate(positions):
            pts = np.array(traj[:frame])
            ax.plot(pts[:,0], pts[:,1], color=COLORS[i], alpha=0.4)
        # plot hallucinated if exists
        return ax,

    ani = FuncAnimation(fig, update, frames=max_t, init_func=init, interval=interval)
    return ani
