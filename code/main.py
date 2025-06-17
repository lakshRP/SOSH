# --- main.py ---
"""
Driver script: parses args, runs simulation, logs metrics, and generates plots/animation.
"""
import argparse
import os
import csv
import random
import numpy as np
from system import create_default_system, neighbor_mapping, desired_distances
from subsystem import Subsystem
from animation import plot_residuals, plot_formation_error, animate_trajectories


def run_experiment(steps, attack_cfg, out_dir):
    # Setup
    system = create_default_system()
    subs = [Subsystem(system, np.random.rand(2)) for _ in range(4)]
    if attack_cfg:
        idx, start, end, mag = attack_cfg
        subs[idx].add_attack(start, end, np.array(mag))

    positions, hall = [[] for _ in subs], [[] for _ in subs]
    residuals = [[] for _ in subs]

    for t in range(steps):
        # collect prior positions
        for i, s in enumerate(subs): positions[i].append(s.state.copy())
        # update each
        for s in subs: s.update_state(t)
        # detection & residual
        for i, s in enumerate(subs):
            res = s.compute_residual()
            residuals[i].append(res)
        # optional hallucination logic...

    # Save metrics CSV
    csv_path = os.path.join(out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node','time','residual'])
        for i, res in enumerate(residuals):
            for t,val in enumerate(res): writer.writerow([i+1,t,val])

    # Generate plots
    plot_residuals(residuals)
    plot_formation_error(positions, desired_distances)
    # animate_trajectories(positions, hall, list(neighbor_mapping.items()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SOSH Simulation')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--out', type=str, default='results')
    parser.add_argument('--attack', nargs=4, metavar=('IDX','START','END','MAG'),
                        help='attack cfg: node index, start, end, magnitude scalar')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    attack_cfg = None
    if args.attack:
        idx = int(args.attack[0])
        start = int(args.attack[1])
        end = int(args.attack[2])
        mag = float(args.attack[3])
        attack_cfg = (idx, start, end, [mag, mag])

    run_experiment(args.steps, attack_cfg, args.out)
