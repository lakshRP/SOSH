import numpy as np
import pandas as pd
import yaml
import os

# Load experiment configurations
def load_config(path='experiments/experiment_config.yaml'):
    with open(path,'r') as f:
        return yaml.safe_load(f)

# Helper: compute desired displacements for N‑node formation
def make_displacement_map(positions):
    dmap = {}
    N = len(positions)
    for i in range(N):
        for j in range(i+1, N):
            dmap[(i,j)] = positions[i] - positions[j]
    return dmap

# Formation‑error metric
def formation_error(x, dmap):
    total = 0.0
    for (i,j),disp in dmap.items():
        err = x[i] - x[j] - disp
        total += err.dot(err)
    return 0.5*total

# Monte Carlo runner
if __name__ == '__main__':
    cfg = load_config()
    N = cfg['N']
    dt = cfg['dt']
    horizon = cfg['horizon']
    trials = cfg['trials']
    methods = cfg['methods']
    out_dir = cfg['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    # Precompute target formation (pentagon, square, etc.)
    angles = np.linspace(0,2*np.pi,N,endpoint=False)
    positions_ref = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    dmap = make_displacement_map(positions_ref)

    # Data accumulators
    all_positions = []
    agg_metrics = []

    # Trial loop
    for name, fn in methods.items():
        V_accum = np.zeros(horizon)
        for t in range(trials):
            x = np.random.uniform(-0.5,1.5,(N,2))
            prev = x.copy()
            for k in range(horizon):
                V_accum[k] += formation_error(x, dmap)
                y = x.copy()
                x_next = x.copy()
                fn(x, y, prev, x_next)  # method-specific step
                # Record positions
                for i in range(N):
                    all_positions.append({
                        'Method': name,
                        'Trial': t,
                        'TimeStep': k,
                        'Agent': i,
                        'X': x[i,0],
                        'Y': x[i,1]
                    })
                prev = x.copy()
                x = x_next
        V_avg = V_accum / trials
        # Compute metrics
        V100 = V_avg[100]
        V_inf = V_avg[150:].mean()
        AUC = V_avg[:101].sum() * dt
        T1 = next((k for k in range(horizon) if V_avg[k] <= 0.01 * V_avg[0]), -1)
        agg_metrics.append({
            'Method': name,
            'V100': V100,
            'V_inf': V_inf,
            'AUC0-100': AUC,
            'T1%': T1
        })

    # Save datasets
    pd.DataFrame(all_positions).to_csv(f'{out_dir}/all_positions.csv', index=False)
    pd.DataFrame(agg_metrics).to_csv(f'{out_dir}/aggregated_metrics.csv', index=False)
    np.savez(f'{out_dir}/error_curves.npz', **{m: np.load(f"{out_dir}/{m}_error.npy") for m in methods})
    print('Datasets created in', out_dir)
