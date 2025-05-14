import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
N = 5
dt = 0.05
horizon = 201     # time steps 0…200
trials = 30       # Monte Carlo trials
F = 1             # W-MSR trimming parameter
c = 1.0           # Huber threshold

# Complete graph neighbors
neighbors = {i: [j for j in range(N) if j != i] for i in range(N)}

# Pentagon formation positions
angles = np.linspace(0, 2*np.pi, N, endpoint=False)
P = np.stack([np.cos(angles), np.sin(angles)], axis=1)

# Desired displacement map
d_map = {}
for i in range(N):
    for j in range(i+1, N):
        d_map[(i, j)] = P[i] - P[j]

# Attack parameters
attacked = 2
attack_vec = np.array([3.0, 3.0])

# SOSH parameters
gamma = 0.3
M = 0.5

# Helper functions
def delta_fun(x):
    norm = np.linalg.norm(x)
    return (M/2) * norm**2 * x / (norm + 1e-6)

def formation_error(x):
    total = 0.0
    for (i, j), disp in d_map.items():
        err = x[i] - x[j] - disp
        total += err.dot(err)
    return 0.5 * total

def get_disp(i, j):
    if (i, j) in d_map:   return d_map[(i, j)]
    if (j, i) in d_map:   return -d_map[(j, i)]
    return np.zeros(2)

# Simulation engine
def simulate_step(step_fn):
    V = np.zeros(horizon)
    for _ in range(trials):
        x = np.random.uniform(-0.5, 1.5, (N, 2))
        prev = x.copy()
        for k in range(horizon):
            V[k] += formation_error(x)
            y = x.copy()
            x_next = x.copy()
            step_fn(x, y, prev, x_next)
            prev = x.copy()
            x = x_next
    return V / trials

# Define each mitigation
def attack_step(x, y, prev, x_next):
    y[attacked] = x[attacked] + attack_vec
    for i in range(N):
        s = sum((x[i] - y[j] - get_disp(i,j)) for j in neighbors[i])
        x_next[i] = x[i] - dt * s

def sosh_step(x, y, prev, x_next):
    y[attacked] = gamma*x[attacked] + delta_fun(x[attacked])
    for i in range(N):
        s = sum((x[i] - y[j] - get_disp(i,j)) for j in neighbors[i])
        x_next[i] = x[i] - dt * s

def wmsr_step(x, y, prev, x_next):
    y[attacked] = x[attacked] + attack_vec
    for i in range(N):
        vals = sorted(
            [(j, y[j], np.linalg.norm(x[i]-y[j])) for j in neighbors[i]],
            key=lambda t: t[2]
        )
        trimmed = vals[F:len(vals)-F]
        s = sum((x[i] - yj - get_disp(i,j)) for j,yj,_ in trimmed)
        x_next[i] = x[i] - dt * s

def huber_step(x, y, prev, x_next):
    y[attacked] = x[attacked] + attack_vec
    for i in range(N):
        s = np.zeros(2)
        for j in neighbors[i]:
            err = x[i] - y[j] - get_disp(i,j)
            norm = np.linalg.norm(err)
            w = 1 if norm<=c else c/norm
            s += w*err
        x_next[i] = x[i] - dt * s

# Run
V_attack = simulate_step(attack_step)
V_sosh   = simulate_step(sosh_step)
V_wmsr   = simulate_step(wmsr_step)
V_huber  = simulate_step(huber_step)

# Plot
plt.figure(figsize=(8,5))
plt.plot(V_attack, label='No Mitigation')
plt.plot(V_sosh,   label='SOSH')
plt.plot(V_wmsr,   label='W-MSR')
plt.plot(V_huber,  label='Huber')
plt.xlabel('Time Step k')
plt.ylabel('Average Formation Error V[k]')
plt.title('Mitigation Comparison (5-Node Complete Graph)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Metrics
def metrics(V):
    V100 = V[100]
    V_inf = V[150:].mean()
    AUC = V[:101].sum() * dt
    T1 = next((k for k in range(horizon) if V[k]<=0.01*V[0]), '—')
    return V100, V_inf, AUC, T1

methods = ['No Mitigation','SOSH','W-MSR','Huber']
rows = [metrics(V) for V in [V_attack, V_sosh, V_wmsr, V_huber]]
df = pd.DataFrame(rows, index=methods, columns=['V100','V_inf','AUC0-100','T1%'])
print(df)
