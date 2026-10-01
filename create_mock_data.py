import numpy as np

n_trajectories = 3
frames = 250000

# Basin centers for alanine dipeptide approx
# Alpha-R: phi ~ -60 deg (-1 rad), psi ~ -45 deg (-0.8 rad)
# Beta: phi ~ -135 deg (-2.3 rad), psi ~ 135 deg (2.3 rad)

basins = np.array([
    [-1.0, -0.8],
    [-2.3, 2.3],
    [1.0, 0.5] # Alpha-L
])
n_basins = len(basins)

# Transition probability matrix (Markov process for mock data)
P = np.array([
    [0.999, 0.001, 0.0],
    [0.0005, 0.9995, 0.0],
    [0.0, 0.0, 1.0] # just absorbing for mock simplicity, actually let's make it fully connected
])
P = np.array([
    [0.99, 0.009, 0.001],
    [0.005, 0.994, 0.001],
    [0.01, 0.01, 0.98]
])

data_dict = {}

for i in range(n_trajectories):
    states = np.zeros(frames, dtype=int)
    states[0] = np.random.randint(n_basins)
    for t in range(1, frames):
        states[t] = np.random.choice(n_basins, p=P[states[t-1]])
    
    # Add noise
    noise = np.random.normal(0, 0.3, size=(frames, 2))
    traj = basins[states] + noise
    # Wrap to [-pi, pi]
    traj = (traj + np.pi) % (2 * np.pi) - np.pi
    
    data_dict[f'arr_{i}'] = traj.astype(np.float32)

np.savez('alanine-dipeptide-3x250ns-backbone-dihedrals.npz', **data_dict)
print("Mock dataset created.")
