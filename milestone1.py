import numpy as np
import os
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.signal import correlate

# 1. Download Dataset
local_filename = 'alanine-dipeptide-3x250ns-backbone-dihedrals.npz'

# 2. Load Data
with np.load(local_filename) as data:
    # Files are usually something like arr_0, arr_1, arr_2 for 3 trajectories
    keys = data.files
    print(f"Dataset keys: {keys}")
    trajectories = [data[k] for k in keys]

n_trajectories = len(trajectories)
frames_per_traj = [traj.shape[0] for traj in trajectories]
print(f"Number of trajectories: {n_trajectories}")
print(f"Frames per trajectory: {frames_per_traj}")

# In the prompt: 250 ns total per trajectory, 250,000 frames. 
# Therefore frame spacing = 250 ns / 250,000 = 1 ps = 1000 fs.
dt_ps = 1.0

# Concatenate for some global stats, but keep separate for time series
all_data = np.concatenate(trajectories)
phi_all = all_data[:, 0]
psi_all = all_data[:, 1]

# 3. Data Audit
audit_data = {
    "missing_values": int(np.isnan(all_data).sum()),
    "range_phi": [float(np.min(phi_all)), float(np.max(phi_all))],
    "range_psi": [float(np.min(psi_all)), float(np.max(psi_all))]
}
print("Audit data:", audit_data)
pd.DataFrame([audit_data]).to_csv('data_audit.csv', index=False)

# 4. Ramachandran Plot
plt.figure(figsize=(8, 6))
plt.hist2d(phi_all, psi_all, bins=100, cmap='viridis', norm='log')
plt.colorbar(label='Count (log scale)')
plt.xlabel(r'$\phi$ (radians)')
plt.ylabel(r'$\psi$ (radians)')
plt.title('Ramachandran Plot of Alanine Dipeptide')
plt.xlim(-np.pi, np.pi)
plt.ylim(-np.pi, np.pi)
plt.savefig('fig1.png', dpi=300)
plt.close()

# 5. Time Series Plot
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i in range(3):
    time_ns = np.arange(1000) * dt_ps / 1000.0 # Just plot first 1000 frames (1 ns) for visibility
    axs[i].plot(time_ns, trajectories[i][:1000, 0], label=r'$\phi$', alpha=0.7)
    axs[i].plot(time_ns, trajectories[i][:1000, 1], label=r'$\psi$', alpha=0.7)
    axs[i].set_ylabel(f'Trajectory {i+1}')
    axs[i].legend(loc='upper right')
axs[2].set_xlabel('Time (ns)')
plt.suptitle('Sample Trajectories (first 1 ns)')
plt.tight_layout()
plt.savefig('fig2.png', dpi=300)
plt.close()

# 6. Autocorrelation Functions
def circular_autocorr(x, max_lag):
    # For periodic variables, using exp(i*x) is better
    z = np.exp(1j * x)
    z_mean = np.mean(z)
    z_centered = z - z_mean
    
    # Compute using fft
    n = len(z_centered)
    f = np.fft.fft(z_centered, n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:max_lag].real
    acf /= acf[0]
    return acf

max_lag = 10000 # 10 ns
acf_phi = np.zeros(max_lag)
acf_psi = np.zeros(max_lag)

for traj in trajectories:
    acf_phi += circular_autocorr(traj[:, 0], max_lag)
    acf_psi += circular_autocorr(traj[:, 1], max_lag)

acf_phi /= n_trajectories
acf_psi /= n_trajectories

time_lags = np.arange(max_lag) * dt_ps
plt.figure(figsize=(8, 5))
plt.plot(time_lags, acf_phi, label='$\phi$ autocorrelation')
plt.plot(time_lags, acf_psi, label='$\psi$ autocorrelation')
plt.xlabel('Lag Time (ps)')
plt.ylabel('Autocorrelation')
plt.title('Circular Autocorrelation Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('autocorrelation.png', dpi=300)
plt.close()

# 7. Metadata JSON
metadata = {
    "dataset": "alanine-dipeptide-3x250ns-backbone-dihedrals",
    "n_trajectories": 3,
    "frames_per_trajectory": 250000,
    "temperature_K": 300,
    "solvent": "TIP3P",
    "forcefield": "AMBER ff99SB-ILDN",
    "frame_spacing_ps": 1.0,
    "timestep_fs": 2.0, # based on OpenMM setup mentioned in prompt
    "discovery_trajectories": ["T1", "T2"],
    "validation_trajectory": ["T3"],
    "representations_tested": [
        "(phi,psi)"
    ],
    "lags_ps": [],
    "models_tested": [],
    "interventions": []
}
with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("Milestone 1 execution complete. Outputs generated.")
