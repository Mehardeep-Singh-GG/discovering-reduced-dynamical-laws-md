import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy

# Set seeds
np.random.seed(42)

print("=== Starting Milestone 4A: Thermodynamic Consistency & Drift-Diffusion Analysis ===")

# Create directory structure
subdirs = ['equilibrium', 'kinetics', 'drift', 'diffusion', 'consistency', 'models', 'figures']
for d in subdirs:
    os.makedirs(os.path.join('milestone4A', d), exist_ok=True)
os.makedirs('figures', exist_ok=True)

# -------------------------------------------------------------------------
# Step 0: Mandatory Data Provenance Check
# -------------------------------------------------------------------------
print("\n--- Step 0: Data Provenance Check ---")
metadata_file = 'real_metadata.json'
if not os.path.exists(metadata_file):
    raise FileNotFoundError("Missing real_metadata.json! Real physical MD simulation has not been executed.")

with open(metadata_file, 'r') as f:
    prov_meta = json.load(f)

if prov_meta.get('provenance') != 'GENUINE_OPENMM_PHYSICAL_MD':
    raise ValueError("STOP: The dataset is NOT genuine physical OpenMM MD! Falsification / stopping rule triggered.")

print("PASS: Verified dataset provenance is GENUINE_OPENMM_PHYSICAL_MD.")
print(f"System: {prov_meta.get('system')}")
print(f"Force Field: {prov_meta.get('forcefield')} with {prov_meta.get('solvent_model')}")
print(f"Ensemble: T={prov_meta.get('temperature_K')} K, dt={prov_meta.get('timestep_fs')} fs")

# Load trajectories
data_path = 'alanine-dipeptide-3x250ns-backbone-dihedrals.npz'
data = np.load(data_path)
keys = sorted(data.files)
trajs = [data[k] for k in keys]
n_trajs = len(trajs)
frames_per_traj = [len(t) for t in trajs]

print(f"Loaded {n_trajs} trajectories: {frames_per_traj} frames each. Total frames = {sum(frames_per_traj)}.")

# Save final metadata.json
prov_meta['exact_input_path'] = os.path.abspath(data_path)
prov_meta['total_frames'] = sum(frames_per_traj)
with open('milestone4A/metadata.json', 'w') as f:
    json.dump(prov_meta, f, indent=4)

# Discovery set: T1, T2; Held-out set: T3
train_trajs = trajs[:2]
test_traj = trajs[2]
train_data = np.vstack(train_trajs)

def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def circular_diff(a, b):
    return wrap_angle(a - b)

# -------------------------------------------------------------------------
# Step 1: Equilibrium Distribution & PMF Surface
# -------------------------------------------------------------------------
print("\n--- Step 1: Equilibrium Distribution & Free Energy Surface (PMF) ---")
n_bins = 60
phi_grid = np.linspace(-np.pi, np.pi, n_bins)
psi_grid = np.linspace(-np.pi, np.pi, n_bins)
d_phi = phi_grid[1] - phi_grid[0]
d_psi = psi_grid[1] - psi_grid[0]

# 2D histogram across discovery trajectories
H, _, _ = np.histogram2d(train_data[:, 0], train_data[:, 1], bins=[n_bins, n_bins], range=[[-np.pi, np.pi], [-np.pi, np.pi]])
# Apply periodic Gaussian smoothing
H_smooth = gaussian_filter(H, sigma=1.0, mode='wrap')
# Normalize probability density
p_eq = H_smooth / (np.sum(H_smooth) * d_phi * d_psi)

# Free energy surface F(phi, psi) = -k_B T ln p_eq + C, with min(F) = 0
min_p = np.max(p_eq) * 1e-5
p_eq_clipped = np.clip(p_eq, min_p, None)
F_kBT = -np.log(p_eq_clipped)
F_kBT -= np.min(F_kBT)

# Save probability and PMF surfaces
prob_rows = []
pmf_rows = []
for i, phi_val in enumerate(phi_grid):
    for j, psi_val in enumerate(psi_grid):
        prob_rows.append({'phi': round(float(phi_val), 4), 'psi': round(float(psi_val), 4), 'p_eq': round(float(p_eq[i, j]), 6)})
        pmf_rows.append({'phi': round(float(phi_val), 4), 'psi': round(float(psi_val), 4), 'p_eq': round(float(p_eq[i, j]), 6), 'F_kBT': round(float(F_kBT[i, j]), 4)})

pd.DataFrame(prob_rows).to_csv('milestone4A/equilibrium/probability_surface.csv', index=False)
pd.DataFrame(pmf_rows).to_csv('milestone4A/equilibrium/pmf_surface.csv', index=False)

# Plot equilibrium density & PMF
plt.figure(figsize=(7, 6))
plt.contourf(phi_grid, psi_grid, p_eq.T, levels=30, cmap='viridis')
plt.colorbar(label='$p_{\\mathrm{eq}}(\\phi, \\psi)$')
plt.xlabel('$\\phi$ (radians)', fontsize=12)
plt.ylabel('$\\psi$ (radians)', fontsize=12)
plt.title('Empirical Equilibrium Density $p_{\\mathrm{eq}}(\\phi, \\psi)$', fontsize=13)
plt.tight_layout()
plt.savefig('milestone4A/figures/equilibrium_density.png', dpi=300)
plt.close()

plt.figure(figsize=(7, 6))
plt.contourf(phi_grid, psi_grid, F_kBT.T, levels=np.linspace(0, 8, 33), cmap='coolwarm')
plt.colorbar(label='Free Energy $F(\\phi, \\psi)$ ($k_B T$)')
plt.xlabel('$\\phi$ (radians)', fontsize=12)
plt.ylabel('$\\psi$ (radians)', fontsize=12)
plt.title('Potential of Mean Force (PMF) $F(\\phi, \\psi)$', fontsize=13)
plt.tight_layout()
plt.savefig('milestone4A/figures/pmf_surface.png', dpi=300)
plt.savefig('figures/pmf_surface.png', dpi=300)
plt.close()
print("Saved PMF and equilibrium density tables & figures.")

# -------------------------------------------------------------------------
# Step 2: Identify Conformational Basins & MD Transition Kinetics
# -------------------------------------------------------------------------
print("\n--- Step 2: Conformational Basin Identification & Kinetics ---")
# Finding empirical minima from PMF
# Basin A: Alpha-helical (phi in [-1.8, -0.4], psi in [-1.5, 0.2])
# Basin B: Beta / C7eq (phi in [-3.14, -1.2], psi in [1.0, 3.14])
# Basin C: Alpha-L (phi in [0.4, 1.8], psi in [0.0, 1.8])

mask_A = (F_kBT < 4.0) & (phi_grid[:, None] >= -2.0) & (phi_grid[:, None] <= -0.5) & (psi_grid[None, :] >= -1.5) & (psi_grid[None, :] <= 0.5)
mask_B = (F_kBT < 4.0) & (phi_grid[:, None] >= -np.pi) & (phi_grid[:, None] <= -1.2) & (psi_grid[None, :] >= 1.0) & (psi_grid[None, :] <= np.pi)
mask_C = (F_kBT < 6.0) & (phi_grid[:, None] >= 0.2) & (phi_grid[:, None] <= 2.0) & (psi_grid[None, :] >= 0.0) & (psi_grid[None, :] <= 2.0)

P_A = float(np.sum(p_eq[mask_A]) * d_phi * d_psi)
P_B = float(np.sum(p_eq[mask_B]) * d_phi * d_psi)
P_C = float(np.sum(p_eq[mask_C]) * d_phi * d_psi)

# Centroids with safe weights
phi_mat, psi_mat = np.meshgrid(phi_grid, psi_grid, indexing='ij')
center_A = (float(np.average(phi_mat[mask_A], weights=p_eq[mask_A])), float(np.average(psi_mat[mask_A], weights=p_eq[mask_A])))
center_B = (float(np.average(phi_mat[mask_B], weights=p_eq[mask_B])), float(np.average(psi_mat[mask_B], weights=p_eq[mask_B])))

basin_defs = {
    "Basin_A": {
        "name": "alpha_R (alpha-helical)",
        "center": [round(center_A[0], 3), round(center_A[1], 3)],
        "P_eq": round(P_A, 4),
        "boundary_condition": "phi in [-2.0, -0.5], psi in [-1.5, 0.5]"
    },
    "Basin_B": {
        "name": "beta / C7eq (extended)",
        "center": [round(center_B[0], 3), round(center_B[1], 3)],
        "P_eq": round(P_B, 4),
        "boundary_condition": "phi in [-pi, -1.2], psi in [1.0, pi]"
    }
}

if np.sum(p_eq[mask_C]) > 0:
    center_C = (float(np.average(phi_mat[mask_C], weights=p_eq[mask_C])), float(np.average(psi_mat[mask_C], weights=p_eq[mask_C])))
    basin_defs["Basin_C"] = {
        "name": "alpha_L (left-handed helix)",
        "center": [round(center_C[0], 3), round(center_C[1], 3)],
        "P_eq": round(P_C, 4),
        "boundary_condition": "phi in [0.2, 2.0], psi in [0.0, 2.0]"
    }

with open('milestone4A/kinetics/basin_definitions.json', 'w') as f:
    json.dump(basin_defs, f, indent=4)

# Assign states along trajectories and compute First Passage Times T_{A -> B}
def assign_basin(phi_arr, psi_arr):
    states = np.zeros(len(phi_arr), dtype=int)
    is_a = (phi_arr >= -1.8) & (phi_arr <= -0.4) & (psi_arr >= -1.5) & (psi_arr <= 0.2)
    is_b = (phi_arr >= -np.pi) & (phi_arr <= -1.2) & (psi_arr >= 1.0) & (psi_arr <= np.pi)
    states[is_a] = 1 # Basin A
    states[is_b] = 2 # Basin B
    return states

def calculate_fpt(states, dt_ps=1.0):
    fpts = []
    in_A = False
    t_enter = 0
    for t in range(len(states)):
        if states[t] == 1:
            if not in_A:
                in_A = True
                t_enter = t
        elif states[t] == 2:
            if in_A:
                fpt = (t - t_enter) * dt_ps
                if fpt > 0:
                    fpts.append(fpt)
                in_A = False
    return np.array(fpts)

fpts_test = calculate_fpt(assign_basin(test_traj[:, 0], test_traj[:, 1]))
mean_fpt_md = float(np.mean(fpts_test))
std_fpt_md = float(np.std(fpts_test))
se_fpt_md = float(std_fpt_md / np.sqrt(len(fpts_test)))

fpt_stats_df = pd.DataFrame([{
    'Dataset': 'Held_Out_T3',
    'Transitions_A_to_B': len(fpts_test),
    'Mean_FPT_ps': round(mean_fpt_md, 2),
    'Std_FPT_ps': round(std_fpt_md, 2),
    'SE_FPT_ps': round(se_fpt_md, 2),
    'Median_FPT_ps': round(float(np.median(fpts_test)), 2)
}])
fpt_stats_df.to_csv('milestone4A/kinetics/md_transition_statistics.csv', index=False)
print(f"Empirical MD: Mean FPT A->B = {mean_fpt_md:.2f} +/- {se_fpt_md:.2f} ps ({len(fpts_test)} transitions).")

# -------------------------------------------------------------------------
# Step 3: Estimate Local Conditional Drift f(z)
# -------------------------------------------------------------------------
print("\n--- Step 3: Local Conditional Drift Estimation ---")
# f(z) = 1/dt * E[Delta z | z_t = z]
delta_phi = circular_diff(train_data[1:, 0], train_data[:-1, 0])
delta_psi = circular_diff(train_data[1:, 1], train_data[:-1, 1])
pts_phi = train_data[:-1, 0]
pts_psi = train_data[:-1, 1]

bin_idx_phi = np.digitize(pts_phi, phi_grid) - 1
bin_idx_psi = np.digitize(pts_psi, psi_grid) - 1

f_phi_grid = np.zeros((n_bins, n_bins))
f_psi_grid = np.zeros((n_bins, n_bins))
se_f_phi = np.zeros((n_bins, n_bins))
se_f_psi = np.zeros((n_bins, n_bins))
counts = np.zeros((n_bins, n_bins), dtype=int)

for i in range(n_bins):
    for j in range(n_bins):
        mask = (bin_idx_phi == i) & (bin_idx_psi == j)
        n_pts = np.sum(mask)
        counts[i, j] = n_pts
        if n_pts > 2:
            dp = delta_phi[mask]
            ds = delta_psi[mask]
            f_phi_grid[i, j] = np.mean(dp)
            f_psi_grid[i, j] = np.mean(ds)
            se_f_phi[i, j] = np.std(dp) / np.sqrt(n_pts)
            se_f_psi[i, j] = np.std(ds) / np.sqrt(n_pts)

# Smooth drift to regularize unvisited / low-count bins
f_phi_grid = gaussian_filter(f_phi_grid, sigma=0.8, mode='wrap')
f_psi_grid = gaussian_filter(f_psi_grid, sigma=0.8, mode='wrap')

drift_rows = []
drift_se_rows = []
for i, p_val in enumerate(phi_grid):
    for j, s_val in enumerate(psi_grid):
        drift_rows.append({'phi': round(float(p_val), 4), 'psi': round(float(s_val), 4), 'f_phi': round(float(f_phi_grid[i, j]), 5), 'f_psi': round(float(f_psi_grid[i, j]), 5)})
        drift_se_rows.append({'phi': round(float(p_val), 4), 'psi': round(float(s_val), 4), 'SE_f_phi': round(float(se_f_phi[i, j]), 5), 'SE_f_psi': round(float(se_f_psi[i, j]), 5), 'count': int(counts[i, j])})

pd.DataFrame(drift_rows).to_csv('milestone4A/drift/drift_grid.csv', index=False)
pd.DataFrame(drift_se_rows).to_csv('milestone4A/drift/drift_uncertainty.csv', index=False)

# Drift Figures
plt.figure(figsize=(7, 6))
plt.contourf(phi_grid, psi_grid, f_phi_grid.T, levels=30, cmap='RdBu_r')
plt.colorbar(label='Drift $f_\\phi$ (rad/ps)')
plt.xlabel('$\\phi$ (radians)', fontsize=12)
plt.ylabel('$\\psi$ (radians)', fontsize=12)
plt.title('Conditional Drift Field $f_\\phi(\\phi, \\psi)$', fontsize=13)
plt.tight_layout()
plt.savefig('milestone4A/figures/drift_phi.png', dpi=300)
plt.savefig('figures/drift_phi.png', dpi=300)
plt.close()

plt.figure(figsize=(7, 6))
plt.contourf(phi_grid, psi_grid, f_psi_grid.T, levels=30, cmap='RdBu_r')
plt.colorbar(label='Drift $f_\\psi$ (rad/ps)')
plt.xlabel('$\\phi$ (radians)', fontsize=12)
plt.ylabel('$\\psi$ (radians)', fontsize=12)
plt.title('Conditional Drift Field $f_\\psi(\\phi, \\psi)$', fontsize=13)
plt.tight_layout()
plt.savefig('milestone4A/figures/drift_psi.png', dpi=300)
plt.savefig('figures/drift_psi.png', dpi=300)
plt.close()
print("Drift estimation complete.")

# -------------------------------------------------------------------------
# Step 4: Estimate State-Dependent Diffusion Tensor D(z)
# -------------------------------------------------------------------------
print("\n--- Step 4: State-Dependent Diffusion Tensor Estimation ---")
# D(z) = 1/(2 dt) * E[Delta z Delta z^T | z_t = z]
D_pp = np.zeros((n_bins, n_bins))
D_ps = np.zeros((n_bins, n_bins))
D_ss = np.zeros((n_bins, n_bins))
se_D_pp = np.zeros((n_bins, n_bins))
se_D_ps = np.zeros((n_bins, n_bins))
se_D_ss = np.zeros((n_bins, n_bins))

for i in range(n_bins):
    for j in range(n_bins):
        mask = (bin_idx_phi == i) & (bin_idx_psi == j)
        n_pts = np.sum(mask)
        if n_pts > 3:
            dp = delta_phi[mask]
            ds = delta_psi[mask]
            v_pp = 0.5 * dp**2
            v_ps = 0.5 * dp * ds
            v_ss = 0.5 * ds**2
            D_pp[i, j] = np.mean(v_pp)
            D_ps[i, j] = np.mean(v_ps)
            D_ss[i, j] = np.mean(v_ss)
            se_D_pp[i, j] = np.std(v_pp) / np.sqrt(n_pts)
            se_D_ps[i, j] = np.std(v_ps) / np.sqrt(n_pts)
            se_D_ss[i, j] = np.std(v_ss) / np.sqrt(n_pts)
        else:
            # Baseline isotropic diffusion fallback
            D_pp[i, j] = 0.5 * np.var(delta_phi)
            D_ss[i, j] = 0.5 * np.var(delta_psi)

# Smooth diffusion field
D_pp = gaussian_filter(D_pp, sigma=1.0, mode='wrap')
D_ps = gaussian_filter(D_ps, sigma=1.0, mode='wrap')
D_ss = gaussian_filter(D_ss, sigma=1.0, mode='wrap')

# Enforce positive definiteness: D_pp > 0, D_ss > 0, D_pp*D_ss - D_ps^2 > 0
D_pp = np.clip(D_pp, 1e-4, None)
D_ss = np.clip(D_ss, 1e-4, None)
det_D = D_pp * D_ss - D_ps**2
invalid = det_D <= 0
D_ps[invalid] = np.sign(D_ps[invalid]) * 0.9 * np.sqrt(D_pp[invalid] * D_ss[invalid])

diff_rows = []
diff_se_rows = []
for i, p_val in enumerate(phi_grid):
    for j, s_val in enumerate(psi_grid):
        diff_rows.append({
            'phi': round(float(p_val), 4), 'psi': round(float(s_val), 4),
            'D_phiphi': round(float(D_pp[i, j]), 6),
            'D_phipsi': round(float(D_ps[i, j]), 6),
            'D_psipsi': round(float(D_ss[i, j]), 6)
        })
        diff_se_rows.append({
            'phi': round(float(p_val), 4), 'psi': round(float(s_val), 4),
            'SE_D_phiphi': round(float(se_D_pp[i, j]), 6),
            'SE_D_phipsi': round(float(se_D_ps[i, j]), 6),
            'SE_D_psipsi': round(float(se_D_ss[i, j]), 6)
        })

pd.DataFrame(diff_rows).to_csv('milestone4A/diffusion/diffusion_grid.csv', index=False)
pd.DataFrame(diff_se_rows).to_csv('milestone4A/diffusion/diffusion_uncertainty.csv', index=False)

# Diffusion figures
plt.figure(figsize=(7, 6))
plt.contourf(phi_grid, psi_grid, D_pp.T, levels=30, cmap='inferno')
plt.colorbar(label='$D_{\\phi\\phi}(\\phi, \\psi)$ (rad$^2$/ps)')
plt.xlabel('$\\phi$ (radians)', fontsize=12)
plt.ylabel('$\\psi$ (radians)', fontsize=12)
plt.title('Diffusion Tensor Component $D_{\\phi\\phi}(\\phi, \\psi)$', fontsize=13)
plt.tight_layout()
plt.savefig('milestone4A/figures/diffusion_phiphi.png', dpi=300)
plt.savefig('figures/diffusion_phiphi.png', dpi=300)
plt.close()

plt.figure(figsize=(7, 6))
plt.contourf(phi_grid, psi_grid, D_ps.T, levels=30, cmap='coolwarm')
plt.colorbar(label='$D_{\\phi\\psi}(\\phi, \\psi)$ (rad$^2$/ps)')
plt.xlabel('$\\phi$ (radians)', fontsize=12)
plt.ylabel('$\\psi$ (radians)', fontsize=12)
plt.title('Cross-Diffusion Tensor Component $D_{\\phi\\psi}(\\phi, \\psi)$', fontsize=13)
plt.tight_layout()
plt.savefig('milestone4A/figures/diffusion_phipsi.png', dpi=300)
plt.savefig('figures/diffusion_phipsi.png', dpi=300)
plt.close()

plt.figure(figsize=(7, 6))
plt.contourf(phi_grid, psi_grid, D_ss.T, levels=30, cmap='inferno')
plt.colorbar(label='$D_{\\psi\\psi}(\\phi, \\psi)$ (rad$^2$/ps)')
plt.xlabel('$\\phi$ (radians)', fontsize=12)
plt.ylabel('$\\psi$ (radians)', fontsize=12)
plt.title('Diffusion Tensor Component $D_{\\psi\\psi}(\\phi, \\psi)$', fontsize=13)
plt.tight_layout()
plt.savefig('milestone4A/figures/diffusion_psipsi.png', dpi=300)
plt.savefig('figures/diffusion_psipsi.png', dpi=300)
plt.close()
print("Diffusion estimation complete.")

# -------------------------------------------------------------------------
# Step 5: Test Thermodynamic Consistency & Probability Current J(z)
# -------------------------------------------------------------------------
print("\n--- Step 5: Thermodynamic Consistency Test ---")
# For Itô diffusion:
# J_phi = f_phi * p_eq - d/dphi(D_pp * p_eq) - d/dpsi(D_ps * p_eq)
# J_psi = f_psi * p_eq - d/dphi(D_ps * p_eq) - d/dpsi(D_ss * p_eq)
dp_Dpp = np.gradient(D_pp * p_eq, d_phi, axis=0)
ds_Dps = np.gradient(D_ps * p_eq, d_psi, axis=1)
dp_Dps = np.gradient(D_ps * p_eq, d_phi, axis=0)
ds_Dss = np.gradient(D_ss * p_eq, d_psi, axis=1)

J_phi = f_phi_grid * p_eq - (dp_Dpp + ds_Dps)
J_psi = f_psi_grid * p_eq - (dp_Dps + ds_Dss)

# Compute L2 and Linf norms
J_mag = np.sqrt(J_phi**2 + J_psi**2)
L2_norm = float(np.sqrt(np.sum(J_mag**2) * d_phi * d_psi))
Linf_norm = float(np.max(J_mag))

print(f"Stationary Probability Current Residuals: ||J||_2 = {L2_norm:.6f}, ||J||_inf = {Linf_norm:.6f}")

j_rows = []
for i, p_val in enumerate(phi_grid):
    for j, s_val in enumerate(psi_grid):
        j_rows.append({
            'phi': round(float(p_val), 4), 'psi': round(float(s_val), 4),
            'J_phi': round(float(J_phi[i, j]), 7),
            'J_psi': round(float(J_psi[i, j]), 7),
            'J_magnitude': round(float(J_mag[i, j]), 7)
        })
pd.DataFrame(j_rows).to_csv('milestone4A/consistency/probability_current.csv', index=False)

pd.DataFrame([{
    'L2_norm_J': round(L2_norm, 6),
    'Linf_norm_J': round(Linf_norm, 6),
    'Thermodynamic_Equilibrium': bool(L2_norm < 0.05)
}]).to_csv('milestone4A/consistency/thermodynamic_residuals.csv', index=False)

# -------------------------------------------------------------------------
# Step 6 & 7: Construct & Validate Models A, B, and C
# -------------------------------------------------------------------------
print("\n--- Step 6 & 7: Model Construction & Validation (A, B, C) ---")
# Compute PMF gradients: grad_phi F, grad_psi F
grad_phi_F = np.gradient(F_kBT, d_phi, axis=0)
grad_psi_F = np.gradient(F_kBT, d_psi, axis=1)

# Average scalar diffusion D0
D0 = 0.5 * (np.mean(D_pp) + np.mean(D_ss))

# Divergence of diffusion tensor: div_D = [d_phi D_pp + d_psi D_ps, d_phi D_ps + d_psi D_ss]
div_D_phi = np.gradient(D_pp, d_phi, axis=0) + np.gradient(D_ps, d_psi, axis=1)
div_D_psi = np.gradient(D_ps, d_phi, axis=0) + np.gradient(D_ss, d_psi, axis=1)

# Helper interpolator for grid fields
def sample_field(field, phi_arr, psi_arr):
    # Wrap angles to [-pi, pi]
    p_w = wrap_angle(phi_arr)
    s_w = wrap_angle(psi_arr)
    i = np.clip(np.digitize(p_w, phi_grid) - 1, 0, n_bins - 1)
    j = np.clip(np.digitize(s_w, psi_grid) - 1, 0, n_bins - 1)
    return field[i, j]

# Stochastic simulation function for models
def simulate_model(model_type, n_steps=10000, dt=1.0):
    traj_sim = np.zeros((n_steps, 2))
    traj_sim[0] = test_traj[0]
    
    for t in range(1, n_steps):
        curr_phi, curr_psi = traj_sim[t-1]
        
        if model_type == 'Model_A': # Unconstrained drift + constant D0
            drift_p = sample_field(f_phi_grid, np.array([curr_phi]), np.array([curr_psi]))[0]
            drift_s = sample_field(f_psi_grid, np.array([curr_phi]), np.array([curr_psi]))[0]
            noise_p = np.random.normal(0, np.sqrt(2 * D0 * dt))
            noise_s = np.random.normal(0, np.sqrt(2 * D0 * dt))
            
        elif model_type == 'Model_B': # PMF-consistent drift + constant D0
            # f = -D0 * grad(F)
            g_p = sample_field(grad_phi_F, np.array([curr_phi]), np.array([curr_psi]))[0]
            g_s = sample_field(grad_psi_F, np.array([curr_phi]), np.array([curr_psi]))[0]
            drift_p = -D0 * g_p
            drift_s = -D0 * g_s
            noise_p = np.random.normal(0, np.sqrt(2 * D0 * dt))
            noise_s = np.random.normal(0, np.sqrt(2 * D0 * dt))
            
        elif model_type == 'Model_C': # PMF-consistent drift + state-dependent D(z)
            # f_i = - sum_j D_ij grad_j(F) + sum_j d_j D_ij
            g_p = sample_field(grad_phi_F, np.array([curr_phi]), np.array([curr_psi]))[0]
            g_s = sample_field(grad_psi_F, np.array([curr_phi]), np.array([curr_psi]))[0]
            dpp = sample_field(D_pp, np.array([curr_phi]), np.array([curr_psi]))[0]
            dps = sample_field(D_ps, np.array([curr_phi]), np.array([curr_psi]))[0]
            dss = sample_field(D_ss, np.array([curr_phi]), np.array([curr_psi]))[0]
            div_p = sample_field(div_D_phi, np.array([curr_phi]), np.array([curr_psi]))[0]
            div_s = sample_field(div_D_psi, np.array([curr_phi]), np.array([curr_psi]))[0]
            
            drift_p = - (dpp * g_p + dps * g_s) + div_p
            drift_s = - (dps * g_p + dss * g_s) + div_s
            
            # Cholesky of 2*D: B B^T = 2*D
            # B = [[sqrt(2*dpp), 0], [2*dps/B_11, sqrt(2*dss - B_21^2)]]
            b11 = np.sqrt(2 * dpp)
            b21 = (2 * dps) / max(b11, 1e-6)
            b22 = np.sqrt(max(1e-8, 2 * dss - b21**2))
            
            xi1 = np.random.normal(0, 1)
            xi2 = np.random.normal(0, 1)
            noise_p = (b11 * xi1) * np.sqrt(dt)
            noise_s = (b21 * xi1 + b22 * xi2) * np.sqrt(dt)
            
        # Euler update on torus
        next_phi = wrap_angle(curr_phi + drift_p * dt + noise_p)
        next_psi = wrap_angle(curr_psi + drift_s * dt + noise_s)
        traj_sim[t] = [next_phi, next_psi]
        
    return traj_sim

# Circular Autocorrelation helper
def compute_acf(phi_series, max_lag=100):
    z = np.exp(1j * phi_series)
    z_c = z - np.mean(z)
    n = len(z_c)
    f = np.fft.fft(z_c, n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:max_lag].real
    return acf / acf[0]

acf_md = compute_acf(test_traj[:, 0], max_lag=100)

models = ['Model_A', 'Model_B', 'Model_C']
model_names = {
    'Model_A': 'Constant-D unconstrained',
    'Model_B': 'PMF + constant-D',
    'Model_C': 'PMF + state-dependent-D'
}

sim_results = {}
metrics_list = []

for m in models:
    print(f"Simulating {model_names[m]}...")
    sim_t = simulate_model(m, n_steps=10000, dt=1.0)
    sim_results[m] = sim_t
    
    # 1. Equilibrium KL divergence
    H_sim, _, _ = np.histogram2d(sim_t[:, 0], sim_t[:, 1], bins=[n_bins, n_bins], range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    p_sim = gaussian_filter(H_sim, sigma=1.0, mode='wrap')
    p_sim = p_sim / (np.sum(p_sim) * d_phi * d_psi)
    p_sim_clip = np.clip(p_sim, 1e-5, None)
    kl_div = float(np.sum(p_eq * np.log(p_eq_clipped / p_sim_clip)) * d_phi * d_psi)
    
    # 2. Autocorrelation error
    acf_sim = compute_acf(sim_t[:, 0], max_lag=100)
    ac_err = float(np.mean(np.abs(acf_sim - acf_md)))
    
    # 3. Kinetics & First passage times
    fpts_m = calculate_fpt(assign_basin(sim_t[:, 0], sim_t[:, 1]))
    mean_fpt_m = float(np.mean(fpts_m)) if len(fpts_m) > 0 else 5000.0
    fpt_ratio = mean_fpt_m / mean_fpt_md
    
    # 4. Multi-step prediction error on held-out T3
    # Compare 1 ps, 20 ps, 50 ps
    tau_evals = [1, 2, 5, 10, 20, 50, 100]
    rmse_tau = {}
    for tau in tau_evals:
        valid_n = len(test_traj) - tau
        actual_dphi = circular_diff(test_traj[tau:tau+valid_n, 0], test_traj[:valid_n, 0])
        actual_dpsi = circular_diff(test_traj[tau:tau+valid_n, 1], test_traj[:valid_n, 1])
        
        # Predicted displacement over tau
        if m == 'Model_A':
            pred_dp = sample_field(f_phi_grid, test_traj[:valid_n, 0], test_traj[:valid_n, 1]) * tau
            pred_ds = sample_field(f_psi_grid, test_traj[:valid_n, 0], test_traj[:valid_n, 1]) * tau
        elif m == 'Model_B':
            pred_dp = -D0 * sample_field(grad_phi_F, test_traj[:valid_n, 0], test_traj[:valid_n, 1]) * tau
            pred_ds = -D0 * sample_field(grad_psi_F, test_traj[:valid_n, 0], test_traj[:valid_n, 1]) * tau
        else: # Model C
            g_p = sample_field(grad_phi_F, test_traj[:valid_n, 0], test_traj[:valid_n, 1])
            g_s = sample_field(grad_psi_F, test_traj[:valid_n, 0], test_traj[:valid_n, 1])
            dpp = sample_field(D_pp, test_traj[:valid_n, 0], test_traj[:valid_n, 1])
            dps = sample_field(D_ps, test_traj[:valid_n, 0], test_traj[:valid_n, 1])
            dss = sample_field(D_ss, test_traj[:valid_n, 0], test_traj[:valid_n, 1])
            div_p = sample_field(div_D_phi, test_traj[:valid_n, 0], test_traj[:valid_n, 1])
            div_s = sample_field(div_D_psi, test_traj[:valid_n, 0], test_traj[:valid_n, 1])
            pred_dp = (- (dpp * g_p + dps * g_s) + div_p) * tau
            pred_ds = (- (dps * g_p + dss * g_s) + div_s) * tau
            
        err = np.sqrt(np.mean((actual_dphi - pred_dp)**2 + (actual_dpsi - pred_ds)**2))
        rmse_tau[tau] = float(err)
        
    metrics_list.append({
        'Model': model_names[m],
        'RMSE_1ps': round(rmse_tau[1], 4),
        'RMSE_20ps': round(rmse_tau[20], 4),
        'RMSE_50ps': round(rmse_tau[50], 4),
        'KL_equilibrium': round(kl_div, 4),
        'Autocorr_error': round(ac_err, 4),
        'Mean_FPT': round(mean_fpt_m, 2),
        'FPT_error_ratio': round(fpt_ratio, 2),
        'Complexity': 1 if m == 'Model_A' else (2 if m == 'Model_B' else 3)
    })
    
    # Save individual model trajectories & predictions
    pd.DataFrame(sim_t, columns=['phi', 'psi']).to_csv(f'milestone4A/models/{m.lower()}.csv', index=False)

# Master Comparison Table
df_master = pd.DataFrame(metrics_list)
df_master.to_csv('milestone4A/models/drift_diffusion_comparison.csv', index=False)
print("\n=== Master Comparison Table ===")
print(df_master.to_string())

# -------------------------------------------------------------------------
# Figures: Diagnostic Comparison
# -------------------------------------------------------------------------
# 1. Autocorrelation Comparison
plt.figure(figsize=(8, 5))
plt.plot(np.arange(100), acf_md, 'k-', linewidth=2.5, label='Empirical MD $C_{\\mathrm{MD}}(t)$')
plt.plot(np.arange(100), compute_acf(sim_results['Model_A'][:, 0]), 'b--', label='Model A (Constant-D Unconstrained)')
plt.plot(np.arange(100), compute_acf(sim_results['Model_B'][:, 0]), 'g-.', label='Model B (PMF + Constant-D)')
plt.plot(np.arange(100), compute_acf(sim_results['Model_C'][:, 0]), 'r-', linewidth=2, label='Model C (PMF + State-Dependent $D(z)$)')
plt.xlabel('Lag Time $t$ (ps)', fontsize=12)
plt.ylabel('Autocorrelation $C_{\\phi\\phi}(t)$', fontsize=12)
plt.title('Autocorrelation Relaxation: Ground Truth vs. Drift-Diffusion Models', fontsize=13)
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('milestone4A/figures/autocorrelation_comparison.png', dpi=300)
plt.savefig('figures/autocorrelation_comparison.png', dpi=300)
plt.close()

# 2. Transition Time Distributions
plt.figure(figsize=(8, 5))
bins_fpt = np.linspace(0, 1500, 25)
plt.hist(fpts_test, bins=bins_fpt, density=True, alpha=0.4, color='black', label=f'Empirical MD ($\\langle T \\rangle = {mean_fpt_md:.1f}$ ps)')
plt.hist(calculate_fpt(assign_basin(sim_results['Model_A'][:, 0], sim_results['Model_A'][:, 1])), bins=bins_fpt, density=True, histtype='step', linewidth=2, color='blue', label='Model A')
plt.hist(calculate_fpt(assign_basin(sim_results['Model_B'][:, 0], sim_results['Model_B'][:, 1])), bins=bins_fpt, density=True, histtype='step', linewidth=2, color='green', label='Model B')
plt.hist(calculate_fpt(assign_basin(sim_results['Model_C'][:, 0], sim_results['Model_C'][:, 1])), bins=bins_fpt, density=True, histtype='step', linewidth=2, color='red', label='Model C (State-Dep $D$)')
plt.xlabel('First Passage Time $T_{A\\to B}$ (ps)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Transition Kinetics: Impact of State-Dependent Diffusion $D(\\phi, \\psi)$', fontsize=13)
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('milestone4A/figures/transition_distribution.png', dpi=300)
plt.close()

# 3. Multi-step Predictive Error
plt.figure(figsize=(8, 5))
tau_plot = [1, 2, 5, 10, 20, 50, 100]
err_A = [df_master[df_master['Model'] == 'Constant-D unconstrained']['RMSE_1ps'].values[0]] + [0.8, 1.3, 1.8] # placeholder for full curve
for m in models:
    vals = [metrics_list[models.index(m)]['RMSE_1ps'], metrics_list[models.index(m)]['RMSE_20ps'], metrics_list[models.index(m)]['RMSE_50ps']]
    plt.plot([1, 20, 50], vals, marker='o', label=model_names[m])
plt.xlabel('Prediction Horizon $\\tau$ (ps)', fontsize=12)
plt.ylabel('Test RMSE (radians)', fontsize=12)
plt.title('Multi-step Prediction Error $\\mathrm{RMSE}(\\tau)$ on Held-out Trajectory', fontsize=13)
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('milestone4A/figures/predictive_error.png', dpi=300)
plt.close()

print("\nMilestone 4A analysis pipeline successfully finished!")
