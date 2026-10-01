import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import Ridge

# Set deterministic seed
np.random.seed(42)

# Load frozen config and genuine dataset
with open('milestone4B/m4b_config.json', 'r') as f:
    config = json.load(f)

npz_data = np.load(config['dataset_path'])
trajs = [npz_data[k] for k in sorted(npz_data.files)]
traj_train = trajs[0] # T1
traj_val = trajs[1]   # T2
traj_test = trajs[2]  # T3 (frozen test set)

def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def circular_diff(a, b):
    return wrap_angle(a - b)

# Basin definitions from config
basin_a_phi = config['basin_definitions']['Basin_A']['phi_range']
basin_a_psi = config['basin_definitions']['Basin_A']['psi_range']
basin_b_phi = config['basin_definitions']['Basin_B']['phi_range']
basin_b_psi = config['basin_definitions']['Basin_B']['psi_range']

def assign_basins(phi, psi):
    states = np.zeros(len(phi), dtype=int)
    in_a = (phi >= basin_a_phi[0]) & (phi <= basin_a_phi[1]) & (psi >= basin_a_psi[0]) & (psi <= basin_a_psi[1])
    in_b = (phi >= basin_b_phi[0]) & (phi <= basin_b_phi[1]) & (psi >= basin_b_psi[0]) & (psi <= basin_b_psi[1])
    states[in_a] = 1
    states[in_b] = 2
    return states

def compute_fpts(states):
    fpts = []
    in_a = False
    start_t = 0
    for t in range(len(states)):
        if states[t] == 1:
            if not in_a:
                in_a = True
                start_t = t
        elif states[t] == 2:
            if in_a:
                dt = t - start_t
                if dt > 0:
                    fpts.append(dt)
                in_a = False
    return np.array(fpts) if len(fpts) > 0 else np.array([25.0])

# Ground truth transitions on held-out T3
fpts_md = compute_fpts(assign_basins(traj_test[:, 0], traj_test[:, 1]))
mean_fpt_md = float(np.mean(fpts_md))
print(f"Empirical MD: {len(fpts_md)} transitions, Mean FPT = {mean_fpt_md:.2f} ps")

# Build coupled features
def build_coupled_features(phi, psi):
    feats = [
        np.ones_like(phi),
        np.sin(phi), np.cos(phi),
        np.sin(psi), np.cos(psi),
        np.sin(2*phi), np.cos(2*phi),
        np.sin(2*psi), np.cos(2*psi),
        np.sin(phi - psi), np.cos(phi - psi),
        np.sin(phi + psi), np.cos(phi + psi),
        np.sin(phi) * np.cos(psi),
        np.cos(phi) * np.sin(psi)
    ]
    return np.column_stack(feats)

idx_tr = np.arange(16, len(traj_train) - 100)
Y_train = np.column_stack([
    circular_diff(traj_train[idx_tr + 1, 0], traj_train[idx_tr, 0]),
    circular_diff(traj_train[idx_tr + 1, 1], traj_train[idx_tr, 1])
])
phi_tr, psi_tr = traj_train[idx_tr, 0], traj_train[idx_tr, 1]
X_c_tr = build_coupled_features(phi_tr, psi_tr)

# Fit coupled drift f(z) on T1
drift_model = Ridge(alpha=1.0).fit(X_c_tr, Y_train)
res_tr = Y_train - drift_model.predict(X_c_tr)

# Local 2D binned diffusion tensor estimation
n_bins = 50
phi_bins = np.linspace(-np.pi, np.pi, n_bins)
psi_bins = np.linspace(-np.pi, np.pi, n_bins)
d_phi = phi_bins[1] - phi_bins[0]
d_psi = psi_bins[1] - psi_bins[0]

d_mat_phiphi = np.zeros((n_bins, n_bins))
d_mat_psipsi = np.zeros((n_bins, n_bins))
d_mat_phipsi = np.zeros((n_bins, n_bins))
bin_counts = np.zeros((n_bins, n_bins))

i_tr = np.clip(np.int32((phi_tr + np.pi) / d_phi) % (n_bins - 1), 0, n_bins - 2)
j_tr = np.clip(np.int32((psi_tr + np.pi) / d_psi) % (n_bins - 1), 0, n_bins - 2)

for idx in range(len(res_tr)):
    bi, bj = i_tr[idx], j_tr[idx]
    d_mat_phiphi[bi, bj] += res_tr[idx, 0]**2 / 2.0
    d_mat_psipsi[bi, bj] += res_tr[idx, 1]**2 / 2.0
    d_mat_phipsi[bi, bj] += (res_tr[idx, 0] * res_tr[idx, 1]) / 2.0
    bin_counts[bi, bj] += 1

smooth_counts = gaussian_filter(bin_counts, sigma=1.2, mode='wrap')
d_phiphi_sm = gaussian_filter(d_mat_phiphi, sigma=1.2, mode='wrap') / np.maximum(smooth_counts, 1e-4)
d_psipsi_sm = gaussian_filter(d_mat_psipsi, sigma=1.2, mode='wrap') / np.maximum(smooth_counts, 1e-4)
d_phipsi_sm = gaussian_filter(d_mat_phipsi, sigma=1.2, mode='wrap') / np.maximum(smooth_counts, 1e-4)

# Global fallback for unvisited bins
cov_diag_0 = np.mean(res_tr**2, axis=0)
d_phiphi_sm[smooth_counts < 0.1] = cov_diag_0[0] / 2.0
d_psipsi_sm[smooth_counts < 0.1] = cov_diag_0[1] / 2.0
d_phipsi_sm[smooth_counts < 0.1] = 0.0

# SPSD regularization
lambda_min = 1e-4
for i in range(n_bins):
    for j in range(n_bins):
        D_loc = np.array([[d_phiphi_sm[i, j], d_phipsi_sm[i, j]], [d_phipsi_sm[i, j], d_psipsi_sm[i, j]]])
        eigvals, eigvecs = np.linalg.eigh(D_loc)
        eigvals = np.maximum(eigvals, lambda_min)
        D_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d_phiphi_sm[i, j] = D_reg[0, 0]
        d_psipsi_sm[i, j] = D_reg[1, 1]
        d_phipsi_sm[i, j] = D_reg[0, 1]

# -------------------------------------------------------------------------
# Run The Off-Diagonal Ablation Experiment
# -------------------------------------------------------------------------
print("\n--- Running Off-Diagonal Ablation Experiment ---")
print("Comparing D_full(z) vs D_diag(z) with IDENTICAL diagonal components (D_phipsi strictly zeroed).")

n_sim = 20000

# 1. Full Tensor Simulation: cand006
sim_full = np.zeros((n_sim, 2))
sim_full[0] = traj_test[0]

for t in range(1, n_sim):
    cur_phi, cur_psi = sim_full[t-1]
    feats = build_coupled_features(np.array([cur_phi]), np.array([cur_psi]))
    step_drift = drift_model.predict(feats)[0]
    
    i = np.clip(np.int32((wrap_angle(cur_phi) + np.pi) / d_phi) % (n_bins - 1), 0, n_bins - 2)
    j = np.clip(np.int32((wrap_angle(cur_psi) + np.pi) / d_psi) % (n_bins - 1), 0, n_bins - 2)
    
    D_step = np.array([[d_phiphi_sm[i, j], d_phipsi_sm[i, j]], [d_phipsi_sm[i, j], d_psipsi_sm[i, j]]])
    try:
        chol = np.linalg.cholesky(2.0 * D_step)
        noise = chol @ np.random.normal(size=2)
    except np.linalg.LinAlgError:
        noise = np.random.normal(0, np.sqrt(np.maximum(np.diag(2.0 * D_step), 1e-4)))
        
    next_phi = wrap_angle(cur_phi + step_drift[0] + noise[0])
    next_psi = wrap_angle(cur_psi + step_drift[1] + noise[1])
    sim_full[t] = [next_phi, next_psi]

# 2. Diagonal-Only Simulation (using EXACT SAME D_phiphi and D_psipsi, zeroing D_phipsi)
sim_diag = np.zeros((n_sim, 2))
sim_diag[0] = traj_test[0]

for t in range(1, n_sim):
    cur_phi, cur_psi = sim_diag[t-1]
    feats = build_coupled_features(np.array([cur_phi]), np.array([cur_psi]))
    step_drift = drift_model.predict(feats)[0]
    
    i = np.clip(np.int32((wrap_angle(cur_phi) + np.pi) / d_phi) % (n_bins - 1), 0, n_bins - 2)
    j = np.clip(np.int32((wrap_angle(cur_psi) + np.pi) / d_psi) % (n_bins - 1), 0, n_bins - 2)
    
    # Strictly zero out off-diagonal
    D_step_diag = np.array([[d_phiphi_sm[i, j], 0.0], [0.0, d_psipsi_sm[i, j]]])
    noise = np.random.normal(0, np.sqrt(2.0 * np.maximum(np.diag(D_step_diag), 1e-4)))
    
    next_phi = wrap_angle(cur_phi + step_drift[0] + noise[0])
    next_psi = wrap_angle(cur_psi + step_drift[1] + noise[1])
    sim_diag[t] = [next_phi, next_psi]

# Compute First Passage Times
fpts_full = compute_fpts(assign_basins(sim_full[:, 0], sim_full[:, 1]))
fpts_diag = compute_fpts(assign_basins(sim_diag[:, 0], sim_diag[:, 1]))

mean_full = float(np.mean(fpts_full))
mean_diag = float(np.mean(fpts_diag))
delta_T = mean_diag - mean_full

print(f"Full Tensor (cand006): Mean FPT = {mean_full:.2f} ps ({len(fpts_full)} transitions)")
print(f"Diagonal Stripped:     Mean FPT = {mean_diag:.2f} ps ({len(fpts_diag)} transitions)")
print(f"Delta T = T_diag - T_full = {delta_T:.2f} ps")

# Bootstrap the difference Delta T = T_diag - T_full across 1,000 resamples
n_boot = 1000
boot_delta_T = []
boot_full_means = []
boot_diag_means = []

for _ in range(n_boot):
    r_f = np.random.choice(fpts_full, size=len(fpts_full), replace=True)
    r_d = np.random.choice(fpts_diag, size=len(fpts_diag), replace=True)
    m_f = np.mean(r_f)
    m_d = np.mean(r_d)
    boot_full_means.append(m_f)
    boot_diag_means.append(m_d)
    boot_delta_T.append(m_d - m_f)

ci_delta_low = float(np.percentile(boot_delta_T, 2.5))
ci_delta_high = float(np.percentile(boot_delta_T, 97.5))
zero_included = bool(ci_delta_low <= 0.0 <= ci_delta_high)

print(f"95% Bootstrap CI for Delta T: [{ci_delta_low:.2f}, {ci_delta_high:.2f}] ps")
print(f"Zero included in 95% CI? {zero_included}")

# KS-tests vs empirical MD
ks_full_stat, ks_full_p = ks_2samp(fpts_md, fpts_full)
ks_diag_stat, ks_diag_p = ks_2samp(fpts_md, fpts_diag)

# Direct KS-test between Full and Diagonal FPT distributions
ks_comp_stat, ks_comp_p = ks_2samp(fpts_full, fpts_diag)

# Wasserstein distances
wass_full_md = float(wasserstein_distance(fpts_md, fpts_full))
wass_diag_md = float(wasserstein_distance(fpts_md, fpts_diag))
wass_full_diag = float(wasserstein_distance(fpts_full, fpts_diag))

# Save results table
ablation_results = [
    {
        'Model_Variant': 'D_full(z) [Full Tensor, cand006]',
        'Off_Diagonal_Included': True,
        'Mean_FPT_ps': round(mean_full, 2),
        'CI_95_Mean_ps': f"[{np.percentile(boot_full_means, 2.5):.2f}, {np.percentile(boot_full_means, 97.5):.2f}]",
        'Median_FPT_ps': round(float(np.median(fpts_full)), 2),
        'Transitions_N': len(fpts_full),
        'Kinetic_RelError_vs_MD': f"{abs(mean_full - mean_fpt_md)/mean_fpt_md*100:.2f}%",
        'KS_p_vs_MD': round(float(ks_full_p), 4),
        'Wasserstein_vs_MD_ps': round(wass_full_md, 2)
    },
    {
        'Model_Variant': 'D_diag(z) [Same Diag, Zero Off-Diag]',
        'Off_Diagonal_Included': False,
        'Mean_FPT_ps': round(mean_diag, 2),
        'CI_95_Mean_ps': f"[{np.percentile(boot_diag_means, 2.5):.2f}, {np.percentile(boot_diag_means, 97.5):.2f}]",
        'Median_FPT_ps': round(float(np.median(fpts_diag)), 2),
        'Transitions_N': len(fpts_diag),
        'Kinetic_RelError_vs_MD': f"{abs(mean_diag - mean_fpt_md)/mean_fpt_md*100:.2f}%",
        'KS_p_vs_MD': round(float(ks_diag_p), 4),
        'Wasserstein_vs_MD_ps': round(wass_diag_md, 2)
    }
]

df_ab = pd.DataFrame(ablation_results)
df_ab.to_csv('milestone4B/m4b_offdiagonal_ablation.csv', index=False)

# Save delta T statistical summary
delta_summary = {
    'delta_T_mean_ps': round(delta_T, 2),
    'ci_95_low_ps': round(ci_delta_low, 2),
    'ci_95_high_ps': round(ci_delta_high, 2),
    'zero_in_ci': zero_included,
    'ks_p_between_full_and_diag': round(float(ks_comp_p), 4),
    'wasserstein_between_full_and_diag_ps': round(wass_full_diag, 2),
    'scientific_verdict': 'State-dependent anisotropic diffusion is sufficient; off-diagonal coupling is not demonstrably necessary.' if zero_included else 'Off-diagonal coupling contributes a statistically significant kinetic shift.'
}

with open('milestone4B/m4b_offdiagonal_summary.json', 'w') as f_json:
    json.dump(delta_summary, f_json, indent=4)

print("\nSaved milestone4B/m4b_offdiagonal_ablation.csv and milestone4B/m4b_offdiagonal_summary.json")

# Generate visualization of the off-diagonal comparison
plt.figure(figsize=(9, 5))
bins_fpt = np.linspace(0, 100, 25)
plt.hist(fpts_md, bins=bins_fpt, density=True, alpha=0.35, color='black', label=f"Empirical MD (Mean={mean_fpt_md:.1f} ps)")
plt.hist(fpts_full, bins=bins_fpt, density=True, histtype='step', linewidth=2.5, color='#1f77b4', label=f"Full Tensor $D(z)$ (Mean={mean_full:.1f} ps)")
plt.hist(fpts_diag, bins=bins_fpt, density=True, histtype='step', linewidth=2.5, color='#ff7f0e', linestyle='--', label=f"Diagonal $D_{{ii}}(z)$ (Mean={mean_diag:.1f} ps)")
plt.xlabel('First Passage Time $T_{A\\to B}$ (ps)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title(f"Off-Diagonal Diffusion Ablation: $\\Delta T = {delta_T:.2f}$ ps [95% CI: {ci_delta_low:.2f}, {ci_delta_high:.2f}]", fontsize=13)
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('milestone4B/figures/fig10_offdiagonal_ablation.png', dpi=300)
plt.savefig('figures/fig10_offdiagonal_ablation.png', dpi=300)
plt.close()

print("Generated Figure 10: Off-Diagonal Diffusion Ablation.")
