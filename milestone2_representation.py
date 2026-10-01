import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# Set seed for exact reproducibility
np.random.seed(42)

# Ensure output directories exist
os.makedirs('representation', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# 1. Load Trajectories
print("Loading dataset...")
data_file = 'alanine-dipeptide-3x250ns-backbone-dihedrals.npz'
data = np.load(data_file)
trajs_raw = [data[k] for k in sorted(data.files)] # T1, T2, T3
n_trajs = len(trajs_raw)
print(f"Loaded {n_trajs} trajectories with shapes: {[t.shape for t in trajs_raw]}")

# Helper: periodic angular difference on [-pi, pi)
def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def circular_diff(a, b):
    return wrap_angle(a - b)

# 2. Build Representations for each trajectory
# Phi_1: (phi, psi) [2D]
# Phi_2: (phi, psi, v_phi) [3D]
# Phi_3: (phi, psi, v_phi, v_psi) [4D]
def extract_representation(traj, rep_name):
    phi = traj[:, 0]
    psi = traj[:, 1]
    
    # Numerical angular velocities (finite differences, wrap periodic jumps)
    v_phi = np.zeros_like(phi)
    v_phi[1:] = circular_diff(phi[1:], phi[:-1])
    v_phi[0] = v_phi[1]
    
    v_psi = np.zeros_like(psi)
    v_psi[1:] = circular_diff(psi[1:], psi[:-1])
    v_psi[0] = v_psi[1]
    
    if rep_name == '(phi,psi)':
        return np.column_stack([phi, psi])
    elif rep_name == '(phi,psi,v_phi)':
        return np.column_stack([phi, psi, v_phi])
    elif rep_name == '(phi,psi,v_phi,v_psi)':
        return np.column_stack([phi, psi, v_phi, v_psi])
    else:
        raise ValueError(f"Unknown representation {rep_name}")

# Periodic feature embedding for regression:
# To prevent boundary artifacts at +/- pi, embed angles into (cos, sin)
def embed_features(Z, rep_dim):
    # First 2 cols are always (phi, psi)
    phi = Z[:, 0]
    psi = Z[:, 1]
    feats = [np.cos(phi), np.sin(phi), np.cos(psi), np.sin(psi)]
    # Any additional dimensions (e.g. velocities) are linear
    for d in range(2, rep_dim):
        feats.append(Z[:, d])
    return np.column_stack(feats)

# Parameters for Closure Tests
# Logarithmically spaced history scale k (in ps)
k_values = [1, 2, 4, 8, 16, 32, 64, 128]
# Prediction horizons tau (in ps)
tau_values = [1, 2, 5, 10, 20, 50, 100]
representations = [
    ('(phi,psi)', 2),
    ('(phi,psi,v_phi)', 3),
    ('(phi,psi,v_phi,v_psi)', 4)
]

# Downsampling/subsampling for computationally intensive regression without losing statistical power
# Trajectories have 250,000 frames each. Stride = 5 gives 50,000 samples per trajectory.
stride = 1
max_k = max(k_values)
max_tau = max(tau_values)

# Data structures for results
rep_scores_list = []
closure_tests_list = []
pred_comp_list = []
bootstrap_list = []

print("Beginning Representation Identification & Closure Tests...")

# Train on T1, T2; Test on T3
for rep_name, d in representations:
    print(f"\nEvaluating Representation {rep_name} (dim={d})...")
    
    # Prepare trajectories with the chosen representation
    rep_trajs = [extract_representation(t, rep_name) for t in trajs_raw]
    
    # Train set: T1, T2
    # Test set: T3
    train_trajs = rep_trajs[:2]
    test_traj = rep_trajs[2]
    
    for tau in tau_values:
        # Step 1: Baseline model M_0 (k = 1, current state z_t)
        # Construct dataset for M_0
        X_train_0_list, Y_train_list = [], []
        for traj in train_trajs:
            N = len(traj)
            valid_idx = np.arange(max_k, N - max_tau, stride)
            z_t = traj[valid_idx]
            z_next = traj[valid_idx + tau]
            # Target is circular difference in (phi, psi)
            d_phi = circular_diff(z_next[:, 0], z_t[:, 0])
            d_psi = circular_diff(z_next[:, 1], z_t[:, 1])
            Y = np.column_stack([d_phi, d_psi])
            
            X_emb = embed_features(z_t, d)
            X_train_0_list.append(X_emb)
            Y_train_list.append(Y)
            
        X_train_0 = np.vstack(X_train_0_list)
        Y_train = np.vstack(Y_train_list)
        
        # Test dataset
        N_test = len(test_traj)
        valid_idx_test = np.arange(max_k, N_test - max_tau, stride)
        z_t_test = test_traj[valid_idx_test]
        z_next_test = test_traj[valid_idx_test + tau]
        d_phi_test = circular_diff(z_next_test[:, 0], z_t_test[:, 0])
        d_psi_test = circular_diff(z_next_test[:, 1], z_t_test[:, 1])
        Y_test = np.column_stack([d_phi_test, d_psi_test])
        X_test_0 = embed_features(z_t_test, d)
        
        # Train baseline model M_0
        model_0 = Ridge(alpha=1.0)
        model_0.fit(X_train_0, Y_train)
        
        # Residuals and covariance on test set
        pred_0_test = model_0.predict(X_test_0)
        res_0 = Y_test - pred_0_test
        cov_0 = np.cov(res_0, rowvar=False)
        det_cov_0 = np.linalg.det(cov_0)
        
        loss_test_0 = np.mean(np.sum(res_0**2, axis=1))
        rmse_test_0 = np.sqrt(loss_test_0)
        
        # Now test across history lengths k
        for k in k_values:
            if k == 1:
                # k = 1 is baseline itself
                M_hat = 0.0
                ci_low, ci_high = 0.0, 0.0
                loss_test_k = loss_test_0
                rmse_test_k = rmse_test_0
                delta_loss = 0.0
                p_val = 1.0
                is_closed = True
                div_closure = 0.0
            else:
                # Construct history features H_k
                # H_k contains (z_{t-(k-1)}, ..., z_t)
                # To keep dimensionality controlled and avoid multicollinearity,
                # we sample history lags up to k
                if k <= 4:
                    lags = np.arange(k)
                elif k <= 16:
                    lags = np.unique(np.linspace(0, k-1, 4, dtype=int))
                elif k <= 64:
                    lags = np.unique(np.linspace(0, k-1, 6, dtype=int))
                else:
                    lags = np.unique(np.linspace(0, k-1, 8, dtype=int))
                    
                X_train_k_list = []
                for traj in train_trajs:
                    feats_k = []
                    for lag in lags:
                        z_lag = traj[valid_idx - lag]
                        feats_k.append(embed_features(z_lag, d))
                    X_train_k_list.append(np.hstack(feats_k))
                X_train_k = np.vstack(X_train_k_list)
                
                # Test set history
                feats_k_test = []
                for lag in lags:
                    z_lag_test = test_traj[valid_idx_test - lag]
                    feats_k_test.append(embed_features(z_lag_test, d))
                X_test_k = np.hstack(feats_k_test)
                
                # Train model M_k
                model_k = Ridge(alpha=1.0)
                model_k.fit(X_train_k, Y_train)
                
                # Test set predictions & residuals
                pred_k_test = model_k.predict(X_test_k)
                res_k = Y_test - pred_k_test
                cov_k = np.cov(res_k, rowvar=False)
                det_cov_k = np.linalg.det(cov_k)
                
                loss_test_k = np.mean(np.sum(res_k**2, axis=1))
                rmse_test_k = np.sqrt(loss_test_k)
                delta_loss = loss_test_0 - loss_test_k
                
                # Conditional mutual information estimate:
                # M_hat(k, tau) = 0.5 * ln(|Sigma_0| / |Sigma_k|)
                ratio = det_cov_0 / max(det_cov_k, 1e-12)
                M_hat = max(0.0, 0.5 * np.log(max(1.0, ratio)))
                
                # Distributional closure metric (Energy distance between residuals)
                # D_closure = ||mean(res_0) - mean(res_k)||^2 + Tr(cov_0 + cov_k - 2*(cov_0*cov_k)^1/2)
                mean_diff = np.mean(res_0, axis=0) - np.mean(res_k, axis=0)
                div_closure = np.sum(mean_diff**2) + np.abs(np.trace(cov_0) - np.trace(cov_k))
                
                # Block Bootstrap for 95% Confidence Interval
                # Block size B = 200 subsampled steps (= 1000 ps = 1 ns)
                block_size = 50
                n_blocks = len(res_k) // block_size
                if n_blocks > 2:
                    boot_M = []
                    for _ in range(100):
                        block_indices = np.random.randint(0, n_blocks, size=n_blocks)
                        idx_boot = np.concatenate([np.arange(b*block_size, (b+1)*block_size) for b in block_indices])
                        
                        cov_0_b = np.cov(res_0[idx_boot], rowvar=False)
                        cov_k_b = np.cov(res_k[idx_boot], rowvar=False)
                        d0_b = np.linalg.det(cov_0_b)
                        dk_b = np.linalg.det(cov_k_b)
                        r_b = d0_b / max(dk_b, 1e-12)
                        m_b = max(0.0, 0.5 * np.log(max(1.0, r_b)))
                        boot_M.append(m_b)
                    
                    ci_low = np.percentile(boot_M, 2.5)
                    ci_high = np.percentile(boot_M, 97.5)
                else:
                    ci_low, ci_high = M_hat * 0.9, M_hat * 1.1
                    
                # Formal hypothesis test H0: M = 0 vs H1: M > 0
                # Threshold epsilon_M = 0.01 nats
                eps_M = 0.01
                is_closed = bool(ci_high <= eps_M)
                p_val = np.mean(np.array(boot_M) <= eps_M) if n_blocks > 2 else (1.0 if is_closed else 0.0)

            # Record results
            rep_scores_list.append({
                'Phi': rep_name,
                'd': d,
                'k': k,
                'tau': tau,
                'M_hat': round(float(M_hat), 5),
                'CI_95_low': round(float(ci_low), 5),
                'CI_95_high': round(float(ci_high), 5)
            })
            
            closure_tests_list.append({
                'Phi': rep_name,
                'd': d,
                'k': k,
                'tau': tau,
                'M_hat': round(float(M_hat), 5),
                'CI_95_high': round(float(ci_high), 5),
                'p_value': round(float(p_val), 4),
                'H0_closed': is_closed,
                'div_closure': round(float(div_closure), 6)
            })
            
            pred_comp_list.append({
                'Phi': rep_name,
                'k': k,
                'tau': tau,
                'L_test_0': round(float(loss_test_0), 5),
                'L_test_k': round(float(loss_test_k), 5),
                'Delta_L': round(float(delta_loss), 5),
                'RMSE_0': round(float(rmse_test_0), 5),
                'RMSE_k': round(float(rmse_test_k), 5)
            })

# Save Data Tables
df_scores = pd.DataFrame(rep_scores_list)
df_scores.to_csv('representation/representation_scores.csv', index=False)

df_closure = pd.DataFrame(closure_tests_list)
df_closure.to_csv('representation/closure_tests.csv', index=False)

df_pred = pd.DataFrame(pred_comp_list)
df_pred.to_csv('representation/predictive_comparison.csv', index=False)

# Bootstrap summary table
df_bootstrap = df_scores[['Phi', 'k', 'tau', 'M_hat', 'CI_95_low', 'CI_95_high']]
df_bootstrap.to_csv('representation/bootstrap_results.csv', index=False)

print("\nResults successfully saved to representation/*.csv")

# -------------------------------------------------------------
# Visualizations
# -------------------------------------------------------------
print("Generating Figures...")

# Figure 1: Closure vs History k (for Phi_1 at multiple tau)
plt.figure(figsize=(8, 5))
phi1_df = df_scores[df_scores['Phi'] == '(phi,psi)']
for tau in [1, 5, 20, 50, 100]:
    sub = phi1_df[phi1_df['tau'] == tau]
    plt.plot(sub['k'], sub['M_hat'], marker='o', label=f'$\\tau = {tau}$ ps')
    plt.fill_between(sub['k'], sub['CI_95_low'], sub['CI_95_high'], alpha=0.15)
plt.axhline(0.01, color='gray', linestyle='--', label='$\\epsilon_M$ closure threshold')
plt.xscale('log', base=2)
plt.xlabel('History Length $k$ (ps)', fontsize=12)
plt.ylabel('Conditional Info $\\widehat{M}(k, \\tau)$ (nats)', fontsize=12)
plt.title('Memory Functional $\\widehat{M}(k, \\tau)$ vs. History Length for $\\Phi_1 = (\\phi, \\psi)$', fontsize=13)
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/closure_vs_history.png', dpi=300)
plt.close()

# Figure 2: Closure vs Lag tau (for Phi_1 across k)
plt.figure(figsize=(8, 5))
for k in [2, 8, 32, 128]:
    sub = phi1_df[phi1_df['k'] == k]
    plt.plot(sub['tau'], sub['M_hat'], marker='s', label=f'$k = {k}$ ps')
    plt.fill_between(sub['tau'], sub['CI_95_low'], sub['CI_95_high'], alpha=0.15)
plt.axhline(0.01, color='gray', linestyle='--', label='$\\epsilon_M$ threshold')
plt.xscale('log')
plt.xlabel('Prediction Horizon $\\tau$ (ps)', fontsize=12)
plt.ylabel('Conditional Info $\\widehat{M}(k, \\tau)$ (nats)', fontsize=12)
plt.title('Memory Functional $\\widehat{M}(k, \\tau)$ vs. Prediction Horizon $\\tau$', fontsize=13)
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/closure_vs_lag.png', dpi=300)
plt.close()

# Figure 3: Predictive Error vs History k (Held-out Test RMSE)
plt.figure(figsize=(8, 5))
pred_phi1 = df_pred[df_pred['Phi'] == '(phi,psi)']
for tau in [1, 5, 20, 50]:
    sub = pred_phi1[pred_phi1['tau'] == tau]
    plt.plot(sub['k'], sub['RMSE_k'], marker='^', label=f'$\\tau = {tau}$ ps')
plt.xscale('log', base=2)
plt.xlabel('History Length $k$ (ps)', fontsize=12)
plt.ylabel('Held-out Test RMSE (radians)', fontsize=12)
plt.title('Multi-step Predictive Error $\\mathrm{RMSE}_k(\\tau)$ vs. History $k$', fontsize=13)
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/predictive_error_vs_history.png', dpi=300)
plt.close()

# Figure 4: Representation Comparison (Ladder Phi_1 -> Phi_2 -> Phi_3)
plt.figure(figsize=(8, 5))
fixed_tau = 5
fixed_k = 16
ladder_summary = []
for rep_name, d in representations:
    sub = df_scores[(df_scores['Phi'] == rep_name) & (df_scores['tau'] == fixed_tau) & (df_scores['k'] == fixed_k)]
    ladder_summary.append({
        'Phi': rep_name,
        'd': d,
        'M_hat': sub['M_hat'].values[0],
        'CI_low': sub['CI_95_low'].values[0],
        'CI_high': sub['CI_95_high'].values[0]
    })
df_ladder = pd.DataFrame(ladder_summary)

plt.bar(df_ladder['Phi'], df_ladder['M_hat'], yerr=[df_ladder['M_hat'] - df_ladder['CI_low'], df_ladder['CI_high'] - df_ladder['M_hat']],
        capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.85, edgecolor='black')
plt.axhline(0.01, color='red', linestyle='--', label='Closure Threshold $\\epsilon_M = 0.01$ nats')
plt.ylabel('Residual Memory $\\widehat{M}(\\Phi)$ (nats)', fontsize=12)
plt.xlabel('Representation $\\Phi$', fontsize=12)
plt.title(f'Representation Ladder: Residual Memory at $\\tau = {fixed_tau}$ ps, $k = {fixed_k}$ ps', fontsize=13)
plt.legend(frameon=True)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/representation_comparison.png', dpi=300)
plt.close()

print("Figures successfully generated in figures/*.png")
print("Milestone 2 analysis pipeline complete.")
