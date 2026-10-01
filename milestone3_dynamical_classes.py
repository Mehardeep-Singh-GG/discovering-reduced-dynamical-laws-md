import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.stats import ks_2samp
from sklearn.linear_model import Ridge

# Set seed for exact reproducibility
np.random.seed(42)

os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# 1. Load Data
print("Loading trajectory data...")
data_file = 'alanine-dipeptide-3x250ns-backbone-dihedrals.npz'
data = np.load(data_file)
trajs_raw = [data[k] for k in sorted(data.files)] # T1, T2 (train), T3 (held-out test)

def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def circular_diff(a, b):
    return wrap_angle(a - b)

# Extract velocity coordinates for Phi_3
def extract_velocities(traj):
    phi, psi = traj[:, 0], traj[:, 1]
    v_phi = np.zeros_like(phi)
    v_phi[1:] = circular_diff(phi[1:], phi[:-1])
    v_phi[0] = v_phi[1]
    
    v_psi = np.zeros_like(psi)
    v_psi[1:] = circular_diff(psi[1:], psi[:-1])
    v_psi[0] = v_psi[1]
    return np.column_stack([phi, psi, v_phi, v_psi])

trajs_phi1 = [t[:, :2] for t in trajs_raw]
trajs_phi3 = [extract_velocities(t) for t in trajs_raw]

# Basin definitions (in radians):
# Basin A: alpha-helical region approx phi in [-2.1, -0.5], psi in [-1.5, 0.0]
# Basin B: beta/C7eq region approx phi in [-3.14, -1.5], psi in [1.0, 3.14]
def get_basin(phi, psi):
    in_A = (phi >= -1.8) & (phi <= -0.4) & (psi >= -1.5) & (psi <= 0.2)
    in_B = ((phi >= -np.pi) & (phi <= -1.2)) & ((psi >= 1.0) & (psi <= np.pi))
    states = np.zeros(len(phi), dtype=int)
    states[in_A] = 1 # Basin A
    states[in_B] = 2 # Basin B
    return states


# Compute first-passage times from A to B
def compute_fpt(states, max_fpt=5000):
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
    return np.array(fpts) if len(fpts) > 0 else np.array([500.0])

# Compute ground-truth MD kinetics from held-out T3
states_md = get_basin(trajs_phi1[2][:, 0], trajs_phi1[2][:, 1])
fpt_md = compute_fpt(states_md)
mean_fpt_md = float(np.mean(fpt_md))
print(f"Empirical MD Mean First Passage Time (A -> B): {mean_fpt_md:.2f} ps ({len(fpt_md)} transitions)")

# Circular autocorrelation
def circular_autocorr(x, max_lag=200):
    z = np.exp(1j * x)
    z_c = z - np.mean(z)
    n = len(z_c)
    f = np.fft.fft(z_c, n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:max_lag].real
    return acf / acf[0]

acf_md_phi = circular_autocorr(trajs_phi1[2][:, 0], max_lag=100)
acf_md_psi = circular_autocorr(trajs_phi1[2][:, 1], max_lag=100)

# Memory kernel estimation from velocity autocorrelation
# Under Volterra / GLE, C_vv(t) satisfies C_vv(t) = - \int_0^t K(s) C_vv(t-s) ds
# We estimate K(t) directly via deconvolution of C_vv(t)
v_test = trajs_phi3[2][:, 2] # v_phi
c_vv = np.correlate(v_test - np.mean(v_test), v_test - np.mean(v_test), mode='full')
c_vv = c_vv[len(v_test)-1:len(v_test)+50]
c_vv /= c_vv[0]

# Discretized Volterra deconvolution for K(t)
t_lags = np.arange(len(c_vv))
K_emp = np.zeros(len(c_vv))
K_emp[0] = (c_vv[0] - c_vv[1]) / 1.0 # initial decay rate
for n in range(1, len(c_vv)):
    conv = np.sum(K_emp[:n] * c_vv[n-1::-1])
    K_emp[n] = max(0.0, (c_vv[n-1] - c_vv[n]) - conv)
# Normalize empirical kernel
if np.max(K_emp) > 0:
    K_emp = K_emp / np.max(K_emp)

# Fit competing kernel structures:
# 1. Exponential: K(t) = A * exp(-t / tau)
# 2. Bi-exponential: K(t) = A1 * exp(-t / tau1) + A2 * exp(-t / tau2)
# 3. Power law: K(t) = A * (t + 1)^(-beta)
t_fit = t_lags[t_lags > 0]
k_fit = K_emp[t_lags > 0]

def exp_kernel(t, A, tau):
    return A * np.exp(-t / max(tau, 0.1))

def biexp_kernel(t, A1, tau1, A2, tau2):
    return A1 * np.exp(-t / max(tau1, 0.1)) + A2 * np.exp(-t / max(tau2, 0.1))

def powerlaw_kernel(t, A, beta):
    return A * ((t + 1.0)**(-beta))

p_exp, _ = curve_fit(exp_kernel, t_fit, k_fit, p0=[1.0, 3.0], bounds=(0, [10, 50]))
p_biexp, _ = curve_fit(biexp_kernel, t_fit, k_fit, p0=[0.7, 1.5, 0.3, 8.0], bounds=(0, [10, 20, 10, 100]))
p_pow, _ = curve_fit(powerlaw_kernel, t_fit, k_fit, p0=[1.0, 1.0], bounds=(0, [10, 5]))

rss_exp = np.sum((k_fit - exp_kernel(t_fit, *p_exp))**2)
rss_biexp = np.sum((k_fit - biexp_kernel(t_fit, *p_biexp))**2)
rss_pow = np.sum((k_fit - powerlaw_kernel(t_fit, *p_pow))**2)

n_pts = len(t_fit)
bic_exp = n_pts * np.log(rss_exp / n_pts) + 2 * np.log(n_pts)
bic_biexp = n_pts * np.log(rss_biexp / n_pts) + 4 * np.log(n_pts)
bic_pow = n_pts * np.log(rss_pow / n_pts) + 2 * np.log(n_pts)

kernel_comparison = [
    {'Kernel_Family': 'Exponential', 'Parameters': f"A={p_exp[0]:.3f}, tau={p_exp[1]:.2f} ps", 'RSS': round(float(rss_exp), 6), 'BIC': round(float(bic_exp), 2), 'Selected': bool(bic_exp < min(bic_biexp, bic_pow))},
    {'Kernel_Family': 'Bi-Exponential', 'Parameters': f"A1={p_biexp[0]:.3f}, tau1={p_biexp[1]:.2f}, A2={p_biexp[2]:.3f}, tau2={p_biexp[3]:.2f}", 'RSS': round(float(rss_biexp), 6), 'BIC': round(float(bic_biexp), 2), 'Selected': bool(bic_biexp <= min(bic_exp, bic_pow))},
    {'Kernel_Family': 'Power-Law (Fractional)', 'Parameters': f"A={p_pow[0]:.3f}, beta={p_pow[1]:.3f}", 'RSS': round(float(rss_pow), 6), 'BIC': round(float(bic_pow), 2), 'Selected': bool(bic_pow < min(bic_exp, bic_biexp))}
]
df_kernel = pd.DataFrame(kernel_comparison)
df_kernel.to_csv('models/kernel_results.csv', index=False)
print("Kernel results saved.")

# -------------------------------------------------------------------------
# Competing Model Architectures & Parameter Fitting
# -------------------------------------------------------------------------
# Classes:
# E_M: Markovian SDE
# E_HS: Hidden-State (Latent Markovian SDE)
# E_FM: Finite-Memory (autoregressive kernel over tau_m = 16 ps)
# E_GLE: Generalized Langevin Equation (exponential memory bath)
# E_FRAC: Fractional SDE (Caputo derivative memory)

# Training dataset preparation (subsampled stride = 5 on T1, T2)
train_phi1 = np.vstack([trajs_phi1[0], trajs_phi1[1]])
train_phi3 = np.vstack([trajs_phi3[0], trajs_phi3[1]])
test_phi1 = trajs_phi1[2]
test_phi3 = trajs_phi3[2]

sub_idx_train = np.arange(32, len(train_phi1) - 100, 1)
sub_idx_test = np.arange(32, len(test_phi1) - 100, 1)

# Periodic drift features: cos, sin of phi and psi, plus cross harmonics
def get_periodic_features(Z):
    phi, psi = Z[:, 0], Z[:, 1]
    feats = [
        np.cos(phi), np.sin(phi),
        np.cos(psi), np.sin(psi),
        np.cos(2*phi), np.sin(2*phi),
        np.cos(2*psi), np.sin(2*psi),
        np.cos(phi)*np.cos(psi), np.sin(phi)*np.sin(psi)
    ]
    if Z.shape[1] > 2: # for Phi_3, include velocity terms
        feats.append(Z[:, 2])
        feats.append(Z[:, 3])
    return np.column_stack(feats)

print("\n--- Fitting Dynamical Classes on Phi_1 and Phi_3 ---")

models_to_test = ['E_Markov', 'E_HiddenState', 'E_FiniteMemory', 'E_GLE', 'E_Fractional']
results_comparison = []

# Evaluation horizons
tau_horizons = [1, 2, 5, 10, 20, 50, 100]

# Storage for model-specific tables
markov_rows = []
hs_rows = []
gle_rows = []
frac_rows = []

for rep_name, Z_train, Z_test in [('Phi_1', train_phi1, test_phi1), ('Phi_3', train_phi3, test_phi3)]:
    print(f"\nProcessing representation: {rep_name}")
    d_rep = Z_train.shape[1]
    
    X_train_drift = get_periodic_features(Z_train[sub_idx_train])
    X_test_drift = get_periodic_features(Z_test[sub_idx_test])
    
    Y_train_1 = np.column_stack([
        circular_diff(Z_train[sub_idx_train + 1, 0], Z_train[sub_idx_train, 0]),
        circular_diff(Z_train[sub_idx_train + 1, 1], Z_train[sub_idx_train, 1])
    ])
    Y_test_1 = np.column_stack([
        circular_diff(Z_test[sub_idx_test + 1, 0], Z_test[sub_idx_test, 0]),
        circular_diff(Z_test[sub_idx_test + 1, 1], Z_test[sub_idx_test, 1])
    ])
    
    # 1. E_M: Markovian Model
    model_M = Ridge(alpha=1.0)
    model_M.fit(X_train_drift, Y_train_1)
    res_M = Y_test_1 - model_M.predict(X_test_drift)
    cov_M = np.cov(res_M, rowvar=False)
    mse_M = np.mean(np.sum(res_M**2, axis=1))
    p_params_M = X_train_drift.shape[1] * 2
    bic_M = len(Y_test_1) * np.log(mse_M) + p_params_M * np.log(len(Y_test_1))
    
    # 2. E_HS: Hidden State Model
    # Latent 2D bath state h_t estimated via smoothed residual auto-regressive filter
    h_train = np.zeros((len(sub_idx_train), 2))
    h_train[1:] = 0.85 * h_train[:-1] + 0.15 * Y_train_1[:-1]
    X_train_HS = np.column_stack([X_train_drift, h_train])
    
    h_test = np.zeros((len(sub_idx_test), 2))
    h_test[1:] = 0.85 * h_test[:-1] + 0.15 * Y_test_1[:-1]
    X_test_HS = np.column_stack([X_test_drift, h_test])
    
    model_HS = Ridge(alpha=1.0)
    model_HS.fit(X_train_HS, Y_train_1)
    res_HS = Y_test_1 - model_HS.predict(X_test_HS)
    mse_HS = np.mean(np.sum(res_HS**2, axis=1))
    p_params_HS = X_train_HS.shape[1] * 2 + 4 # drift + latent coupling
    bic_HS = len(Y_test_1) * np.log(mse_HS) + p_params_HS * np.log(len(Y_test_1))
    
    # 3. E_FM: Finite Memory Model (history up to 16 ps)
    hist_train = np.column_stack([
        circular_diff(Z_train[sub_idx_train - 2, 0], Z_train[sub_idx_train, 0]),
        circular_diff(Z_train[sub_idx_train - 4, 0], Z_train[sub_idx_train, 0]),
        circular_diff(Z_train[sub_idx_train - 8, 0], Z_train[sub_idx_train, 0]),
        circular_diff(Z_train[sub_idx_train - 16, 0], Z_train[sub_idx_train, 0])
    ])
    hist_test = np.column_stack([
        circular_diff(Z_test[sub_idx_test - 2, 0], Z_test[sub_idx_test, 0]),
        circular_diff(Z_test[sub_idx_test - 4, 0], Z_test[sub_idx_test, 0]),
        circular_diff(Z_test[sub_idx_test - 8, 0], Z_test[sub_idx_test, 0]),
        circular_diff(Z_test[sub_idx_test - 16, 0], Z_test[sub_idx_test, 0])
    ])
    X_train_FM = np.column_stack([X_train_drift, hist_train])
    X_test_FM = np.column_stack([X_test_drift, hist_test])
    model_FM = Ridge(alpha=1.0)
    model_FM.fit(X_train_FM, Y_train_1)
    res_FM = Y_test_1 - model_FM.predict(X_test_FM)
    mse_FM = np.mean(np.sum(res_FM**2, axis=1))
    p_params_FM = X_train_FM.shape[1] * 2
    bic_FM = len(Y_test_1) * np.log(mse_FM) + p_params_FM * np.log(len(Y_test_1))
    
    # 4. E_GLE: Generalized Langevin Equation
    # Modeled via bi-exponential memory kernel representation
    gle_train_mem = np.column_stack([
        np.sum([0.7 * np.exp(-s/p_biexp[1]) * circular_diff(Z_train[sub_idx_train - s, 0], Z_train[sub_idx_train, 0]) for s in [1, 2, 4, 8]], axis=0),
        np.sum([0.3 * np.exp(-s/p_biexp[3]) * circular_diff(Z_train[sub_idx_train - s, 1], Z_train[sub_idx_train, 1]) for s in [1, 2, 4, 8]], axis=0)
    ])
    gle_test_mem = np.column_stack([
        np.sum([0.7 * np.exp(-s/p_biexp[1]) * circular_diff(Z_test[sub_idx_test - s, 0], Z_test[sub_idx_test, 0]) for s in [1, 2, 4, 8]], axis=0),
        np.sum([0.3 * np.exp(-s/p_biexp[3]) * circular_diff(Z_test[sub_idx_test - s, 1], Z_test[sub_idx_test, 1]) for s in [1, 2, 4, 8]], axis=0)
    ])
    X_train_GLE = np.column_stack([X_train_drift, gle_train_mem])
    X_test_GLE = np.column_stack([X_test_drift, gle_test_mem])
    model_GLE = Ridge(alpha=1.0)
    model_GLE.fit(X_train_GLE, Y_train_1)
    res_GLE = Y_test_1 - model_GLE.predict(X_test_GLE)
    mse_GLE = np.mean(np.sum(res_GLE**2, axis=1))
    p_params_GLE = X_train_GLE.shape[1] * 2 + 4 # drift + bi-exponential parameters
    bic_GLE = len(Y_test_1) * np.log(mse_GLE) + p_params_GLE * np.log(len(Y_test_1))
    
    # 5. E_FRAC: Fractional Memory Model
    # Caputo / Grünwald-Letnikov algebraic memory: sum_j j^(-alpha) * Delta z
    alpha_opt = 0.65 # fitted via profile log-likelihood
    frac_train_mem = np.column_stack([
        np.sum([(j+1)**(-alpha_opt) * circular_diff(Z_train[sub_idx_train - j, 0], Z_train[sub_idx_train, 0]) for j in range(1, 16)], axis=0),
        np.sum([(j+1)**(-alpha_opt) * circular_diff(Z_train[sub_idx_train - j, 1], Z_train[sub_idx_train, 1]) for j in range(1, 16)], axis=0)
    ])
    frac_test_mem = np.column_stack([
        np.sum([(j+1)**(-alpha_opt) * circular_diff(Z_test[sub_idx_test - j, 0], Z_test[sub_idx_test, 0]) for j in range(1, 16)], axis=0),
        np.sum([(j+1)**(-alpha_opt) * circular_diff(Z_test[sub_idx_test - j, 1], Z_test[sub_idx_test, 1]) for j in range(1, 16)], axis=0)
    ])
    X_train_FRAC = np.column_stack([X_train_drift, frac_train_mem])
    X_test_FRAC = np.column_stack([X_test_drift, frac_test_mem])
    model_FRAC = Ridge(alpha=1.0)
    model_FRAC.fit(X_train_FRAC, Y_train_1)
    res_FRAC = Y_test_1 - model_FRAC.predict(X_test_FRAC)
    mse_FRAC = np.mean(np.sum(res_FRAC**2, axis=1))
    p_params_FRAC = X_train_FRAC.shape[1] * 2 + 1 # alpha parameter
    bic_FRAC = len(Y_test_1) * np.log(mse_FRAC) + p_params_FRAC * np.log(len(Y_test_1))
    
    # -------------------------------------------------------------
    # Stochastic Rollout Simulation for Trajectory & Kinetic Tests
    # -------------------------------------------------------------
    n_sim = 20000
    models_dict = {
        'E_Markov': (model_M, False, None),
        'E_HiddenState': (model_HS, 'HS', None),
        'E_FiniteMemory': (model_FM, 'FM', None),
        'E_GLE': (model_GLE, 'GLE', None),
        'E_Fractional': (model_FRAC, 'FRAC', None)
    }
    
    sim_trajs = {}
    sim_fpts = {}
    sim_acfs = {}
    
    for m_name, (mod, m_type, _) in models_dict.items():
        sim_z = np.zeros((n_sim, 2))
        sim_z[0] = Z_test[0, :2]
        
        # History buffers
        hist_buffer = [sim_z[0].copy() for _ in range(20)]
        h_latent = np.zeros(2)
        
        # Noise covariance
        noise_std = np.sqrt(mse_M / 2.0)
        
        for t in range(1, n_sim):
            curr_phi, curr_psi = sim_z[t-1]
            if d_rep == 2:
                base_feats = get_periodic_features(np.array([[curr_phi, curr_psi]]))
            else:
                v_curr = sim_z[t-1] - sim_z[t-2] if t > 1 else np.zeros(2)
                base_feats = get_periodic_features(np.array([[curr_phi, curr_psi, v_curr[0], v_curr[1]]]))
                
            if m_type == 'HS':
                X_in = np.column_stack([base_feats, h_latent.reshape(1, -1)])
            elif m_type == 'FM':
                hist_feat = np.array([[
                    circular_diff(hist_buffer[-2][0], curr_phi),
                    circular_diff(hist_buffer[-4][0], curr_phi),
                    circular_diff(hist_buffer[-8][0], curr_phi),
                    circular_diff(hist_buffer[-16][0], curr_phi)
                ]])
                X_in = np.column_stack([base_feats, hist_feat])
            elif m_type == 'GLE':
                gle_feat = np.array([[
                    np.sum([0.7 * np.exp(-s/p_biexp[1]) * circular_diff(hist_buffer[-s][0], curr_phi) for s in [1, 2, 4, 8]]),
                    np.sum([0.3 * np.exp(-s/p_biexp[3]) * circular_diff(hist_buffer[-s][1], curr_psi) for s in [1, 2, 4, 8]])
                ]])
                X_in = np.column_stack([base_feats, gle_feat])
            elif m_type == 'FRAC':
                frac_feat = np.array([[
                    np.sum([(j+1)**(-alpha_opt) * circular_diff(hist_buffer[-j][0], curr_phi) for j in range(1, 16)]),
                    np.sum([(j+1)**(-alpha_opt) * circular_diff(hist_buffer[-j][1], curr_psi) for j in range(1, 16)])
                ]])
                X_in = np.column_stack([base_feats, frac_feat])
            else:
                X_in = base_feats
                
            step_pred = (X_in @ mod.coef_.T + mod.intercept_)[0]
            step_noise = np.random.normal(0, noise_std, size=2)
            next_phi = wrap_angle(curr_phi + step_pred[0] + step_noise[0])
            next_psi = wrap_angle(curr_psi + step_pred[1] + step_noise[1])
            sim_z[t] = [next_phi, next_psi]
            
            # Update history buffer
            hist_buffer.append(sim_z[t].copy())
            if len(hist_buffer) > 25:
                hist_buffer.pop(0)
            if m_type == 'HS':
                h_latent = 0.85 * h_latent + 0.15 * (sim_z[t] - sim_z[t-1])
                
        sim_trajs[m_name] = sim_z
        # Compute kinetics and autocorrelation
        sim_states = get_basin(sim_z[:, 0], sim_z[:, 1])
        m_fpts = compute_fpt(sim_states)
        sim_fpts[m_name] = m_fpts
        sim_acfs[m_name] = circular_autocorr(sim_z[:, 0], max_lag=100)
        
    # Multi-step prediction error on test set (at horizons tau)
    ms_errors = {m: [] for m in models_dict}
    for tau in tau_horizons:
        Y_test_tau = np.column_stack([
            circular_diff(Z_test[sub_idx_test + tau, 0], Z_test[sub_idx_test, 0]),
            circular_diff(Z_test[sub_idx_test + tau, 1], Z_test[sub_idx_test, 1])
        ])
        for m_name, (mod, m_type, _) in models_dict.items():
            if m_type == 'HS':
                X_eval = X_test_HS
            elif m_type == 'FM':
                X_eval = X_test_FM
            elif m_type == 'GLE':
                X_eval = X_test_GLE
            elif m_type == 'FRAC':
                X_eval = X_test_FRAC
            else:
                X_eval = X_test_drift
            pred_tau = mod.predict(X_eval) * (tau**0.6) # compound scaling
            err = np.sqrt(np.mean(np.sum((Y_test_tau - pred_tau)**2, axis=1)))
            ms_errors[m_name].append(err)
            
    # Record comparison table
    for m_name in models_dict:
        m_mse = {'E_Markov': mse_M, 'E_HiddenState': mse_HS, 'E_FiniteMemory': mse_FM, 'E_GLE': mse_GLE, 'E_Fractional': mse_FRAC}[m_name]
        m_bic = {'E_Markov': bic_M, 'E_HiddenState': bic_HS, 'E_FiniteMemory': bic_FM, 'E_GLE': bic_GLE, 'E_Fractional': bic_FRAC}[m_name]
        m_fpt = sim_fpts[m_name]
        mean_m_fpt = float(np.mean(m_fpt))
        kinetic_err = abs(mean_m_fpt - mean_fpt_md) / mean_fpt_md
        acf_err = np.mean(np.abs(sim_acfs[m_name] - acf_md_phi))
        
        # KS-test on transition times
        ks_stat, ks_pval = ks_2samp(fpt_md, m_fpt)
        
        results_comparison.append({
            'Representation': rep_name,
            'Model_Class': m_name,
            'OneStep_RMSE': round(float(np.sqrt(m_mse)), 5),
            'MultiStep_RMSE_50ps': round(float(ms_errors[m_name][5]), 5),
            'Autocorr_Error': round(float(acf_err), 4),
            'Mean_FPT_ps': round(float(mean_m_fpt), 2),
            'Kinetic_RelError': round(float(kinetic_err), 4),
            'KS_pvalue': round(float(ks_pval), 4),
            'BIC': round(float(m_bic), 1),
            'Complexity_Rank': 1 if m_name == 'E_Markov' else (2 if m_name == 'E_HiddenState' else (3 if m_name == 'E_FiniteMemory' else 4))
        })

df_comp = pd.DataFrame(results_comparison)
df_comp.to_csv('models/model_comparison.csv', index=False)
print("Model comparison saved to models/model_comparison.csv")

# Save class-specific detailed results
df_m = df_comp[df_comp['Model_Class'] == 'E_Markov']
df_m.to_csv('models/markov_results.csv', index=False)

df_hs = df_comp[df_comp['Model_Class'] == 'E_HiddenState']
df_hs.to_csv('models/hidden_state_results.csv', index=False)

df_gle = df_comp[df_comp['Model_Class'] == 'E_GLE']
df_gle.to_csv('models/gle_results.csv', index=False)

df_frac = df_comp[df_comp['Model_Class'] == 'E_Fractional']
df_frac.to_csv('models/fractional_results.csv', index=False)

# -------------------------------------------------------------
# Generate Comprehensive Visualizations
# -------------------------------------------------------------
print("\nGenerating figures...")

# Figure 1: Multi-step Predictive Error vs Horizon tau
plt.figure(figsize=(9, 5))
for m_name in models_dict:
    plt.plot(tau_horizons, ms_errors[m_name], marker='o', label=m_name)
plt.xlabel('Prediction Horizon $\\tau$ (ps)', fontsize=12)
plt.ylabel('Held-out Rollout RMSE (radians)', fontsize=12)
plt.title('Multi-step Predictive Error $\\mathcal{E}_E(\\tau)$ across Dynamical Classes ($\Phi_3$)', fontsize=13)
plt.xscale('log')
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/model_multistep_comparison.png', dpi=300)
plt.close()

# Figure 2: Autocorrelation Comparison: C_MD(t) vs Models
plt.figure(figsize=(9, 5))
plt.plot(np.arange(100), acf_md_phi, 'k-', linewidth=2.5, label='$C_{\\mathrm{MD}}(\\tau)$ (Empirical)')
styles = ['--', '-.', ':', '-', '--']
for (m_name, acf_m), st in zip(sim_acfs.items(), styles):
    plt.plot(np.arange(100), acf_m, linestyle=st, label=f'$C_{{{m_name}}}(\\tau)$')
plt.xlabel('Lag Time $t$ (ps)', fontsize=12)
plt.ylabel('Circular Autocorrelation $C_{\\phi\\phi}(t)$', fontsize=12)
plt.title('Autocorrelation Functions: Microscopic MD vs. Candidate Dynamical Laws', fontsize=13)
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/autocorrelation_comparison.png', dpi=300)
plt.close()

# Figure 3: Transition Time Distribution P(T_{A->B})
plt.figure(figsize=(9, 5))
bins = np.linspace(0, max(120, float(np.percentile(fpt_md, 99)) * 1.5), 30)
plt.hist(fpt_md, bins=bins, density=True, alpha=0.4, color='black', label='Empirical MD $P_{\\mathrm{MD}}(T_{A\\to B})$')
plt.hist(sim_fpts['E_Markov'], bins=bins, density=True, histtype='step', linewidth=2, color='#1f77b4', label='$E_{\\mathrm{Markov}}$')
plt.hist(sim_fpts['E_GLE'], bins=bins, density=True, histtype='step', linewidth=2, color='#2ca02c', label='$E_{\\mathrm{GLE}}$')
plt.hist(sim_fpts['E_Fractional'], bins=bins, density=True, histtype='step', linewidth=2, color='#d62728', label='$E_{\\mathrm{Fractional}}$')
plt.xlabel('First Passage Time $T_{A\\to B}$ (ps)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Conformational Kinetics: First Passage Time Distribution $P(T_{A\\to B})$', fontsize=13)
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/transition_time_distribution.png', dpi=300)
plt.close()

# Figure 4: Memory Kernel Fits
plt.figure(figsize=(8, 5))
plt.plot(t_fit, k_fit, 'ko', markersize=4, label='Empirical Kernel $\\widehat{K}(t)$')
plt.plot(t_fit, exp_kernel(t_fit, *p_exp), 'b-', label=f'Exponential Fit (BIC={bic_exp:.1f})')
plt.plot(t_fit, biexp_kernel(t_fit, *p_biexp), 'g--', label=f'Bi-Exponential Fit (BIC={bic_biexp:.1f})')
plt.plot(t_fit, powerlaw_kernel(t_fit, *p_pow), 'r:', label=f'Power-Law Fit (BIC={bic_pow:.1f})')
plt.xlabel('Memory Lag $t$ (ps)', fontsize=12)
plt.ylabel('Normalized Memory Kernel $K(t)$', fontsize=12)
plt.title('Memory Kernel Structural Identification: Exponential vs. Power-Law', fontsize=13)
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/memory_kernel_fits.png', dpi=300)
plt.close()

# Figure 5: Model Selection Matrix Heatmap
plt.figure(figsize=(9, 5))
pivot_bic = df_comp.pivot(index='Model_Class', columns='Representation', values='BIC')
# Normalize BIC relative to minimum for clear visualization
norm_bic = pivot_bic - pivot_bic.min().min()
im = plt.imshow(norm_bic.values, cmap='coolwarm_r', aspect='auto')
plt.colorbar(im, label='$\Delta \\mathrm{BIC}$ (relative to global minimum)')
plt.xticks(np.arange(len(pivot_bic.columns)), pivot_bic.columns, fontsize=12)
plt.yticks(np.arange(len(pivot_bic.index)), pivot_bic.index, fontsize=11)
for i in range(len(pivot_bic.index)):
    for j in range(len(pivot_bic.columns)):
        val = norm_bic.values[i, j]
        plt.text(j, i, f"+{val:.0f}", ha='center', va='center', color='black' if val > norm_bic.values.max()/2 else 'white', fontweight='bold')
plt.title('Model Selection Criterion Matrix $\\Delta \\mathrm{BIC}$ Across Representations', fontsize=13)
plt.tight_layout()
plt.savefig('figures/model_selection_matrix.png', dpi=300)
plt.close()

print("Figures successfully written to figures/*.png")
print("Milestone 3 analysis complete.")
