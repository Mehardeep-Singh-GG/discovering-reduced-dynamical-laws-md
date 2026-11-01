#!/usr/bin/env python3
"""
r31_rank_scaling_v4.py - Milestone 31: Law 1 Intrinsic Operator Rank & Manifold Scaling.
Riemannian dPCA+ on Clifford torus T^d, short-time diffusion operator D(tau=1 ps) rank,
continuous Participation Ratio, and held-out out-of-scale power-law extrapolation across
the complete 15-System Macromolecular Atlas (d = 2 to 738).
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "version4_godtier", "lib"))
import core_v4 as C

RESULTS_DIR = os.path.join(ROOT, "version4_godtier", "results")
FIGS_DIR = os.path.join(ROOT, "version4_godtier", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)


def spectral_ranks(cov_matrix):
    cov_2d = np.atleast_2d(cov_matrix)
    evals = np.linalg.eigvalsh(cov_2d)
    evals = np.sort(np.maximum(evals, 0.0))[::-1]
    tot = float(np.sum(evals))
    if tot < 1e-12:
        return {"r80": 1, "r90": 1, "r95": 1, "r90_cont": 1.0, "pr": 1.0, "n_ent": 1.0, "evals_top10": []}

    p = evals / tot
    cum = np.cumsum(p)

    r80 = int(np.searchsorted(cum, 0.80 - 1e-12) + 1)
    r90 = int(np.searchsorted(cum, 0.90 - 1e-12) + 1)
    r95 = int(np.searchsorted(cum, 0.95 - 1e-12) + 1)

    k = r90
    s_prev = float(cum[k - 2]) if k > 1 else 0.0
    p_k = max(float(p[k - 1]), 1e-12)
    r90_cont = float((k - 1) + (0.90 - s_prev) / p_k)

    pr = float((tot ** 2) / np.sum(evals ** 2))
    p_pos = p[p > 1e-15]
    h_spec = -float(np.sum(p_pos * np.log(p_pos)))
    n_ent = float(np.exp(h_spec))

    return {
        "r80": r80,
        "r90": r90,
        "r95": r95,
        "r90_cont": round(r90_cont, 3),
        "pr": round(pr, 3),
        "n_ent": round(n_ent, 3),
        "evals_top10": [round(float(v), 6) for v in evals[:10]],
    }


def fit_power_law(x_vals, y_vals, seed=42):
    x = np.log(np.array(x_vals, dtype=np.float64))
    y = np.log(np.array(y_vals, dtype=np.float64))
    N = len(x)

    slope, intercept = np.polyfit(x, y, 1)
    c = float(np.exp(intercept))
    alpha = float(slope)

    y_pred = slope * x + intercept
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - y_pred) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)

    # Bootstrap 95% CI on exponent alpha across systems
    rng = np.random.default_rng(seed)
    boot_alphas = []
    if N >= 4:
        for _ in range(1000):
            idx = rng.integers(0, N, size=N)
            if len(np.unique(x[idx])) >= 2:
                s_b, _ = np.polyfit(x[idx], y[idx], 1)
                boot_alphas.append(float(s_b))
    ci_lo, ci_hi = (
        (float(np.percentile(boot_alphas, 2.5)), float(np.percentile(boot_alphas, 97.5)))
        if boot_alphas else (alpha, alpha)
    )

    # Null linear scaling: y = 1.0 * x + ln_c0
    ln_c0 = float(np.mean(y - 1.0 * x))
    c0 = float(np.exp(ln_c0))
    y_null = 1.0 * x + ln_c0
    ss_null = float(np.sum((y - y_null) ** 2))
    r2_null = 1.0 - (ss_null / ss_tot if ss_tot > 0 else 0.0)

    k_pow = 2
    aicc_pow = N * np.log(max(ss_res / N, 1e-12)) + 2 * k_pow + (2 * k_pow * (k_pow + 1)) / max(N - k_pow - 1, 1)
    k_null = 1
    aicc_null = N * np.log(max(ss_null / N, 1e-12)) + 2 * k_null + (2 * k_null * (k_null + 1)) / max(N - k_null - 1, 1)

    return {
        "n_systems": int(N),
        "alpha": round(alpha, 4),
        "alpha_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "c": round(c, 4),
        "r2": round(float(r2), 4),
        "aicc": round(float(aicc_pow), 2),
        "null_c0": round(c0, 4),
        "null_r2": round(float(r2_null), 4),
        "null_aicc": round(float(aicc_null), 2),
        "delta_aicc_over_null": round(float(aicc_pow - aicc_null), 2),
        "is_sublinear": bool(alpha < 1.0),
    }


def main():
    print("[M31] Starting Milestone 31: Law 1 Rank Scaling Across All 15 Atlas Systems...", flush=True)

    system_results = {}

    for sys_key in C.ORDER_V4:
        meta_sys = C.SYSTEMS_V4[sys_key]
        d_dim = int(meta_sys["d"])
        full_name = meta_sys["name"]
        print(f"  Processing {sys_key:10s} (d={d_dim:3d})...", flush=True)

        reps = C.load_atlas_replicas_v4(sys_key, verify_hash=True, max_frames_per_rep=20000)
        ang_list = [r["angles"] for r in reps]
        solvent_used = reps[0]["solvent"]

        # Learn maximal-gap chart cuts strictly on Rep 1 + Rep 2 (Discovery + Validation)
        centers_train = C.maxgap_shift(ang_list[:2])
        x_train_val, _ = C.dpca_plus_coords(ang_list[:2], offs=centers_train)
        x_test_rep3, _ = C.dpca_plus_coords([ang_list[2]], offs=centers_train)

        # Full pooled dPCA+ covariance & held-out Rep 3 covariance
        centers_all = C.maxgap_shift(ang_list)
        x_all_list, _ = C.dpca_plus_coords(ang_list, offs=centers_all)
        x_all = np.vstack(x_all_list)
        cov_dpca = np.atleast_2d(np.cov(x_all, rowvar=False))
        metrics_dpca = spectral_ranks(cov_dpca)

        cov_rep3_heldout = np.atleast_2d(np.cov(x_test_rep3[0], rowvar=False))
        metrics_rep3_heldout = spectral_ranks(cov_rep3_heldout)

        # Short-time dynamical diffusion tensor D(tau=1 ps) spectrum
        D_op = C.diffusion_tensor(ang_list, lag=1)
        metrics_diff_op = spectral_ranks(D_op)

        # Null diagonal model
        cov_null = np.diag(np.diag(cov_dpca))
        metrics_null = spectral_ranks(cov_null)

        # Sin-Cos 2d embedding representation
        all_raw = np.vstack(ang_list)
        sin_cos_coords = np.hstack([np.sin(all_raw), np.cos(all_raw)])
        cov_sincos = np.cov(sin_cos_coords, rowvar=False)
        metrics_sincos = spectral_ranks(cov_sincos)

        # Raw wrapped representation (demonstrating branch-cut artifact without dPCA+)
        cov_raw = np.atleast_2d(np.cov(all_raw, rowvar=False))
        metrics_raw = spectral_ranks(cov_raw)

        rep_ranks = {}
        for idx_r, xr in enumerate(x_all_list, start=1):
            rep_ranks[f"rep{idx_r}"] = spectral_ranks(np.atleast_2d(np.cov(xr, rowvar=False)))

        system_results[sys_key] = {
            "d": d_dim,
            "name": full_name,
            "tier": int(meta_sys["tier"]),
            "fold": meta_sys["fold"],
            "solvent": solvent_used,
            "cohort": "discovery_d_le_150" if d_dim <= 150 else "heldout_extrapolation_d_gt_150",
            "total_frames_analyzed": int(len(all_raw)),
            "dpca_plus": metrics_dpca,
            "dpca_plus_heldout_rep3": metrics_rep3_heldout,
            "diffusion_operator_tau1ps": metrics_diff_op,
            "null_diagonal": metrics_null,
            "sincos_dpca": metrics_sincos,
            "raw_wrapped": metrics_raw,
            "replicas": rep_ranks,
        }

    # 1. Full 15-System Atlas Scaling Laws (d = 2 -> 738)
    ordered_keys = [k for k in C.ORDER_V4 if k in system_results]
    dims = [system_results[k]["d"] for k in ordered_keys]
    r90_vals = [system_results[k]["dpca_plus"]["r90"] for k in ordered_keys]
    r90_cont_vals = [system_results[k]["dpca_plus"]["r90_cont"] for k in ordered_keys]
    pr_vals = [system_results[k]["dpca_plus"]["pr"] for k in ordered_keys]
    null_r90_vals = [system_results[k]["null_diagonal"]["r90"] for k in ordered_keys]
    diff_r90_vals = [system_results[k]["diffusion_operator_tau1ps"]["r90_cont"] for k in ordered_keys]

    fit_r90 = fit_power_law(dims, r90_vals)
    fit_r90_cont = fit_power_law(dims, r90_cont_vals)
    fit_pr = fit_power_law(dims, pr_vals)
    fit_null = fit_power_law(dims, null_r90_vals)
    fit_diff_op = fit_power_law(dims, diff_r90_vals)

    # 2. Strict Out-of-Scale Extrapolation: Fit on d <= 150 (10 systems), Predict d = 218..738 (5 unseen proteins)
    disc_keys = [k for k in ordered_keys if system_results[k]["d"] <= 150]
    extrap_keys = [k for k in ordered_keys if system_results[k]["d"] > 150]

    disc_dims = [system_results[k]["d"] for k in disc_keys]
    disc_r90c = [system_results[k]["dpca_plus"]["r90_cont"] for k in disc_keys]
    fit_disc = fit_power_law(disc_dims, disc_r90c)

    extrap_preds = {}
    extrap_ape = []
    null_extrap_ape = []
    for k in extrap_keys:
        d_k = system_results[k]["d"]
        obs = system_results[k]["dpca_plus"]["r90_cont"]
        pred_pow = fit_disc["c"] * (d_k ** fit_disc["alpha"])
        pred_lin = fit_disc["null_c0"] * d_k
        ape_pow = 100.0 * abs(pred_pow - obs) / obs
        ape_lin = 100.0 * abs(pred_lin - obs) / obs
        extrap_ape.append(ape_pow)
        null_extrap_ape.append(ape_lin)
        extrap_preds[k] = {
            "d": d_k,
            "observed_r90_cont": round(obs, 3),
            "predicted_sublinear_r90": round(float(pred_pow), 3),
            "predicted_linear_null_r90": round(float(pred_lin), 3),
            "ape_sublinear_pct": round(float(ape_pow), 2),
            "ape_linear_null_pct": round(float(ape_lin), 2),
        }

    # 3. 13-Protein Folded Atlas Only (excluding peptide dipeptides ala1, ala2: d = 18 -> 738)
    prot_keys = [k for k in C.PROTEINS_V4 if k in system_results]
    fit_proteins_only = fit_power_law(
        [system_results[k]["d"] for k in prot_keys],
        [system_results[k]["dpca_plus"]["r90_cont"] for k in prot_keys],
    )

    print("\n" + "=" * 72)
    print(f"LAW 1 RESULTS ({len(ordered_keys)}/15 Systems, d = {min(dims)} -> {max(dims)}):")
    print(f"  r90(d)      = {fit_r90['c']:.3f} * d^{fit_r90['alpha']:.3f}  (95% CI {fit_r90['alpha_ci95']}, R2={fit_r90['r2']:.3f}, dAICc={fit_r90['delta_aicc_over_null']:.2f})")
    print(f"  r90_cont(d) = {fit_r90_cont['c']:.3f} * d^{fit_r90_cont['alpha']:.3f}  (R2={fit_r90_cont['r2']:.3f}, dAICc={fit_r90_cont['delta_aicc_over_null']:.2f})")
    print(f"  PR(d)       = {fit_pr['c']:.3f} * d^{fit_pr['alpha']:.3f}  (R2={fit_pr['r2']:.3f})")
    print(f"  D_op r90(d) = {fit_diff_op['c']:.3f} * d^{fit_diff_op['alpha']:.3f}  (R2={fit_diff_op['r2']:.3f})")
    if extrap_ape:
        print(f"  Out-of-Scale Extrapolation (d=218..738): Sublinear MAPE = {np.mean(extrap_ape):.1f}% vs Linear Null MAPE = {np.mean(null_extrap_ape):.1f}%")
    print("=" * 72 + "\n")

    out_payload = {
        "milestone": "M31",
        "campaign": "version4_godtier",
        "n_systems_verified": len(ordered_keys),
        "description": "Law 1: Held-out Extrapolation of Sub-Linear Operator Rank Scaling (15 Systems, d = 2 -> 738)",
        "scaling_laws": {
            "r90_power_law": fit_r90,
            "r90_continuous_power_law": fit_r90_cont,
            "participation_ratio_power_law": fit_pr,
            "diffusion_operator_r90_power_law": fit_diff_op,
            "proteins_only_d18_to_738_power_law": fit_proteins_only,
            "null_diagonal_power_law": fit_null,
        },
        "out_of_scale_extrapolation": {
            "discovery_fit_d_le_150": fit_disc,
            "heldout_large_proteins_d_218_to_738": extrap_preds,
            "heldout_sublinear_mape_pct": round(float(np.mean(extrap_ape)), 2) if extrap_ape else None,
            "heldout_linear_null_mape_pct": round(float(np.mean(null_extrap_ape)), 2) if null_extrap_ape else None,
        },
        "systems": system_results,
    }

    out_json = os.path.join(RESULTS_DIR, "r31_rank_and_scaling_v4.json")
    with open(out_json, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[M31] Saved results JSON to {out_json}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), dpi=300)
    d_arr = np.logspace(np.log10(2.0), np.log10(850.0), 200)

    disc_d = [system_results[k]["d"] for k in disc_keys]
    disc_y = [system_results[k]["dpca_plus"]["r90"] for k in disc_keys]
    ext_d = [system_results[k]["d"] for k in extrap_keys]
    ext_y = [system_results[k]["dpca_plus"]["r90"] for k in extrap_keys]

    ax1.scatter(disc_d, disc_y, color="#0f766e", s=65, zorder=5, label="Discovery Cohort ($d \\leq 150$, 10 sys)")
    if ext_d:
        ax1.scatter(ext_d, ext_y, color="#dc2626", marker="*", s=130, zorder=6, label="Held-Out Extrapolation ($d = 218\\to 738$, 5 sys)")
    ax1.plot(d_arr, fit_r90["c"] * (d_arr ** fit_r90["alpha"]), color="#0f766e", lw=2.2,
             label=f"Sub-Linear Law: $r_{{90}} = {fit_r90['c']:.2f}\\, d^{{{fit_r90['alpha']:.2f}}}$ ($R^2={fit_r90['r2']:.3f}$)")
    ax1.plot(d_arr, fit_r90["null_c0"] * d_arr, color="#64748b", ls="--", lw=1.6,
             label=f"Linear Null $\\mathcal{{H}}_0$: $c_0 d^1$ ($\\Delta\\mathrm{{AIC}}_c = {fit_r90['delta_aicc_over_null']:.1f}$)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Backbone Dihedral Dimension $d$", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Intrinsic Manifold Rank $r_{90}$", fontsize=11, fontweight="bold")
    ax1.set_title("A. Sub-Linear Manifold Rank Across $d = 2 \\to 738$", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, fontsize=8.5, loc="upper left")
    ax1.grid(True, which="both", ls=":", alpha=0.5)

    ax2.scatter(dims, pr_vals, color="#1d4ed8", s=65, zorder=5, label="dPCA+ Participation Ratio $PR(d)$")
    ax2.scatter(dims, diff_r90_vals, color="#ea580c", marker="s", s=50, zorder=4, label="Diffusion Operator $\\mathbf{D}(\\tau)$ $r_{90}$")
    ax2.plot(d_arr, fit_pr["c"] * (d_arr ** fit_pr["alpha"]), color="#1d4ed8", lw=2.0,
             label=f"$PR = {fit_pr['c']:.2f}\\, d^{{{fit_pr['alpha']:.2f}}}$ ($R^2={fit_pr['r2']:.3f}$)")
    ax2.plot(d_arr, fit_diff_op["c"] * (d_arr ** fit_diff_op["alpha"]), color="#ea580c", ls="-.", lw=1.8,
             label=f"$r_{{90}}(\\mathbf{{D}}) = {fit_diff_op['c']:.2f}\\, d^{{{fit_diff_op['alpha']:.2f}}}$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Backbone Dihedral Dimension $d$", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Effective Rank / Participation Ratio", fontsize=11, fontweight="bold")
    ax2.set_title("B. Equilibrium vs Dynamical Operator Compression", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, fontsize=8.5, loc="upper left")
    ax2.grid(True, which="both", ls=":", alpha=0.5)

    plt.tight_layout()
    out_pdf = os.path.join(FIGS_DIR, "fig31_rank_scaling_extrapolation.pdf")
    out_png = os.path.join(FIGS_DIR, "fig31_rank_scaling_extrapolation.png")
    plt.savefig(out_pdf, dpi=300)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[M31] Saved publication figures to {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
