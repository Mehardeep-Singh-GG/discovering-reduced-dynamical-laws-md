#!/usr/bin/env python3
"""
r32_sequence_contact_coupling_v4.py - Milestone 32: Law 2 Sequence Decay, Tertiary Contact Gating, & Operator Sparsity.
Evaluates across the complete 15-System Macromolecular Atlas (d = 2 -> 738) using exact
covalent chain ordering and exact peptide-plane vs intra-residue dihedral indexing from core_v4:
  1. Short-time increment correlation R_ab(tau=1 ps) and stationary equilibrium dPCA+ correlation C^eq_ab.
  2. Universal crankshaft peptide-plane counter-rotation: exact R(psi_r, phi_{r+1}) vs intra-residue R(phi_r, psi_r).
  3. Sequence decay law competition (Exponential vs Algebraic Power-Law) along exact chain_order with AICc.
  4. Non-local tertiary contact gating (|i - j| >= 4): partial Spearman controlling for sequence separation.
  5. Bounded-degree sparsity law: S_eps(d) = 1 - k_eff / (d - 1) from d = 18 to d = 738.
"""
import os
import sys
import json
import numpy as np
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "version4_godtier", "lib"))
import core_v4 as C

RESULTS_DIR = os.path.join(ROOT, "version4_godtier", "results")
FIGS_DIR = os.path.join(ROOT, "version4_godtier", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)


def compute_increment_correlation(ang_list, tau=1):
    incs = [C.increments(traj, lag=tau) for traj in ang_list if len(traj) > tau]
    X = np.concatenate(incs, axis=0)
    X_cent = X - np.mean(X, axis=0, keepdims=True)
    cov = np.atleast_2d((X_cent.T @ X_cent) / float(len(X_cent) - 1))
    d_var = np.clip(np.diag(cov), 1e-12, None)
    R = cov / np.sqrt(np.outer(d_var, d_var))
    np.fill_diagonal(R, 1.0)
    return R, incs


def compute_equilibrium_correlation(ang_list):
    offs = C.maxgap_shift(ang_list)
    shifted, _ = C.dpca_plus_coords(ang_list, offs=offs)
    X = np.concatenate(shifted, axis=0)
    X_cent = X - np.mean(X, axis=0, keepdims=True)
    cov = np.atleast_2d((X_cent.T @ X_cent) / float(len(X_cent) - 1))
    d_var = np.clip(np.diag(cov), 1e-12, None)
    C_eq = cov / np.sqrt(np.outer(d_var, d_var))
    np.fill_diagonal(C_eq, 1.0)
    return C_eq


def compute_noise_floor(incs, n_surr=10, block=100, seed=42):
    rng = np.random.default_rng(seed)
    surr_means, surr_p95, surr_p99 = [], [], []
    for _ in range(n_surr):
        sh = []
        for x in incs:
            T, d = x.shape
            b_eff = min(block, max(5, T // 4))
            shifts = rng.integers(b_eff, max(b_eff + 1, T - b_eff), size=d)
            idx = (np.arange(T)[:, None] - shifts[None, :]) % T
            sh.append(x[idx, np.arange(d)[None, :]])
        X_s = np.concatenate(sh, axis=0)
        X_s_cent = X_s - np.mean(X_s, axis=0, keepdims=True)
        cov_s = np.atleast_2d((X_s_cent.T @ X_s_cent) / float(len(X_s_cent) - 1))
        d_var = np.clip(np.diag(cov_s), 1e-12, None)
        R_s = cov_s / np.sqrt(np.outer(d_var, d_var))
        iu = np.triu_indices(R_s.shape[0], 1)
        vals = np.abs(R_s[iu])
        if len(vals) > 0:
            surr_means.append(float(np.mean(vals)))
            surr_p95.append(float(np.percentile(vals, 95)))
            surr_p99.append(float(np.percentile(vals, 99)))
    if not surr_means:
        return {"mean_abs_R": 0.005, "p95": 0.015, "p99": 0.025}
    return {
        "mean_abs_R": float(np.mean(surr_means)),
        "p95": float(np.mean(surr_p95)),
        "p99": float(np.mean(surr_p99)),
    }


def profile_with_crankshaft(R, meta):
    order = meta["chain_order"]
    R_chain = R[np.ix_(order, order)]
    d = R_chain.shape[0]
    deltas = np.arange(1, d)
    m_abs = np.array([np.mean(np.abs(np.diag(R_chain, k))) for k in deltas], dtype=np.float64)
    m_sgn = np.array([np.mean(np.diag(R_chain, k)) for k in deltas], dtype=np.float64)

    diag1 = np.diag(R_chain, 1)
    pep_pairs = meta["block_idx_pep"]
    res_pairs = meta["block_idx_res"]

    pep_vals = np.array([R[i, j] for i, j in pep_pairs], dtype=np.float64) if len(pep_pairs) > 0 else np.array([])
    intra_vals = np.array([R[i, j] for i, j in res_pairs], dtype=np.float64) if len(res_pairs) > 0 else np.array([])

    crankshaft = {
        "R_delta1_signed_overall": float(np.mean(diag1)) if len(diag1) > 0 else None,
        "R_delta1_peptide_bond_signed": float(np.mean(pep_vals)) if len(pep_vals) > 0 else None,
        "R_delta1_intra_res_signed": float(np.mean(intra_vals)) if len(intra_vals) > 0 else None,
        "R_delta1_peptide_bond_abs": float(np.mean(np.abs(pep_vals))) if len(pep_vals) > 0 else None,
        "R_delta1_intra_res_abs": float(np.mean(np.abs(intra_vals))) if len(intra_vals) > 0 else None,
    }
    return deltas, m_abs, m_sgn, crankshaft


def fit_sequence_decay(deltas, m_abs, floor):
    # Fit sequence decay over short-to-medium covalent range before tertiary wraparound dominates
    max_fit = min(len(deltas), 30)
    x = deltas[:max_fit].astype(np.float64)
    y = np.clip(m_abs[:max_fit] - floor, 1e-6, None)
    n = len(x)
    if n < 4:
        return {"note": "insufficient points for decay fit"}

    results = {}
    try:
        p_exp, _ = optimize.curve_fit(
            lambda s, A, xi: A * np.exp(-s / xi),
            x, y, p0=[y[0], 1.5], bounds=([0, 0.05], [10.0, 500.0]), maxfev=10000
        )
        res_exp = y - (p_exp[0] * np.exp(-x / p_exp[1]))
        wrss_exp = float(np.sum(res_exp ** 2))
        k = 2
        aic_exp = float(n * np.log(max(wrss_exp / n, 1e-30)) + 2 * k)
        aicc_exp = float(aic_exp + (2 * k * (k + 1)) / max(1, n - k - 1))
        results["exponential"] = {
            "A": float(p_exp[0]), "xi": float(p_exp[1]),
            "wrss": wrss_exp, "aic": aic_exp, "aicc": aicc_exp
        }
    except Exception as e:
        results["exponential"] = {"error": str(e)}

    try:
        p_pow, _ = optimize.curve_fit(
            lambda s, B, beta: B * (s ** (-beta)),
            x, y, p0=[y[0], 1.5], bounds=([0, 0.05], [10.0, 20.0]), maxfev=10000
        )
        res_pow = y - (p_pow[0] * (x ** (-p_pow[1])))
        wrss_pow = float(np.sum(res_pow ** 2))
        k = 2
        aic_pow = float(n * np.log(max(wrss_pow / n, 1e-30)) + 2 * k)
        aicc_pow = float(aic_pow + (2 * k * (k + 1)) / max(1, n - k - 1))
        results["power_law"] = {
            "B": float(p_pow[0]), "beta": float(p_pow[1]),
            "wrss": wrss_pow, "aic": aic_pow, "aicc": aicc_pow
        }
    except Exception as e:
        results["power_law"] = {"error": str(e)}

    if "power_law" in results and "exponential" in results and "aicc" in results["power_law"] and "aicc" in results["exponential"]:
        results["delta_aicc_pow_minus_exp"] = float(results["power_law"]["aicc"] - results["exponential"]["aicc"])
    return results


def compute_contact_coupling_correlations(sys_key, R_eq, ca_positions, meta):
    if ca_positions is None or len(ca_positions) == 0:
        return None
    ca_all = np.concatenate(ca_positions, axis=0)
    top_obj = C.topology(sys_key)
    ca_atoms = top_obj.topology.select("name CA")
    ca_res_indices = [top_obj.topology.atom(a).residue.index for a in ca_atoms]

    # Map residue index -> dihedral indices in R_eq
    all_res = meta["all_res"]
    res_to_dih = {}
    for d_idx, r_idx in enumerate(all_res):
        res_to_dih.setdefault(int(r_idx), []).append(d_idx)

    # Align residues that have both CA and at least 1 backbone dihedral
    aligned_ca_cols = []
    aligned_dih_blocks = []
    aligned_res_ids = []
    for col_i, r_id in enumerate(ca_res_indices):
        if col_i < ca_all.shape[1] and r_id in res_to_dih:
            aligned_ca_cols.append(col_i)
            aligned_dih_blocks.append(res_to_dih[r_id])
            aligned_res_ids.append(r_id)

    n_common = len(aligned_res_ids)
    if n_common < 8:
        return None

    step = max(1, len(ca_all) // 800)
    ca_sub = ca_all[::step][:, aligned_ca_cols, :]
    diff = ca_sub[:, :, None, :] - ca_sub[:, None, :, :]
    dist_traj = np.sqrt(np.sum(diff ** 2, axis=-1))
    mean_dist = np.mean(dist_traj, axis=0)
    contact_prob = np.mean(dist_traj < 0.85, axis=0)
    contact_var = np.var(dist_traj < 0.85, axis=0)

    C_res = np.zeros((n_common, n_common), dtype=np.float64)
    for i in range(n_common):
        bi = aligned_dih_blocks[i]
        for j in range(i, n_common):
            bj = aligned_dih_blocks[j]
            val = float(np.linalg.norm(R_eq[np.ix_(bi, bj)], "fro") / np.sqrt(len(bi) * len(bj)))
            C_res[i, j] = val
            C_res[j, i] = val

    pairs_i, pairs_j = [], []
    for i in range(n_common):
        for j in range(i + 1, n_common):
            if abs(aligned_res_ids[i] - aligned_res_ids[j]) >= 4:
                pairs_i.append(i)
                pairs_j.append(j)
    if len(pairs_i) < 6:
        return None

    pairs_i = np.array(pairs_i)
    pairs_j = np.array(pairs_j)
    c_vals = C_res[pairs_i, pairs_j]
    d_vals = mean_dist[pairs_i, pairs_j]
    p_vals = contact_prob[pairs_i, pairs_j]
    chi_vals = contact_var[pairs_i, pairs_j]
    seq_sep = np.abs(np.array(aligned_res_ids)[pairs_i] - np.array(aligned_res_ids)[pairs_j]).astype(np.float64)

    rho_contact = float(stats.spearmanr(c_vals, p_vals).statistic) if np.std(p_vals) > 1e-12 else 0.0
    rho_dist = float(stats.spearmanr(c_vals, -d_vals).statistic)

    def partial_spearman(x, y, z):
        rx = stats.rankdata(x)
        ry = stats.rankdata(y)
        rz = stats.rankdata(z)
        sx, ix, _, _, _ = stats.linregress(rz, rx)
        sy, iy, _, _, _ = stats.linregress(rz, ry)
        res_x = rx - (sx * rz + ix)
        res_y = ry - (sy * rz + iy)
        if np.std(res_x) < 1e-12 or np.std(res_y) < 1e-12:
            return 0.0, 1.0
        r, p = stats.pearsonr(res_x, res_y)
        return float(r), float(p)

    # Use inverse mean 3D distance (-d_vals) as continuous tertiary proximity metric when contact_prob is sparse
    target_prox = p_vals if np.std(p_vals) > 1e-6 else (-d_vals)
    partial_rho, partial_p = partial_spearman(c_vals, target_prox, seq_sep)
    partial_rho_dist, partial_p_dist = partial_spearman(c_vals, -d_vals, seq_sep)

    in_cont = d_vals < 0.85
    out_cont = d_vals >= 0.85
    if np.any(in_cont) and np.any(out_cont):
        enrich_ratio = float(np.mean(c_vals[in_cont]) / max(float(np.mean(c_vals[out_cont])), 1e-8))
    else:
        enrich_ratio = 1.0

    return {
        "n_nonlocal_pairs": int(len(pairs_i)),
        "enrichment_ratio": enrich_ratio,
        "spearman_coupling_vs_contact_prob": rho_contact,
        "spearman_coupling_vs_inv_distance": rho_dist,
        "partial_spearman_contact_given_seq": partial_rho,
        "partial_p_value": partial_p,
        "partial_spearman_inv_dist_given_seq": partial_rho_dist,
        "partial_p_inv_dist": partial_p_dist,
        "mean_dynamic_susceptibility_chi": float(np.mean(chi_vals)),
    }


def main():
    print("[M32] Starting Milestone 32: Law 2 Sequence Decay, Crankshaft, & Sparsity (15 Systems)...", flush=True)

    results = {
        "milestone": "M32_Law2_Sequence_Contact_Coupling_and_Sparsity",
        "n_systems_verified": 0,
        "systems": {},
        "universal_summary": {},
    }

    profiles = {}
    crankshaft_results = {}
    contact_results = {}
    sparsity_results = {}

    for k in C.ORDER_V4:
        meta_sys = C.SYSTEMS_V4[k]
        meta_dih = C.dihedral_meta(k)
        d = int(meta_dih["d"])
        print(f"  Analyzing {k:10s} (d={d:3d})...", flush=True)

        reps = C.load_atlas_replicas_v4(k, verify_hash=True, max_frames_per_rep=15000, load_3d=True)
        ang_list = [r["angles"] for r in reps]
        ca_list = [r["positions_ca"] for r in reps if r.get("positions_ca") is not None]

        R_tau1, incs = compute_increment_correlation(ang_list, tau=1)
        C_eq = compute_equilibrium_correlation(ang_list)
        noise_floor = compute_noise_floor(incs, n_surr=10)

        deltas, m_abs, m_sgn, crank = profile_with_crankshaft(R_tau1, meta_dih)
        decay = fit_sequence_decay(deltas, m_abs, noise_floor["mean_abs_R"])
        contact = compute_contact_coupling_correlations(k, C_eq, ca_list, meta_dih)

        eps = 0.05
        iu = np.triu_indices(d, 1)
        abs_R = np.abs(R_tau1[iu])
        n_pairs = len(abs_R)
        n_sig = int(np.sum(abs_R > eps))
        k_eff = float(2.0 * n_sig / d) if d > 0 else 0.0
        sparsity_frac = float(1.0 - (n_sig / max(1, n_pairs)))

        results["systems"][k] = {
            "d": d,
            "name": meta_sys["name"],
            "solvent": reps[0]["solvent"],
            "noise_floor": noise_floor,
            "crankshaft_parity": crank,
            "decay_models": decay,
            "contact_gating": contact,
            "sparsity_eps_0.05": {
                "k_eff_mean_partners": round(k_eff, 4),
                "sparsity_negligible_fraction": round(sparsity_frac, 5),
                "theoretical_bound": round(float(1.0 - k_eff / max(1, d - 1)), 5),
            },
        }
        profiles[k] = (deltas, m_abs, m_sgn)
        crankshaft_results[k] = crank
        contact_results[k] = contact
        sparsity_results[k] = (d, k_eff, sparsity_frac)

    results["n_systems_verified"] = len(results["systems"])

    pep_all = [
        v["R_delta1_peptide_bond_signed"]
        for k, v in crankshaft_results.items()
        if k in C.PROTEINS_V4 and v.get("R_delta1_peptide_bond_signed") is not None
    ]
    intra_all = [
        v["R_delta1_intra_res_signed"]
        for k, v in crankshaft_results.items()
        if k in C.PROTEINS_V4 and v.get("R_delta1_intra_res_signed") is not None
    ]
    k_eff_prots = [sparsity_results[k][1] for k in C.PROTEINS_V4 if k in sparsity_results]
    sp_prots = [sparsity_results[k][2] for k in C.PROTEINS_V4 if k in sparsity_results]

    results["universal_summary"] = {
        "proteins_mean_peptide_crankshaft_R": round(float(np.mean(pep_all)), 4),
        "proteins_std_peptide_crankshaft_R": round(float(np.std(pep_all)), 4),
        "proteins_mean_intra_residue_R": round(float(np.mean(intra_all)), 4),
        "proteins_mean_k_eff_eps005": round(float(np.mean(k_eff_prots)), 3),
        "min_protein_sparsity_pct": round(float(min(sp_prots) * 100.0), 2),
        "max_protein_sparsity_pct": round(float(max(sp_prots) * 100.0), 2),
    }

    print("\n" + "=" * 72)
    print(f"LAW 2 SUMMARY ACROSS ALL {results['n_systems_verified']} ATLAS SYSTEMS:")
    print(f"  Mean Peptide-Bond Crankshaft R(psi_i, phi_{{i+1}}) = {results['universal_summary']['proteins_mean_peptide_crankshaft_R']:+.3f} +/- {results['universal_summary']['proteins_std_peptide_crankshaft_R']:.3f}")
    print(f"  Mean Intra-Residue R(phi_i, psi_i)                = {results['universal_summary']['proteins_mean_intra_residue_R']:+.3f}")
    print(f"  Protein Sparsity Range (d=18 -> 738, eps=0.05)    = {results['universal_summary']['min_protein_sparsity_pct']:.1f}% -> {results['universal_summary']['max_protein_sparsity_pct']:.1f}% (mean k_eff={results['universal_summary']['proteins_mean_k_eff_eps005']:.2f})")
    print("=" * 72)

    out_json = os.path.join(RESULTS_DIR, "r32_locality_and_sparsity_v4.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[M32] Saved results JSON to {out_json}")

    # Generate 4-Panel Publication Figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    c_coral = "#dc2626"
    c_navy = "#1e3a8a"
    c_cyan = "#0d9488"

    ax = axes[0, 0]
    highlight_keys = ["chignolin", "trpcage", "villin", "ubiquitin", "lysozyme", "dhfr", "adk", "mbp"]
    for k in highlight_keys:
        if k in profiles:
            deltas, m_abs, _ = profiles[k]
            ax.plot(deltas[:60], m_abs[:60], marker="o", markersize=2.5, lw=1.4, label=f"{C.SYSTEMS_V4[k]['short']} ($d={C.SYSTEMS_V4[k]['d']}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Covalent Dihedral Separation $\\Delta$", fontweight="bold")
    ax.set_ylabel("Mean $|R(\\Delta)|$", fontweight="bold")
    ax.set_title("A. Universal Covalent Sequence Decay ($d = 18 \\to 738$)", fontweight="bold")
    ax.legend(frameon=True, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[0, 1]
    prot_names = [k for k in C.PROTEINS_V4 if k in crankshaft_results]
    pep_v = [crankshaft_results[k]["R_delta1_peptide_bond_signed"] for k in prot_names]
    intra_v = [crankshaft_results[k]["R_delta1_intra_res_signed"] for k in prot_names]
    x_idx = np.arange(len(prot_names))
    w = 0.38
    ax.bar(x_idx - w / 2, pep_v, width=w, color=c_coral, label="Peptide Crankshaft $(\\psi_r, \\phi_{r+1})$")
    ax.bar(x_idx + w / 2, intra_v, width=w, color=c_navy, label="Intra-Residue $(\\phi_r, \\psi_r)$")
    ax.set_xticks(x_idx)
    ax.set_xticklabels([C.SYSTEMS_V4[k]["short"] for k in prot_names], rotation=35, ha="right", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Signed Increment Correlation $R$", fontweight="bold")
    ax.set_title("B. Universal Peptide-Plane Crankshaft Counter-Rotation", fontweight="bold")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.legend(frameon=True, fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 0]
    cg_keys = [k for k in C.PROTEINS_V4 if contact_results.get(k) is not None]
    rhos = [contact_results[k]["partial_spearman_inv_dist_given_seq"] for k in cg_keys]
    ax.bar([C.SYSTEMS_V4[k]["short"] for k in cg_keys], rhos, color=c_cyan, edgecolor=c_navy, alpha=0.88)
    ax.set_ylabel("Partial Spearman $\\rho$ ($C^{\\mathrm{eq}}_{ij}$ vs $-r_{ij} \\mid |i-j|$)", fontweight="bold")
    ax.set_title("C. Non-Local 3D Tertiary Contact Gating ($|i - j| \\geq 4$)", fontweight="bold")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8.5, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    d_vals = [sparsity_results[k][0] for k in C.PROTEINS_V4 if k in sparsity_results]
    s_vals = [sparsity_results[k][2] * 100.0 for k in C.PROTEINS_V4 if k in sparsity_results]
    mean_keff = float(np.mean(k_eff_prots))
    ax.scatter(d_vals, s_vals, color=c_coral, s=75, zorder=5, label="Measured Sparsity $S_{0.05}(d)$ (13 Proteins)")
    d_arr = np.linspace(16, 760, 200)
    s_theory = np.clip((1.0 - mean_keff / (d_arr - 1.0)) * 100.0, 0.0, 100.0)
    ax.plot(d_arr, s_theory, "--", color=c_navy, lw=2.0, label=f"Bounded-Degree Law: $1 - \\bar{{k}}_{{\\mathrm{{eff}}}}/(d-1)$ ($\\bar{{k}}_{{\\mathrm{{eff}}}}={mean_keff:.2f}$)")
    ax.set_xlabel("Backbone Dihedral Dimension $d$", fontweight="bold")
    ax.set_ylabel("Operator Sparsity $S_{0.05}$ (%)", fontweight="bold")
    ax.set_title("D. Asymptotic $O(d)$ Sparsity Scaling ($d = 18 \\to 738$)", fontweight="bold")
    ax.legend(frameon=True, fontsize=8.5, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_pdf = os.path.join(FIGS_DIR, "fig32_sequence_decay_and_contact_gating.pdf")
    fig_png = os.path.join(FIGS_DIR, "fig32_sequence_decay_and_contact_gating.png")
    fig.savefig(fig_pdf, dpi=300)
    fig.savefig(fig_png, dpi=300)
    plt.close(fig)
    print(f"[M32] Saved publication figures to {fig_pdf} and {fig_png}")


if __name__ == "__main__":
    main()
