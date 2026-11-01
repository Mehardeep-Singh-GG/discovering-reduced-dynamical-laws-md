#!/usr/bin/env python3
"""
si_part2_s3.py
==============
Builds Section S3 (Complete Per-Protein & Nested Model-Selection Tables S4-S9 for Laws 1-6)
with zero internal milestone or codebase jargon.
"""
import json
from pathlib import Path
from si_part1_preamble_and_s1_s2 import SYS_LABELS

PROT_LABELS = SYS_LABELS[:13]


def build_part2_s3(res_dir: Path) -> str:
    d31 = json.load(open(res_dir / "r31_rank_and_scaling_v4.json"))
    d32 = json.load(open(res_dir / "r32_locality_and_sparsity_v4.json"))
    d33 = json.load(open(res_dir / "r33_sparse_operator_benchmark_v4.json"))
    d34 = json.load(open(res_dir / "r34_solvent_internal_friction_v4.json"))
    d35 = json.load(open(res_dir / "r35_nmr_s2_bfactors_v4.json"))
    d36 = json.load(open(res_dir / "r36_lopo_cross_protein_v4.json"))

    # Table S4: Law 1 Spectral Rank & Out-of-Scale Extrapolation
    extrap = d31["out_of_scale_extrapolation"]["heldout_large_proteins_d_218_to_738"]
    t_s4_rows = []
    for sk, label, dval, _ in SYS_LABELS:
        s = d31["systems"][sk]
        dp = s["dpca_plus"]
        df = s["diffusion_operator_tau1ps"]
        nd = s["null_diagonal"]
        sc = s["sincos_dpca"]
        if sk in extrap:
            ex = extrap[sk]
            ex_str = f"{ex['predicted_linear_null_r90']:.1f} ({ex['ape_linear_null_pct']:.1f}\\%) / {ex['predicted_sublinear_r90']:.1f} ({ex['ape_sublinear_pct']:.1f}\\%)"
        else:
            ex_str = "Calibration ($d \\le 150$)" if dval >= 18 else "Peptide Control"
        t_s4_rows.append(
            f"\\textbf{{{label}}} & {dval} & {dp['r80']} & {dp['r90']} ({dp['r90_cont']:.1f}) & "
            f"{dp['r95']} & {dp['pr']:.2f} & {dp['n_ent']:.2f} & {df['r90_cont']:.1f} ({df['pr']:.2f}) & "
            f"{nd['r90_cont']:.1f} & {sc['r90']} & {ex_str} \\\\"
        )

    # Table S5: Law 2 Crankshaft Parity, Sequence Decay, & Contact Gating
    t_s5_rows = []
    for sk, label, dval, _ in SYS_LABELS:
        s = d32["systems"][sk]
        cp = s["crankshaft_parity"]
        dm = s["decay_models"]
        sp = s["sparsity_eps_0.05"]
        r_pep = f"{cp['R_delta1_peptide_bond_signed']:+.3f}" if cp["R_delta1_peptide_bond_signed"] is not None else "---"
        r_int = f"{cp['R_delta1_intra_res_signed']:+.3f}"
        if "exponential" in dm and dm["exponential"].get("xi") is not None:
            dec_s = f"$\\xi={dm['exponential']['xi']:.2f},\\;\\beta={dm['power_law']['beta']:.2f}$ ({dm['delta_aicc_pow_minus_exp']:+.1f})"
        else:
            dec_s = "---"
        if "contact_gating" in s and s["contact_gating"] and s["contact_gating"].get("enrichment_ratio") is not None:
            cg = s["contact_gating"]
            cg_s = f"{cg['enrichment_ratio']:.2f}$\\times$ ($r_p={cg['partial_spearman_contact_given_seq']:+.3f}$)"
        else:
            cg_s = "---"
        t_s5_rows.append(
            f"\\textbf{{{label}}} & {dval} & {r_pep} & {r_int} & {dec_s} & {cg_s} & "
            f"{sp['k_eff_mean_partners']:.2f} & {100.0*sp['sparsity_negligible_fraction']:.1f}\\% \\\\"
        )

    # Table S6: Law 3 6-Estimator Sparse Operator Benchmark across 13 Proteins
    t_s6_rows = []
    for sk, label, dval, _ in PROT_LABELS:
        m = d33["systems"][sk]["metrics"]
        cg = m["contact_graph"]
        cb = m["chordal_band"]
        dt = m["dense_tuned"]
        lw = m["ledoit_wolf"]
        gl = m["graphical_lasso"]
        dg = m["diag"]
        t_s6_rows.append(
            f"\\textbf{{{label}}} & {dval} & "
            f"\\textbf{{{cg['ell_per_d']:+.3f}}} / {dt['ell_per_d']:+.3f} / {gl['ell_per_d']:+.3f} & "
            f"\\textbf{{{cg['frob_err']:.4f}}} / {cb['frob_err']:.4f} / {lw['frob_err']:.4f} / {dg['frob_err']:.4f} & "
            f"{cg['eta_subspace']:.3f} & "
            f"\\textbf{{{cg['mem_kb']:.1f}}} vs.\\ {dt['mem_kb']:.1f} & "
            f"\\textbf{{{1000.0*cg['time_s']:.1f}}} vs.\\ {1000.0*gl['time_s']:.0f} \\\\"
        )

    # Table S7: Law 4 Relative SASA Shielding & Nested Thermal Model Selection
    t_s7a_rows = []
    for sk, label, dval, _ in PROT_LABELS:
        ss = d34["structural_sasa_summary"][sk]
        t_s7a_rows.append(
            f"\\textbf{{{label}}} & {dval} & {ss['mean_rsasa_all']:.3f} & {ss['mean_rsasa_interior']:.3f} & "
            f"{ss['mean_rsasa_core']:.3f} & {ss['mean_rsasa_surface']:.3f} \\\\"
        )
    nested_t = d34["temperature_and_friction_sweeps"]["chignolin_folded_interior"]["nested_law_evaluation_temperature"]["models"]
    model_labels_s7 = [
        ("H4a_linear_0p", "Linear Dry Einstein ($D_i(T) = D_i(T_0)\\,T/T_0$, 0 free params)"),
        ("H4b_power_1p", "Empirical Power Law ($D_i(T) = D_i(T_0)(T/T_0)^\\alpha$, 1 free param)"),
        ("H4c_kramers_1p", "1-Parameter Kramers Solvent Coupling (1 free param)"),
        ("H4c_kramers_2p", "2-Parameter Kramers Internal + Solvent Friction (2 free params)"),
    ]
    t_s7b_rows = []
    for mk, mlbl in model_labels_s7:
        mv = nested_t[mk]
        t_s7b_rows.append(
            f"{mlbl} & {mv['heldout_rmse']:.4f} & {mv['heldout_mae']:.4f} & {mv['heldout_mape_pct']:.2f}\\% & {mv['heldout_total_nll']:.2f} \\\\"
        )

    # Table S8: Law 5 NMR S2 (4 BMRB Proteins + Pooled) & 8-Protein X-Ray B-Factors
    nmr_prots = [("ubiquitin", "Ubiquitin (1UBQ, BMRB 6457, 300 K)"),
                 ("gb1", "GB1 Domain (1PGA, BMRB 7280, 298 K)"),
                 ("lysozyme", "HEW Lysozyme (1AKI, BMRB 4562, 308 K)"),
                 ("dhfr", "E.\\ coli DHFR (1RX2, BMRB 4380, 298 K)")]
    t_s8a_rows = []
    for pk, plbl in nmr_prots:
        pm = d35["per_protein_metrics"][pk]
        cr = pm["H5b_crankshaft"]
        fl = pm["H5a_flank_diag"]
        sp = pm["H5c_spectral"]
        md = pm["baseline_direct_md"]
        gn = pm["baseline_tuned_gnm"]
        t_s8a_rows.append(
            f"\\textbf{{{plbl}}} & \\textbf{{{cr['spearman_r']:.3f}}} ({cr['pearson_r']:.3f}) & "
            f"{fl['spearman_r']:.3f} & {sp['spearman_r']:.3f} & {md['spearman_r']:.3f} & "
            f"{gn['spearman_r']:.3f} & {cr['rmse']:.3f} / {gn['rmse']:.3f} \\\\"
        )
    gl = d35["global_lopo_summary"]
    t_s8a_rows.append(
        f"\\midrule[0.5pt]\\textbf{{Pooled Cross-Protein ($N=387$ internal quartets)}} & "
        f"{gl['H5b_crankshaft']['pooled_spearman_r']:.3f} ({gl['H5b_crankshaft']['pooled_pearson_r']:.3f}) & "
        f"{gl['H5a_flank_diag']['pooled_spearman_r']:.3f} & {gl['H5c_spectral']['pooled_spearman_r']:.3f} & "
        f"{gl['baseline_direct_md']['pooled_spearman_r']:.3f} & \\textbf{{{gl['baseline_tuned_gnm']['pooled_spearman_r']:.3f}}} & "
        f"{gl['H5b_crankshaft']['pooled_rmse']:.3f} / \\textbf{{{gl['baseline_tuned_gnm']['pooled_rmse']:.3f}}} \\\\"
    )

    bfac_names = {
        "ubiquitin": "Ubiquitin (1UBQ)", "gb1": "GB1 Domain (1PGA)", "bpti": "BPTI (5PTI)",
        "barnase": "Barnase (1BNI)", "lysozyme": "HEW Lysozyme (1AKI)", "dhfr": "E.\\ coli DHFR (1RX2)",
        "adk": "Adenylate Kinase (4AKE)", "mbp": "Maltose-Binding Protein (1OMP)",
    }
    t_s8b_rows = []
    for pk, bv in d35["crystallographic_bfactor_correlations"].items():
        t_s8b_rows.append(
            f"\\textbf{{{bfac_names.get(pk, pk)}}} & {bv['spearman_bca_vs_flank_var']:+.3f} & "
            f"{bv['p_ca']:.2e} & {bv['spearman_excess_plane_vs_crank_cross']:+.3f} & {bv['p_cross']:.2e} \\\\"
        )

    # Table S9: Law 6 13-Fold Zero-Shot LOPO Transfer (Model Hierarchy + Per-Protein Scorecard)
    lopo_models = [
        ("M0_constant", "Constant Mean Null Predictor"),
        ("M1_v3_ridge9", "9-Feature Structural Ridge Regressor"),
        ("M2_v4_ridge12", "12-Feature Structural + Packing Ridge Regressor"),
        ("M2_plus_gnm", "12-Feature Ridge + Harmonic GNM Hybrid"),
        ("M3_random_forest", "Nonlinear Random Forest Ensemble"),
        ("M3_hist_gbdt", "Nonlinear Histogram Gradient-Boosted Trees (GBDT)"),
        ("noise_floor", "Inter-Split Empirical Reproducibility Ceiling"),
    ]
    t_s9a_rows = []
    for mk, mlbl in lopo_models:
        gm = d36["global_lopo_summary"][mk]
        t_s9a_rows.append(
            f"\\textbf{{{mlbl}}} & {gm['pooled_spearman_r']:+.3f} & {gm['pooled_pearson_r']:+.3f} & "
            f"{gm['mean_oos_r2']:+.3f} & {gm['mean_frob_err_pct']:.1f}\\% & {gm['mean_kl_per_dim']:.4f} \\\\"
        )

    t_s9b_rows = []
    for sk, label, dval, _ in PROT_LABELS:
        ps = d36["per_protein_summary"][sk]
        r12 = ps["M2_v4_ridge12"]
        gb = ps["M3_hist_gbdt"]
        nf = ps["noise_floor"]
        t_s9b_rows.append(
            f"\\textbf{{{label}}} & {dval} & {r12['spearman_r']:.3f} & {r12['oos_r2']:+.3f} & "
            f"{r12['frob_err_pct']:.1f}\\% & \\textbf{{{gb['spearman_r']:.3f}}} & \\textbf{{{gb['oos_r2']:+.3f}}} & "
            f"{gb['frob_err_pct']:.1f}\\% & {nf['spearman_r']:.3f} ({nf['frob_err_pct']:.1f}\\%) \\\\"
        )

    return r"""
\clearpage
\section{Complete Per-Protein Scaling, Locality, Friction, NMR, and Zero-Shot Transfer Tables (Laws 1--6)}

\subsection{Law 1: Dual Spectral Compression vs.\ Extensive Native-Basin Tail Variance}
Table S4 reports cumulative variance ranks ($r_{80}, r_{90}, r_{95}$), participation ratios $PR(d)$, Shannon spectral entropy ranks $N_{\mathrm{ent}}$, short-time jump-diffusion ranks ($\tau = 1\text{ ps}$), uncorrelated diagonal null ranks, $2d$ $(\cos\theta,\sin\theta)$ ranks, and blind out-of-scale extrapolations from small/medium proteins ($d \le 150$) to held-out large enzymes ($d \in \{218, 256, 316, 426, 738\}$). Across all \vFourTotalSystems{} systems, collective slow fluctuations compress sub-linearly as $PR(d) = \vFourPRPrefactor{}\,d^{\vFourPRExponent{}}$ ($\Delta\mathrm{AIC}_c = \vFourPRDeltaAICc{}$) and $r_{90}^D(d) = \vFourDiffOpRankPrefactor{}\,d^{\vFourDiffOpRankAlpha{}}$ ($\Delta\mathrm{AIC}_c = \vFourDiffOpDeltaAICc{}$), whereas cumulative $90\%$ equilibrium covariance rank transitions in folded globular proteins to extensive linear proportionality ($r_{90} \approx \vFourProteinsOnlyNullCZero{}\,d$, achieving $\vFourRankLinearNullExtrapolationMAPE{}\%$ blind extrapolation error vs.\ $\vFourRankExtrapolationMAPE{}\%$ for small-protein sub-linear power laws) because hundreds of quasi-independent surface loops and local peptide-plane librations each contribute $\sim 0.3\%$ thermal variance. Note two physical diagnostics directly visible in Table S4: (i) folded implicit-solvent proteins already exhibit extensive variance rank ($r_{90}/d = 52/110 = 0.47$ in GB1, $55/150 = 0.37$ in Ubiquitin, and $75/218 = 0.34$ in Barnase), matching explicit-PME enzymes ($0.41\text{--}0.51$) and confirming that extensive tail dimensionality reflects folded scaffold size rather than explicit water caging; and (ii) \textit{E.\ coli} DHFR (\texttt{1RX2}, $d=316$) exhibits $PR(d) = \vFourPRDhfr{}$ (Split 3: $77.26$) because its compact single-domain Rossmann fold lacks both a large-amplitude inter-lobe hinge ($\lambda_1 = 0.89\text{ rad}^2$, vs.\ $\lambda_1 = 7.65\text{ rad}^2, PR = \vFourPRAdk{}$ in multi-domain AdK) and disordered termini, distributing thermal variance flatly across dozens of localized catalytic-loop modes.

\begin{table}[!htbp]
\centering
\caption*{Table S4. Toroidal spectral rank, participation ratio, diffusion rank, and blind enzyme extrapolation ($d = 2 \to 738$)}
\resizebox{\textwidth}{!}{%
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{3.8pt}
\begin{tabular}{@{}l c c c c c c c c c c@{}}
\toprule[1.1pt]
\textbf{System (PDB)} & \textbf{$d$} & \textbf{$r_{80}$} & \textbf{$r_{90}$ (Cont.)} & \textbf{$r_{95}$} & \textbf{$PR(d)$} & \textbf{$N_{\mathrm{ent}}$} & \textbf{Diff.\ $r_{90}^D$ ($PR^D$)} & \textbf{Null $r_{90}$} & \textbf{$\sin/\cos$ $r_{90}$} & \textbf{Blind Extrap.: Linear (APE) / Sub-lin.\ (APE)} \\
\midrule[0.6pt]
""" + "\n".join(t_s4_rows) + r"""
\bottomrule[1.1pt]
\end{tabular}%
}
\end{table}

\subsection{Law 2 \& Law 3: Crankshaft Parity, Contact-Gated Sparsity, and 6-Estimator Complexity Benchmark}
Table S5 documents universal peptide-plane crankshaft anti-correlation ($\bar{R}(\psi_i,\phi_{i+1}) = \vFourMeanPeptideCrankshaftR{} \pm \vFourStdPeptideCrankshaftR{}$ vs.\ intra-residue $\bar{R}(\phi_i,\psi_i) = \vFourMeanIntraResR{}$), along-chain correlation decay ($\xi, \beta, \Delta\mathrm{AIC}_c$), non-local tertiary contact enrichment ($|i-j|\ge 4$, $r_c = 0.85\text{ nm}$), and bounded partner degree ($\bar{k}_{\mathrm{eff}} = \vFourKeffMean{}$, reaching $\vFourMaxProteinSparsityPct{}\%$ sparsity on MBP). Table S6 benchmarks six covariance/operator estimators fitted strictly on Split 1, tuned on Split 2, and evaluated on Split 3 across all \vFourNumProteins{} macromolecular proteins, demonstrating $\mathcal{O}(d^{\vFourTimeExpChordal{}})$ runtime and $\mathcal{O}(d^{\vFourMemExpContact{}})$ memory scaling for sparse directed acyclic contact-graph Cholesky regression ($\hat{\bm{\Omega}}_{\mathcal{G}} = (\mathbf{I}-\mathbf{B})^\top\mathbf{D}_{\mathrm{cond}}^{-1}(\mathbf{I}-\mathbf{B})$; Theorem S3). Reporting relative Frobenius errors to four decimal places in Table S6 shows why Contact-Graph and Chordal Sequence-Band agree to three decimal places on large enzymes (e.g., $0.5011$ vs.\ $0.5014$ on MBP, $0.3697$ vs.\ $0.3702$ on Barnase, $0.3992$ vs.\ $0.3985$ on AdK): equal-time $L_2$ covariance norms are dominated by diagonal variances and strong covalent crankshaft blocks ($|\bar{R}|\approx 0.60$), whereas weaker non-local tertiary contacts ($|R|\sim 0.05\text{--}0.12$) shift $\|\hat{\bm{\Sigma}}-\bm{\Sigma}_{\mathrm{test}}\|_F$ at the fourth decimal place while altering precision log-likelihood $\ell_{\mathrm{test}}/d$ and multi-picosecond distal loop routing (Section S4.2).

\begin{table}[!htbp]
\centering
\caption*{Table S5. Peptide-plane crankshaft parity, sequence-separation decay, tertiary contact enrichment, and sparsity ($S_{0.05}$)}
\resizebox{\textwidth}{!}{%
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{4.2pt}
\begin{tabular}{@{}l c c c c c c c@{}}
\toprule[1.1pt]
\textbf{System (PDB)} & \textbf{$d$} & \textbf{$\bar{R}(\psi_i,\phi_{i+1})$} & \textbf{$\bar{R}(\phi_i,\psi_i)$} & \textbf{Decay Fits $\xi, \beta$ ($\Delta\mathrm{AIC}_c$)} & \textbf{Contact Enrichment ($r_p \mid |i-j|$)} & \textbf{$k_{\mathrm{eff}}$} & \textbf{Sparsity $S_{0.05}$} \\
\midrule[0.6pt]
""" + "\n".join(t_s5_rows) + r"""
\bottomrule[1.1pt]
\end{tabular}%
}
\end{table}

\begin{table}[!htbp]
\centering
\caption*{Table S6. Blind Split 3 log-likelihood, Frobenius error, memory footprint, and wall-clock runtime across six operator estimators}
\resizebox{\textwidth}{!}{%
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{3.8pt}
\begin{tabular}{@{}l c c c c c c@{}}
\toprule[1.1pt]
\textbf{Protein (PDB)} & \textbf{$d$} & \textbf{Blind $\ell_{\mathrm{test}}/d$ (Contact / Dense / Glasso)} & \textbf{Frob.\ Err.\ (Contact / Band / LW / Diag)} & \textbf{Subspace $\eta_{10}$} & \textbf{Memory KB (Contact vs.\ Dense)} & \textbf{Time ms (Contact vs.\ Glasso)} \\
\midrule[0.6pt]
""" + "\n".join(t_s6_rows) + r"""
\bottomrule[1.1pt]
\end{tabular}%
}
\end{table}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.92\textwidth]{version4_godtier/figures/fig32_sequence_decay_and_contact_gating.pdf}
\caption*{Figure S1. Enlarged full-width diagnostic plate for Law 2: peptide-plane crankshaft parity ($\bar{R}(\psi_i,\phi_{i+1}) = \vFourMeanPeptideCrankshaftR{}$), primary-sequence correlation decay, tertiary $C_\alpha$ contact gating ($r_c = 0.85\text{ nm}$), and off-diagonal sparsity scaling across all \vFourTotalSystems{} systems.}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.92\textwidth]{version4_godtier/figures/fig33_sparse_operator_scaling_v4.pdf}
\caption*{Figure S2. Enlarged full-width diagnostic plate for Law 3: empirical memory footprint $\mathcal{O}(d^{\vFourMemExpContact{}})$, wall-clock Cholesky runtime $\mathcal{O}(d^{\vFourTimeExpChordal{}})$, blind held-out log-likelihood $\ell_{\mathrm{test}}/d$, and Frobenius error across six covariance/operator estimators ($d = \vFourMinProteinDim{} \to \vFourMaxProteinDim{}$).}
\end{figure}

\subsection{Law 4: Relative SASA Shielding and Internal vs.\ Solvent Friction}
Table S7 summarizes structural relative SASA burial across all \vFourNumProteins{} proteins alongside held-out thermal model selection ($275\text{--}330\text{ K}$) and Kramers viscosity sweeps ($\gamma_{\mathrm{int}} = \vFourGammaIntExposedSurface{}\text{ ps}^{-1}$, 95\% CI $[10.30, 20.55]\text{ ps}^{-1}$ on exposed surface residues vs.\ solvent-decoupled buried core $\gamma_{\mathrm{int}} \ge 250\text{ ps}^{-1}$).

\begin{table}[!htbp]
\centering
\caption*{Table S7. Structural relative SASA burial across \vFourNumProteins{} proteins (top) and held-out thermal nested model selection (bottom)}
\footnotesize
\renewcommand{\arraystretch}{1.10}
\setlength{\tabcolsep}{5.0pt}
\begin{tabular}{@{}l c c c c c@{}}
\toprule[1.1pt]
\textbf{Protein (PDB)} & \textbf{$d$} & \textbf{Mean $\mathrm{rSASA}_{\mathrm{all}}$} & \textbf{Mean $\mathrm{rSASA}_{\mathrm{interior}}$} & \textbf{Mean $\mathrm{rSASA}_{\mathrm{core}}$} & \textbf{Mean $\mathrm{rSASA}_{\mathrm{surface}}$} \\
\midrule[0.6pt]
""" + "\n".join(t_s7a_rows) + r"""
\bottomrule[1.1pt]
\end{tabular}

\vspace{5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}l c c c c@{}}
\toprule[1.1pt]
\textbf{Nested Thermal Scaling Law (Evaluated on Held-Out Temperatures)} & \textbf{RMSE ($\text{rad}^2/\text{ps}$)} & \textbf{MAE ($\text{rad}^2/\text{ps}$)} & \textbf{Held-Out MAPE} & \textbf{Held-Out Total NLL} \\
\midrule[0.6pt]
""" + "\n".join(t_s7b_rows) + r"""
\bottomrule[1.1pt]
\end{tabular}%
}
\end{table}

\subsection{Law 5 \& Law 6: Solution NMR Lipari--Szabo $S^2$, X-Ray $B$-Factors, and 13-Fold Zero-Shot LOPO Transfer}
Table S8 evaluates parameter-free kinematic projection onto experimental BMRB Lipari--Szabo $S^2$ order parameters across $N = 387$ internal non-proline residues possessing complete four-torsion $(\psi_{i-1},\phi_i,\psi_i,\phi_{i+1})$ flanking quartets (excluding 11 chain-terminal border residues retained in the 398-residue single-residue disorder benchmark of Section S5), plus crystallographic $C_\alpha$ $B$-factor correlations across \vFourBFactorCount{} high-resolution PDB structures. Figures S3 and S4 provide enlarged full-width plates of the Law 4 solvent-shielding and Law 5 NMR/$B$-factor benchmarks compressed two-across in Figure 4 of the main text. Table S9 reports 13-fold zero-shot Leave-One-Protein-Out (LOPO) operator prediction from 12 static PDB structural features (local $(\phi_i,\psi_i)$ Fourier harmonics, Tien relative SASA, local $8\text{ \AA}$ and $12\text{ \AA}$ $C_\alpha$ packing densities, sequence-terminal proximity, secondary-structure propensity, proline/glycine steric indicators, and inverse harmonic contact degree) across seven regressors and all \vFourNumProteins{} folds.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.92\textwidth]{version4_godtier/figures/fig34_solvent_internal_friction_v4.pdf}
\caption*{Figure S3. Enlarged full-width diagnostic plate for Law 4: continuous $\mathrm{rSASA}_i$ transition from dry Einstein equipartition in buried protein cores ($\alpha_{\mathrm{core}} = \vFourAlphaBuriedCore{}$) to viscosity-coupled surface loops ($\alpha_{\mathrm{loop}} = \vFourAlphaSurfaceLoop{}$) and short peptides ($\alpha_{\mathrm{peptide}} = \vFourAlphaShortPeptide{}$).}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.92\textwidth]{version4_godtier/figures/fig35_nmr_s2_and_bfactor_benchmarks.pdf}
\caption*{Figure S4. Enlarged full-width diagnostic plate for Law 5: parameter-free $4\times 4$ crankshaft Jacobian prediction of solution NMR Lipari--Szabo $S^2$ order parameters across 387 internal flanking-quartet residues and crystallographic peptide-plane $B$-factor excess.}
\end{figure}

\begin{table}[!htbp]
\centering
\caption*{Table S8. Experimental BMRB solution NMR Lipari--Szabo $S^2$ benchmark (top) and 8-protein X-ray $B$-factor correlations (bottom)}
\resizebox{\textwidth}{!}{%
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{4.2pt}
\begin{tabular}{@{}l c c c c c c@{}}
\toprule[1.1pt]
\textbf{BMRB Experimental Dataset} & \textbf{Crankshaft $r_s$ ($r$)} & \textbf{Flank Diag.\ $r_s$} & \textbf{Spectral $r_s$} & \textbf{Direct MD $r_s$} & \textbf{Tuned GNM $r_s$} & \textbf{RMSE (Crank / GNM)} \\
\midrule[0.6pt]
""" + "\n".join(t_s8a_rows) + r"""
\bottomrule[1.1pt]
\end{tabular}%
}

\vspace{5pt}
\footnotesize
\setlength{\tabcolsep}{6.0pt}
\begin{tabular}{@{}l c c c c@{}}
\toprule[1.1pt]
\textbf{Crystallographic Structure (PDB)} & \textbf{$B_{C_\alpha}$ vs.\ Flanking Variance $r_s$} & \textbf{$p$-value} & \textbf{Peptide-Plane Excess $B$ vs.\ Crankshaft $r_s$} & \textbf{$p$-value} \\
\midrule[0.6pt]
""" + "\n".join(t_s8b_rows) + r"""
\bottomrule[1.1pt]
\end{tabular}
\end{table}

\begin{table}[!htbp]
\centering
\caption*{Table S9. 13-fold zero-shot Leave-One-Protein-Out (LOPO) PDB-to-operator transfer across model architectures (top) and proteins (bottom)}
\resizebox{\textwidth}{!}{%
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{@{}l c c c c c@{}}
\toprule[1.1pt]
\textbf{Zero-Shot PDB-to-Operator Model Family} & \textbf{Pooled Spearman $r_s$} & \textbf{Pooled Pearson $r$} & \textbf{Mean Out-of-Sample $R^2$} & \textbf{Mean Frob.\ Error} & \textbf{Mean KL / $d$ (nats)} \\
\midrule[0.6pt]
""" + "\n".join(t_s9a_rows) + r"""
\bottomrule[1.1pt]
\end{tabular}%
}

\vspace{5pt}
\resizebox{\textwidth}{!}{%
\renewcommand{\arraystretch}{1.10}
\setlength{\tabcolsep}{4.0pt}
\begin{tabular}{@{}l c c c c c c c c@{}}
\toprule[1.1pt]
\textbf{Held-Out Test Protein (PDB)} & \textbf{$d$} & \textbf{12-Feat.\ Ridge $r_s$} & \textbf{Ridge $R^2$} & \textbf{Ridge Frob.\ Err.} & \textbf{Histogram GBDT $r_s$} & \textbf{GBDT $R^2$} & \textbf{GBDT Frob.\ Err.} & \textbf{Split Ceiling $r_s$ (Frob.)} \\
\midrule[0.6pt]
""" + "\n".join(t_s9b_rows) + r"""
\bottomrule[1.1pt]
\end{tabular}%
}
\end{table}
"""
