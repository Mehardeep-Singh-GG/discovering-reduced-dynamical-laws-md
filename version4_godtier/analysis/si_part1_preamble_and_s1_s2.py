#!/usr/bin/env python3
"""
si_part1_preamble_and_s1_s2.py
==============================
Builds Preamble, Title Banner, Table of Contents, Table S1 (Nomenclature),
Section S1 (Protocols + Tables S2-S3 with full 64-char SHA-256 digests),
and Section S2 (Complete Mathematical Proofs of Theorems S1-S7) for SI_APPENDIX_V4.tex.
Zero internal milestone or script jargon.
"""
import json
from pathlib import Path

SYS_LABELS = [
    ("chignolin", "Chignolin (1UAO)", 18, "10"),
    ("trpcage", "Trp-cage (1L2Y)", 38, "20"),
    ("bba", "BBA5 (1FME)", 54, "28"),
    ("villin", "Villin HP35 (2F4K)", 64, "33"),
    ("wwdomain", "WW-FiP35 (2F21)", 68, "35"),
    ("gb1", "GB1 Domain (1PGA)", 110, "56"),
    ("bpti", "BPTI (5PTI)", 114, "58"),
    ("ubiquitin", "Ubiquitin (1UBQ)", 150, "76"),
    ("barnase", "Barnase (1BNI)", 218, "110"),
    ("lysozyme", "HEW Lysozyme (1AKI)", 256, "129"),
    ("dhfr", "E.\\ coli DHFR (1RX2)", 316, "159"),
    ("adk", "Adenylate Kinase (4AKE)", 426, "214"),
    ("mbp", "Maltose-Binding Protein (1OMP)", 738, "370"),
    ("ala1", "Capped Dialanine (ACE-Ala-NME)", 2, "1$^{\\dag}$"),
    ("ala2", "Capped Trialanine (ACE-Ala$_2$-NME)", 4, "2$^{\\dag}$"),
]

def pretty_solvent(raw: str) -> str:
    r = raw.lower()
    if "obc2" in r:
        return "Implicit GB-OBC2"
    if "gbn2" in r:
        return "Implicit GBn2"
    if "pme" in r:
        return "Explicit TIP3P-FB PME"
    if "tip3p" in r:
        return "Explicit TIP3P"
    return raw.replace("_", r"\_")


def build_part1(res_dir: Path) -> str:
    m29 = json.load(open(res_dir / "m29_verification_report.json"))
    m30 = json.load(open(res_dir / "m30_verification_report.json"))

    g1_sys = m30["gates"]["M30_G1_15sys_3rep_atlas"]["systems"]
    g4_stab = m30["gates"]["M30_G4_physical_fold_stability"]["ensembles"]
    g5_spec = m30["gates"]["M30_G5_inter_replica_spectral_convergence"]["log_spectrum_r2_rep1_vs_rep2"]
    g4_pairs = m29["gates"]["M29_G4_allosteric_reference_pairs"]["pairs"]

    # Table S2: Structural stability, native Rg, spectral convergence, and alternate crystal RMSD
    t_s2_rows = []
    alt_names = {
        "gb1": "3GB1 (NMR)",
        "ubiquitin": "1D3Z (NMR)",
        "dhfr": "1RA1 (occluded)",
        "adk": "1AKE (closed)",
        "mbp": "1ANF (closed)",
    }
    for sk, label, dval, nres in SYS_LABELS:
        st = g4_stab[sk]
        r2_spec = g5_spec[sk]
        sol_p = pretty_solvent(g1_sys[sk]["rep1"]["solvent"])
        if sk in g4_pairs:
            alt_str = f"{alt_names[sk]}: {g4_pairs[sk]['ca_rmsd_open_vs_closed_nm']:.3f}"
        else:
            alt_str = "---"
        t_s2_rows.append(
            f"\\textbf{{{label}}} & {nres} & {dval} & {sol_p} & "
            f"{st['native_rg_nm']:.3f} & {st['mean_rg_nm']:.3f} & {st['mean_q_bhe']:.3f} & "
            f"{st['mean_rmsd_ca_nm']:.3f} & {r2_spec:.4f} & {alt_str} \\\\"
        )

    # Table S3: Full 45-ensemble cryptographic registry with FULL 64-char SHA-256 split cleanly on 2 lines
    t_s3_rows = []
    for sk, label, dval, _ in SYS_LABELS:
        for r_idx, r_key in enumerate(["rep1", "rep2", "rep3"], 1):
            robj = g1_sys[sk][r_key]
            sha = robj["lead_sha256"]
            sha_two_line = f"\\shortstack[l]{{\\texttt{{{sha[:32]}}}\\\\\\texttt{{{sha[32:]}}}}}"
            sol_p = pretty_solvent(robj["solvent"])
            role = {1: "Replica 1 (Discovery)", 2: "Replica 2 (Validation)", 3: "Replica 3 (Blind Test)"}[r_idx]
            lbl_cell = f"\\textbf{{{label}}}" if r_idx == 1 else ""
            d_cell = str(dval) if r_idx == 1 else ""
            rule_after = "\\midrule[0.35pt]" if r_idx == 3 and sk != SYS_LABELS[-1][0] else ""
            t_s3_rows.append(
                f"{lbl_cell} & {d_cell} & {role} & {sol_p} & "
                f"{robj['frames']:,} & {robj['total_ns']:.1f} & {robj['mean_q_bhe']:.3f} & "
                f"{robj['mean_rg_nm']:.3f} & {robj['mean_rmsd_ca_nm']:.3f} & {sha_two_line} \\\\\n{rule_after}"
            )

    tex = r"""\documentclass[10pt,letterpaper]{article}
\usepackage[top=19mm,bottom=19mm,left=17.5mm,right=17.5mm]{geometry}
\linespread{1.08}
\usepackage{amsmath,amssymb,amsfonts,bm,mathtools}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{caption}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage{newtxtext,newtxmath}
\usepackage{xurl}
\usepackage[colorlinks=true,linkcolor=navyblue,citecolor=navyblue,urlcolor=navyblue]{hyperref}

\definecolor{navyblue}{HTML}{002F6C}
\definecolor{crimson}{HTML}{C53030}
\definecolor{darkgray}{HTML}{4A5568}
\definecolor{charcoal}{HTML}{1A202C}
\definecolor{boxbg}{HTML}{F8FAFC}

\DeclareMathOperator{\Tr}{Tr}
\DeclareMathOperator{\diag}{diag}
\DeclareMathOperator{\logm}{logm}

\renewcommand{\thepage}{S\arabic{page}}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}
\fancyhead[L]{\sffamily\bfseries\fontsize{8pt}{10pt}\selectfont\textcolor{darkgray}{Supporting Information --- Singh \& Singh (\textit{J.\ Chem.\ Theory Comput.})}}
\fancyhead[R]{\sffamily\bfseries\fontsize{8pt}{10pt}\selectfont\textcolor{navyblue}{\thepage}}
\fancyfoot[C]{\sffamily\fontsize{7.5pt}{9pt}\selectfont\textcolor{darkgray}{Scalable Contact-Gated Stochastic Operators and Universal Reduction Laws ($d = 2 \to 738$)}}

\titleformat{\section}{\sffamily\bfseries\fontsize{11.5pt}{14pt}\selectfont\color{navyblue}}{\thesection.}{0.5em}{}
\titlespacing*{\section}{0pt}{12pt}{5pt}
\titleformat{\subsection}{\sffamily\bfseries\fontsize{9.8pt}{12pt}\selectfont\color{charcoal}}{\thesubsection}{0.5em}{}
\titlespacing*{\subsection}{0pt}{8pt}{3pt}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thesubsection}{S\arabic{section}.\arabic{subsection}}

\captionsetup{font={footnotesize}, labelfont={bf,sf,color=navyblue}, labelsep=period}
\setlength{\parindent}{10pt}
\setlength{\parskip}{3pt}
\emergencystretch=2.2em

\input{version4_godtier/results/v4_metrics_macros.tex}

\begin{document}
\color{charcoal}

\begin{center}
  \textcolor{navyblue}{\rule{\textwidth}{2.2pt}}\vspace{5pt}\\
  {\sffamily\bfseries\fontsize{10pt}{12pt}\selectfont\textcolor{crimson}{SUPPORTING INFORMATION}}\vspace{4pt}\\
  {\sffamily\bfseries\fontsize{15pt}{18.2pt}\selectfont\textcolor{navyblue}{Scalable Contact-Gated Stochastic Operators and Universal Reduction Laws for Macromolecular Protein Dynamics, Allostery, and Kinetics}\par}
  \vspace{6pt}
  {\sffamily\bfseries\fontsize{9.8pt}{12pt}\selectfont Mehardeep Singh\textsuperscript{a,*} and Gurarpit Singh\textsuperscript{b}\par}
  \vspace{3pt}
  {\fontsize{8.4pt}{10.2pt}\selectfont\textcolor{darkgray}{\textsuperscript{a}Sri Chaitanya Educational Institutions, Chandigarh, India (\href{mailto:mehardeep.sim@gmail.com}{mehardeep.sim@gmail.com}); \textsuperscript{b}Vivekananda Institute of Professional Studies (VIPS), Guru Gobind Singh Indraprastha University (GGSIPU), Delhi, India (\href{mailto:gurarpit.sml@gmail.com}{gurarpit.sml@gmail.com})}\par}
  \vspace{4pt}
  \textcolor{navyblue}{\rule{\textwidth}{1pt}}
\end{center}

{\footnotesize\tableofcontents}
\vspace{6pt}
\noindent\textcolor{darkgray}{\rule{\textwidth}{0.4pt}}

\begin{table}[!htbp]
\centering
\caption*{Table S1. Mathematical nomenclature, coordinate definitions, and stochastic operator symbols}
\footnotesize
\renewcommand{\arraystretch}{1.14}
\begin{tabular}{@{}p{0.23\textwidth} p{0.73\textwidth}@{}}
\toprule[1.1pt]
\textbf{Symbol} & \textbf{Physical Definition and Units} \\
\midrule[0.6pt]
$\mathbb{T}^d = [-\pi,\pi)^d$ & Flat Clifford backbone torsion manifold of dimension $d = 2N_{\mathrm{res}}-2$ (uncapped proteins) or $d = 2N_{\mathrm{Ala}}$ (ACE/NME-capped model peptides; $\text{rad}^d$). \\
$\tilde{\bm{\theta}}(t) \in (-\pi,\pi)^d$ & Maximal-gap shifted circular coordinate vector ($\mathrm{dPCA}+$) centered away from each residue's Discovery-split steric eclipsing barrier $b_a$ ($\text{rad}$). \\
$\hat{\bm{\Sigma}}, \hat{\bm{\Omega}} = \hat{\bm{\Sigma}}^{-1}$ & Toroidal equilibrium covariance ($\text{rad}^2$) and sparse contact-gated precision matrix ($\text{rad}^{-2}$). \\
$\mathbf{D}(z), \mathbf{f}_{\mathrm{GENERIC}}(z)$ & Symmetric positive-definite short-time jump-diffusion tensor ($\text{rad}^2\,\text{ps}^{-1}$) and fluctuation--dissipation-consistent drift field $-\mathbf{D}(z)\nabla[\beta F(z)] + \nabla\!\cdot\!\mathbf{D}(z)$ ($\text{rad}\,\text{ps}^{-1}$). \\
$r_{80}, r_{90}, r_{95}, PR(d), N_{\mathrm{ent}}$ & Cumulative variance ranks ($80\%, 90\%, 95\%$), participation ratio $(\sum_k\lambda_k)^2/\sum_k\lambda_k^2$, and Shannon spectral entropy rank $\exp(-\sum_k p_k\ln p_k)$ (dimensionless). \\
$\bar{R}(\psi_i,\phi_{i+1})$ & Signed correlation between flanking torsions sharing rigid trans-peptide plane $i \to i+1$ (dimensionless). \\
$\mathcal{G} = \mathcal{E}_{\mathrm{band}} \cup \mathcal{E}_c$, $k_{\mathrm{eff}}$, $S_{0.05}$ & Chordal primary-sequence band ($|i-j|\le w$) augmented with bounded-degree non-local tertiary $C_\alpha$ contacts ($|i-j|\ge 4$, $\langle r_{ij}\rangle < r_c$, forming non-chordal $\mathcal{G}$), mean coupled partner degree, and negligible-coupling fraction ($|R_{ab}| < 0.05$). \\
$\mathrm{rSASA}_i$, $\gamma_{\mathrm{int}}$, $\eta_s$ & Tien et al.\ relative solvent-accessible surface area, Kramers internal protein friction ($\text{ps}^{-1}$), and bulk solvent viscosity ($\text{mPa}\cdot\text{s}$). \\
$\mathbf{J}_i \in \mathbb{R}^{3\times 4}$, $S^2_i$ & Parameter-free amide N--H unit-vector kinematic Jacobian $\partial\mathbf{u}_i/\partial(\psi_{i-1},\phi_i,\psi_i,\phi_{i+1})$ and Lipari--Szabo generalized order parameter $1 - \frac{3}{2}\Tr(\mathbf{J}_i\hat{\bm{\Sigma}}_{\mathrm{local},i}\mathbf{J}_i^\top)$. \\
$\gamma_M$, $\tau^*$, $q_+(Q)$, $\tau_{\mathrm{TP}}$ & Sub-diffusive Mori--Zwanzig memory attenuation exponent $D_{\mathrm{eff}}(\tau)\propto\tau^{-\gamma_M}$, calibrated Markovian lag ($\text{ps}$), forward committor probability, and mean reactive transition-path duration ($\text{ps}$). \\
$\dot{S}_{\mathrm{irr}}$ & Steady-state irreversible entropy production rate $\int \bm{J}_{\mathrm{ss}}^\top\mathbf{D}^{-1}\bm{J}_{\mathrm{ss}}/p_{\mathrm{ss}}\,dz$ ($k_B\,\text{ns}^{-1}\,d^{-1}$). \\
\bottomrule[1.1pt]
\end{tabular}
\end{table}

\clearpage
\section{Atomistic Simulation Protocols and Complete 45-Ensemble Cryptographic Registry}

\subsection{Force Fields, Explicit PME Solvation, Disulfide Topology, and Split Governance}
All \vFourTotalSystems{} molecular systems (\vFourNumProteins{} macromolecular proteins spanning $d = \vFourMinProteinDim{} \to \vFourMaxProteinDim{}$ followed by two capped analytical verification peptides at $d = 2, 4$; cumulative \vFourTotalAtlasFrames{} trajectory frames = $\vFourTotalSamplingNs{}\text{ ns}$) were integrated in OpenMM 8.4/8.6 under strict three-replica governance: Replica 1 (Discovery, independent random velocity seed and initial equilibration) for coordinate chart centering, tertiary contact-graph extraction, and operator fitting; Replica 2 (Validation) exclusively for hyperparameter selection; and Replica 3 (Blind Test) locked until final out-of-sample evaluation.

Explicit-solvent multi-domain and globular enzymes---HEW Lysozyme (\texttt{1AKI}, $d=\vFourDimLysozyme{}$, \vFourExplicitParticlesLysozyme{} atoms), \textit{E.\ coli} DHFR (\texttt{1RX2}, $d=\vFourDimDhfr{}$, \vFourExplicitParticlesDhfr{} atoms), Adenylate Kinase (\texttt{4AKE}, $d=\vFourDimAdk{}$, \vFourExplicitParticlesAdk{} atoms), and Maltose-Binding Protein (\texttt{1OMP}, $d=\vFourDimMbp{}$, \vFourMaxExplicitAtomsMbp{} atoms)---were parameterized with AMBER \texttt{ff99SB-ILDN} in rhombic dodecahedral TIP3P-FB water boxes ($1.2\text{ nm}$ solute-wall padding, $0.15\text{ M}$ NaCl counterions), neutral histidine protonation verified by local hydrogen-bond networks, Particle Mesh Ewald (PME) long-range electrostatics ($1.0\text{ nm}$ real-space cutoff, $10^{-4}$ Ewald error tolerance), and $4.0\text{ fs}$ \texttt{LangevinMiddleIntegrator} (BAOAB) integration ($\gamma = 1.0\text{ ps}^{-1}$, $300\text{ K}$, Monte Carlo barostat equilibration at $1.0\text{ atm}$) enabled by $1.5\text{ amu}$ hydrogen mass repartitioning (HMR) and rigid water/heavy-atom hydrogen bond constraints, saving coordinates every $1.0\text{ ps}$ ($3 \times 10\text{ ns} = 30{,}000$ frames per enzyme). While $1.5\text{ amu}$ HMR mildly slows sub-picosecond N--H and dihedral librational frequencies entering short-time $\hat{\mathbf{D}}(\tau = 1\text{ ps})$ prefactors, classical canonical configurational averages $\langle A(q)\rangle \propto \int A(q)e^{-\beta U(q)}\,dq$ decouple identically from momentum integrations $\int e^{-\beta \sum_i p_i^2/(2m_i)}\,dp$, rendering all stationary Boltzmann covariances $\hat{\bm{\Sigma}}$ (governing Laws 1, 2, and parameter-free Lipari--Szabo $S^2$ in Law 5) strictly invariant to mass repartitioning. Native covalent disulfide bridges were preserved intact throughout all trajectories: \texttt{Cys5}--\texttt{Cys55} ($0.2030\text{ nm}$), \texttt{Cys14}--\texttt{Cys38} ($0.2029\text{ nm}$), and \texttt{Cys30}--\texttt{Cys51} ($0.2023\text{ nm}$) in BPTI (\texttt{5PTI}); and \texttt{Cys6}--\texttt{Cys127} ($0.2033\text{ nm}$), \texttt{Cys30}--\texttt{Cys115} ($0.2030\text{ nm}$), \texttt{Cys64}--\texttt{Cys80} ($0.2038\text{ nm}$), and \texttt{Cys76}--\texttt{Cys94} ($0.2040\text{ nm}$) in HEW Lysozyme (\texttt{1AKI}). Matched Generalized Born implicit-solvent ensembles (GBn2 and OBC2 with AMBER \texttt{ff14SB}/\texttt{ff99SB-ILDN}: $3\times 50\text{ ns} = 150\text{ ns}$ reversible folding trajectories for Chignolin and Trp-cage; $3\times 10\text{ ns} = 30\text{ ns}$ GBn2 native/near-native ensembles for BBA5, Villin HP35, WW-FiP35, BPTI, and Barnase; $3\times 2.5\text{ ns} = 7.5\text{ ns}$ OBC2 native-basin ensembles for GB1 and Ubiquitin; plus capped dialanine/trialanine controls) were integrated at a $2.0\text{ fs}$ timestep ($1.0\text{ ps}$ saving stride at $300\text{ K}$). Table S2 summarizes native fold stability, inter-replica log-eigenvalue spectral convergence ($\bar{R}^2 = \vFourRepSpectralMeanRTwo{}$), and alternate crystal/NMR reference geometries; Table S3 records the complete 64-character SHA-256 digests across all 45 ensembles.

\begin{table}[!htbp]
\centering
\caption*{Table S2. Structural stability, inter-split log-eigenvalue convergence, and conformational reference pairs across all \vFourTotalSystems{} systems}
\resizebox{\textwidth}{!}{%
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{4.2pt}
\begin{tabular}{@{}l c c l c c c c c l@{}}
\toprule[1.1pt]
\textbf{System (Reference PDB)} & \textbf{$N_{\mathrm{res}}$} & \textbf{$d$} & \textbf{Solvent Protocol} & \textbf{$R_{g,\mathrm{nat}}$ (nm)} & \textbf{$\langle R_g\rangle$ (nm)} & \textbf{$\langle Q_{\mathrm{BHE}}\rangle$} & \textbf{$\langle\mathrm{RMSD}_{C_\alpha}\rangle$ (nm)} & \textbf{Spectral $R^2$} & \textbf{Alt.\ Structure $\mathrm{RMSD}_{C_\alpha}$ (nm)} \\
\midrule[0.6pt]
""" + "\n".join(t_s2_rows) + r"""
\bottomrule[1.1pt]
\end{tabular}%
}
\begin{flushleft}
{\scriptsize $^\dag$ACE-$\text{Ala}_n$-NME peptides carry $n$ central alanine residues flanked by acetyl (ACE) and $N$-methylamide (NME) caps, yielding $d = 2n$ rotatable $(\phi,\psi)$ torsions rather than uncapped $d = 2N_{\mathrm{res}}-2$. Spectral $R^2$ measures linear determination between $\ln\lambda_k$ spectra of independent Discovery and Validation splits.}
\end{flushleft}
\end{table}

{\scriptsize
\setlength{\tabcolsep}{3.4pt}
\renewcommand{\arraystretch}{1.16}
\begin{longtable}{@{}l c l l r r c c c l@{}}
\caption*{Table S3. Complete 45-ensemble trajectory registry and full 64-character SHA-256 cryptographic provenance digests} \\
\toprule[1.1pt]
\textbf{System (PDB)} & \textbf{$d$} & \textbf{Ensemble Split} & \textbf{Solvent} & \textbf{Frames} & \textbf{ns} & \textbf{$\langle Q_{\mathrm{BHE}}\rangle$} & \textbf{$\langle R_g\rangle$} & \textbf{$\langle\mathrm{RMSD}\rangle$} & \textbf{Full 64-Character Lead SHA-256 Digest} \\
\midrule[0.6pt]
\endfirsthead
\multicolumn{10}{@{}l}{\sffamily\bfseries\textcolor{navyblue}{Table S3 (continued). Complete 45-ensemble trajectory registry and full 64-character SHA-256 digests}} \\
\toprule[1.1pt]
\textbf{System (PDB)} & \textbf{$d$} & \textbf{Ensemble Split} & \textbf{Solvent} & \textbf{Frames} & \textbf{ns} & \textbf{$\langle Q_{\mathrm{BHE}}\rangle$} & \textbf{$\langle R_g\rangle$} & \textbf{$\langle\mathrm{RMSD}\rangle$} & \textbf{Full 64-Character Lead SHA-256 Digest} \\
\midrule[0.6pt]
\endhead
\midrule[0.5pt]
\multicolumn{10}{r@{}}{\textit{Continued on next page}} \\
\endfoot
\bottomrule[1.1pt]
\endlastfoot
""" + "\n".join(t_s3_rows) + r"""
\end{longtable}
}

\clearpage
\section{Mathematical Derivations and Proofs of Theorems S1--S7}

\subsection{Theorem S1: Isometric Maximal-Gap Toroidal Chart ($\mathrm{dPCA}+$) Eliminates Wrapping Variance}
\textbf{Proposition.} \textit{Let $\theta_a \in \mathbb{S}^1 \cong [-\pi,\pi)$ be a backbone torsion angle whose stationary Boltzmann density $p(\theta_a) \propto e^{-\beta F(\theta_a)}$ vanishes (up to exponentially rare $\mathcal{O}(e^{-\beta\Delta F_{\mathrm{steric}}})$ eclipsing transitions) on a contiguous steric-barrier arc $I_a = (b_a - \delta_a, b_a + \delta_a)$ of half-width $\delta_a > 0$. Define the Discovery-split maximal-gap shift $\tilde{\theta}_a = (\theta_a - b_a + \pi)\bmod 2\pi - \pi \in (-\pi,\pi)$. Then (i) $\tilde{\theta}_a$ is an orientation-preserving Riemannian isometry of $(\mathbb{S}^1, d_{\mathbb{S}^1})$, (ii) for all configurations in $\mathbb{S}^1 \setminus I_a$, Euclidean differences equal exact geodesic arc lengths $|\tilde{\theta}_a - \tilde{\theta}'_a| = d_{\mathbb{S}^1}(\theta_a,\theta'_a)$, and (iii) the sample covariance $\hat{\bm{\Sigma}}_{\tilde{\bm{\theta}}}$ preserves intrinsic manifold dimension $d$ without the $2\pi$ discontinuity inflation of raw $[-\pi,\pi)$ coordinates or the $2d$ rank doubling of $(\cos\theta_a,\sin\theta_a)$ embeddings.}

\textbf{Proof.} Sorting the $N$ Discovery-split observations $\theta_a^{(1)} \le \dots \le \theta_a^{(N)}$ on $\mathbb{S}^1$ with cyclic wrap $\theta_a^{(N+1)} \equiv \theta_a^{(1)} + 2\pi$ yields adjacent gaps $\Delta_k = \theta_a^{(k+1)} - \theta_a^{(k)}$. Under $\beta\Delta F_{\mathrm{steric}} \gg 1$ (e.g., $\phi > 0$ eclipsing barrier in non-glycine L-residues or $\omega$ peptide planarity), no trajectory point enters $I_a$, so $\max_k \Delta_k \ge 2\delta_a$ straddles $b_a = \frac{1}{2}(\theta_a^{(k^*)} + \theta_a^{(k^*+1)}) \bmod 2\pi$. Because $T_{b_a}(\theta) = (\theta - b_a + \pi)\bmod 2\pi - \pi$ translates the branch cut of $\mathbb{S}^1$ precisely to $b_a$, its differential inside $\mathbb{S}^1\setminus\{b_a\}$ is $dT_{b_a}/d\theta = 1$, proving metric isometry. Any two points $\theta,\theta' \in \mathbb{S}^1\setminus I_a$ map into $(-\pi+\delta_a,\pi-\delta_a)$, where no shortest geodesic crosses the cut $\pm\pi$; hence $\tilde{\theta} - \tilde{\theta}' = \Delta_{\mathbb{T}^1}(\theta,\theta')$. By contrast, if the native basin straddles $\pm\pi$ in unshifted coordinates with split probability $q \in (0,1)$, raw Euclidean variance acquires a spurious additive jump $4\pi^2 q(1-q) = \mathcal{O}(10\text{ rad}^2)$, inflating $r_{90}$ artifactually, whereas $(\cos\theta_a,\sin\theta_a)\in\mathbb{R}^{2d}$ embeds each $\mathbb{S}^1$ factor curvedly into $\mathbb{R}^2$, splitting large-amplitude unimodal torsions across orthogonal cosine and sine axes ($r_{90}^{\sin/\cos} > r_{90}^{\mathrm{dPCA}+}$ across every protein in Table S4). $\blacksquare$

\subsection{Theorem S2: Importance-Weighted Variational Mutual Information Bound Under Prior Shift}
\textbf{Proposition.} \textit{Let $p_{\mathrm{train}}(x,y)$ and $p_{\mathrm{test}}(x,y)$ denote joint densities over reduced coordinates $x \in \mathbb{R}^r$ and conformational macrostates $y \in \{1,\dots,K\}$ satisfying the label-shift invariance $p_{\mathrm{test}}(x\mid y) = p_{\mathrm{train}}(x\mid y)$ with shifted macrostate populations $p_{\mathrm{test}}(y) \ne p_{\mathrm{train}}(y)$. For any classifier $q_\vartheta(y\mid x)$ trained on Split 1, define the prior-corrected posterior $\tilde{q}_\vartheta(y\mid x) = Z(x)^{-1} q_\vartheta(y\mid x)\,[p_{\mathrm{test}}(y)/p_{\mathrm{train}}(y)]$ with $Z(x) = \sum_{y'=1}^K q_\vartheta(y'\mid x)\,[p_{\mathrm{test}}(y')/p_{\mathrm{train}}(y')]$. Then}
\[
I_{\mathrm{var}}(X;Y) \coloneqq H_{\mathrm{test}}(Y) + \mathbb{E}_{p_{\mathrm{test}}(x,y)}\!\big[\ln \tilde{q}_\vartheta(y\mid x)\big] \le I_{\mathrm{test}}(X;Y) \le H_{\mathrm{test}}(Y),
\]
\textit{with equality if and only if $\tilde{q}_\vartheta(y\mid x) = p_{\mathrm{test}}(y\mid x)$ almost everywhere.}

\textbf{Proof.} By definition of mutual information on the test ensemble,
\[
I_{\mathrm{test}}(X;Y) - I_{\mathrm{var}}(X;Y) = \mathbb{E}_{p_{\mathrm{test}}(x,y)}\!\left[\ln\frac{p_{\mathrm{test}}(y\mid x)}{\tilde{q}_\vartheta(y\mid x)}\right] = \mathbb{E}_{p_{\mathrm{test}}(x)}\!\Big[D_{\mathrm{KL}}\!\big(p_{\mathrm{test}}(\cdot\mid x)\,\big\|\,\tilde{q}_\vartheta(\cdot\mid x)\big)\Big] \ge 0,
\]
because $\tilde{q}_\vartheta(\cdot\mid x)$ is strictly normalized ($\sum_y \tilde{q}_\vartheta(y\mid x) = 1$) for every $x$, making the conditional Kullback--Leibler divergence non-negative by Gibbs' inequality. Moreover, under Bayes' rule and $p_{\mathrm{test}}(x\mid y) = p_{\mathrm{train}}(x\mid y)$, when $q_\vartheta(y\mid x) = p_{\mathrm{train}}(y\mid x)$, $\tilde{q}_\vartheta(y\mid x) \propto p_{\mathrm{train}}(x\mid y)p_{\mathrm{test}}(y) \propto p_{\mathrm{test}}(y\mid x)$, recovering the exact Bayes-optimal test mutual information without refitting parameters on Split 3, while $I_{\mathrm{test}}(X;Y) = H_{\mathrm{test}}(Y) - H_{\mathrm{test}}(Y\mid X) \le H_{\mathrm{test}}(Y)$ holds universally for discrete macrostates. $\blacksquare$

\subsection{Theorem S3: Chordal Sequence Bands, Directed Acyclic Contact-Graph Cholesky, and Contact-Masked Shrinkage}
\textbf{Proposition.} \textit{Let $\mathcal{E}_{\mathrm{band}}(w) = \{(a,b): |\mathrm{res}(a)-\mathrm{res}(b)|\le w\}$ be the primary-sequence band of residue half-width $w$ (bandwidth $B = 2w+1$ residues $= 4w+2$ torsions), and $\mathcal{E}_c(r_c) = \{(a,b): |\mathrm{res}(a)-\mathrm{res}(b)|\ge 4,\; \langle r_{C_\alpha,\mathrm{res}(a),\mathrm{res}(b)}\rangle_{\mathrm{Split1}} < r_c\}$ be the Split 1 native tertiary contact graph. Then: (i) $\mathcal{E}_{\mathrm{band}}(w)$ is a chordal (triangulated) graph admitting zero-fill-in banded Cholesky completion in $\mathcal{O}(d\,B^2)$ flops and $\mathcal{O}(d\,B)$ memory; (ii) although $\mathcal{G} = \mathcal{E}_{\mathrm{band}}(w) \cup \mathcal{E}_c(r_c)$ is non-chordal due to long sequence cycles across tertiary contacts, directing edges along sequence order ($b < a$, bounded by $k_{\max} = \mathcal{O}(1)$ sequence-band plus closest tertiary-contact predecessors) yields an exact symmetric positive-definite sparse directed acyclic Cholesky precision estimator $\hat{\bm{\Omega}}_{\mathcal{G}} = (\mathbf{I}-\mathbf{B})^\top\mathbf{D}_{\mathrm{cond}}^{-1}(\mathbf{I}-\mathbf{B})$ in $\mathcal{O}(d\,k_{\max}^3) = \mathcal{O}(d)$ flops and $\mathcal{O}(d\,k_{\max})$ memory without iterative chordal maximum-likelihood completion; and (iii) for $\beta_c \in (0,1]$ and $\varepsilon > 0$, the contact-masked linear shrinkage mobility operator $\hat{\mathbf{D}}_{\mathcal{G}}(\beta_c) = (1-\beta_c)\mathcal{P}_{\mathcal{G}}(\hat{\mathbf{D}}) + \beta_c\diag(\hat{\mathbf{D}}) + \varepsilon\mathbf{I}$ has storage $\mathcal{O}(d\,\bar{k}_{\mathrm{eff}})$ and is strictly SPD whenever $\beta_c \ge \beta_{\min}(\hat{\mathbf{D}},\mathcal{G})$.}

\textbf{Proof.} Along primary sequence order $1,\dots,d$, higher-indexed neighbors of $a$ in $\mathcal{E}_{\mathrm{band}}(w)$ form a contiguous clique $\{a+1,\dots,\min(d,a+4w+1)\}$, so $\mathcal{E}_{\mathrm{band}}(w)$ is chordal with zero Gaussian-elimination fill-in. Adding non-local tertiary contacts $\mathcal{E}_c(r_c)$ introduces chordless sequence cycles, making undirected $\mathcal{G}$ non-chordal; however, 3D steric sphere packing bounds the $C_\alpha$ coordination number inside radius $r_c \le 0.85\text{ nm}$ by $z_{\max} \le (r_c/r_{\mathrm{vdW}})^3 = \mathcal{O}(1)$ independent of $N_{\mathrm{res}}$. Regressing each coordinate $z_a$ via ridge-regularized least squares on its active predecessor set $P(a) \subset \{1,\dots,a-1\}$ ($|P(a)| \le k_{\max}$, $\mathcal{O}(k_{\max}^3)$ per row) produces a strictly lower-triangular sparse factor $(\mathbf{I}-\mathbf{B})$ and positive conditional variances $D_{\mathrm{cond},aa} > 0$, guaranteeing $\hat{\bm{\Omega}}_{\mathcal{G}} = (\mathbf{I}-\mathbf{B})^\top\mathbf{D}_{\mathrm{cond}}^{-1}(\mathbf{I}-\mathbf{B}) \succ 0$ with exact log-determinant $\ln\det\hat{\bm{\Omega}}_{\mathcal{G}} = -\sum_{a=1}^d \ln D_{\mathrm{cond},aa}$ in $\mathcal{O}(d)$ total time and memory (empirically $\bar{k}_{\mathrm{eff}} = \vFourKeffMean{}$, memory $\mathcal{O}(d^{\vFourMemExpContact{}})$, time $\mathcal{O}(d^{\vFourTimeExpChordal{}})$). Finally, writing $\mathbf{S} = \diag(\hat{\mathbf{D}})^{-1/2}\mathcal{P}_{\mathcal{G}}(\hat{\mathbf{D}})\diag(\hat{\mathbf{D}})^{-1/2}$ with unit diagonal, $\hat{\mathbf{D}}_{\mathcal{G}}(\beta_c) = \diag(\hat{\mathbf{D}})^{1/2}[(1-\beta_c)\mathbf{S} + \beta_c\mathbf{I}]\diag(\hat{\mathbf{D}})^{1/2} + \varepsilon\mathbf{I} \succ 0$ whenever $(1-\beta_c)\lambda_{\min}(\mathbf{S}) + \beta_c \ge 0$, enforced by Split 2 validation. $\blacksquare$

\subsection{Theorem S4: Exact Detailed Balance ($\dot{S}_{\mathrm{irr}} \equiv 0$), Hurwitz Stability, and Two-Tier GENERIC Realization}
\textbf{Proposition.} \textit{Consider the It\^o SDE $dz_t = \mathbf{f}(z_t)\,dt + \sqrt{2\mathbf{D}(z_t)}\,d\mathbf{W}_t$ on smooth compact manifold $\mathbb{T}^d$ with smooth SPD mobility $\mathbf{D}(z) \succ 0$ and potential of mean force $F(z) = -k_B T \ln p_{\mathrm{eq}}(z)$. Setting $\mathbf{f}_{\mathrm{GENERIC}}(z) = -\mathbf{D}(z)\nabla[\beta F(z)] + \nabla\!\cdot\!\mathbf{D}(z)$, where $(\nabla\!\cdot\!\mathbf{D})_a = \sum_{b=1}^d \partial_{z_b} D_{ab}(z)$, satisfies: (i) stationary Fokker--Planck invariance $\mathcal{L}^\dagger p_{\mathrm{eq}} = 0$ with identically vanishing probability current $\bm{J}_{\mathrm{ss}}(z) \equiv \mathbf{0}$ and irreversible entropy production rate $\dot{S}_{\mathrm{irr}} \equiv 0$; (ii) self-adjointness of $\mathcal{L}$ in $L^2(p_{\mathrm{eq}})$ with real non-positive spectrum; and (iii) in full $d = 18 \to 738$ macromolecular native basins parameterized by spatially homogeneous $\hat{\mathbf{D}}_{\mathcal{G}} \succ 0$ ($\nabla\!\cdot\!\hat{\mathbf{D}}_{\mathcal{G}} \equiv \mathbf{0}$) and quadratic free energy $\beta F(z) = \frac{1}{2}(z-\mu)^\top\hat{\bm{\Sigma}}_{\mathcal{G}}^{-1}(z-\mu)$, global reduction to a reversible linear Ornstein--Uhlenbeck GENERIC process with strictly Hurwitz-stable generator $\mathbf{A} = -\hat{\mathbf{D}}_{\mathcal{G}}\hat{\bm{\Sigma}}_{\mathcal{G}}^{-1}$ ($\lambda_k(\mathbf{A}) \in \mathbb{R}_{<0}$).}

\textbf{Proof.} Under the It\^o convention on $\mathbb{T}^d$, the Fokker--Planck continuity equation reads $\partial_t p = -\nabla\!\cdot\!\bm{J}(p)$ with probability flux $J_a(p) = f_a(z)p(z) - \sum_{b=1}^d \partial_{z_b}[D_{ab}(z)p(z)]$. Substituting $p_{\mathrm{eq}}(z) = Z^{-1}e^{-\beta F(z)}$ and $f_{\mathrm{GENERIC},a}(z) = -\sum_b D_{ab}\partial_{z_b}(\beta F) + \sum_b \partial_{z_b} D_{ab}$ gives
\[
J_a(p_{\mathrm{eq}}) = \left(-\sum_b D_{ab}\partial_{z_b}(\beta F) + \sum_b \partial_{z_b} D_{ab}\right)p_{\mathrm{eq}} - \sum_b (\partial_{z_b} D_{ab})p_{\mathrm{eq}} + \sum_b D_{ab}\partial_{z_b}(\beta F)\,p_{\mathrm{eq}} \equiv 0
\]
pointwise on $\mathbb{T}^d$. Consequently, steady-state Schnakenberg--Seifert entropy production vanishes identically:
\[
\dot{S}_{\mathrm{irr}} = k_B \int_{\mathbb{T}^d} \bm{J}_{\mathrm{ss}}(z)^\top \mathbf{D}(z)^{-1} \bm{J}_{\mathrm{ss}}(z)\,p_{\mathrm{eq}}(z)^{-1}\,dz \equiv 0,
\]
whereas unconstrained basis/polynomial drift regressions $\delta\mathbf{f}(z)$ with non-zero curl $\nabla \times (\mathbf{D}^{-1}\delta\mathbf{f}) \ne 0$ generate persistent vortex currents and $\dot{S}_{\mathrm{irr}} > 0$ (median $\vFourSOTAUnconstEntropyMedian{}\,k_B/\text{ns}/d$, IQR $\vFourSOTAUnconstEntropyQOne{}\text{--}\vFourSOTAUnconstEntropyQThree{}$, mean $\vFourSOTAUnconstEntropyMean{}$ empirically). Integration by parts yields $\langle g, \mathcal{L}h\rangle_{p_{\mathrm{eq}}} = -\int (\nabla g)^\top \mathbf{D}(z)(\nabla h)\,p_{\mathrm{eq}}\,dz = \langle \mathcal{L}g, h\rangle_{p_{\mathrm{eq}}} \le 0$. Because nonparametric estimation of a position-dependent $d\times d$ tensor field $\mathbf{D}(z)$ scales exponentially as $\mathcal{O}(M^d)$, full nonlinear position-dependent $\mathbf{D}(q)$, anharmonic multi-well $\beta F(q)$, and $\nabla\!\cdot\!\mathbf{D}(q)$ are evaluated on projected 1D/2D reaction coordinates ($Q_{\mathrm{BHE}}, R_g$) and low-dimensional $d=2,4$ peptide manifolds, while on the $d = 18 \to 738$ dPCA+-unwrapped chart setting constant SPD $\hat{\mathbf{D}}_{\mathcal{G}}$ ($\nabla\!\cdot\!\hat{\mathbf{D}}_{\mathcal{G}} \equiv \mathbf{0}$) and $\mathbf{H} = \hat{\bm{\Sigma}}_{\mathcal{G}}^{-1} \succ 0$ gives linear drift $\mathbf{f}_{\mathrm{GENERIC}}(z) = \mathbf{A}(z-\mu)$ with $\mathbf{A} = -\hat{\mathbf{D}}_{\mathcal{G}}\mathbf{H} = -\mathbf{H}^{-1/2}(\mathbf{H}^{1/2}\hat{\mathbf{D}}_{\mathcal{G}}\mathbf{H}^{1/2})\mathbf{H}^{1/2}$, which is similar to a negative-definite symmetric matrix and thus globally Hurwitz stable ($\lambda_k(\mathbf{A}) \in \mathbb{R}_{<0}$) with exact stationary Gaussian covariance $\hat{\bm{\Sigma}}_{\mathcal{G}}$ satisfying Lyapunov detailed balance $\mathbf{A}\hat{\bm{\Sigma}}_{\mathcal{G}} + \hat{\bm{\Sigma}}_{\mathcal{G}}\mathbf{A}^\top = -2\hat{\mathbf{D}}_{\mathcal{G}}$. $\blacksquare$

\subsection{Theorem S5: Kramers--Stokes--Einstein Solvent-Shielding Law via Relative SASA}
\textbf{Proposition.} \textit{Model residue $i$'s local torsional dissipation $\gamma_{\mathrm{tot},i}(T,\eta_s)$ as additive internal protein friction $\gamma_{\mathrm{int},i}$ (steric packing and hydrogen-bond cage collisions) plus solvent hydrodynamic drag proportional to exposed relative solvent-accessible surface area $\mathrm{rSASA}_i \in [0,1]$: $\gamma_{\mathrm{tot},i}(T,\eta_s) = \gamma_{\mathrm{int},i} + c_h\,\mathrm{rSASA}_i\,\eta_s(T)$. By Einstein's fluctuation--dissipation relation $D_i(T,\eta_s) = k_B T / \gamma_{\mathrm{tot},i}(T,\eta_s)$:}
\[
D_i(T,\eta_s) = \frac{k_B T}{\gamma_{\mathrm{int},i} + c_h\,\mathrm{rSASA}_i\,\eta_s(T)}, \qquad \alpha_i \coloneqq \frac{\partial \ln D_i}{\partial \ln T} = 1 + \frac{c_h\,\mathrm{rSASA}_i\,\eta_s(T)}{\gamma_{\mathrm{int},i} + c_h\,\mathrm{rSASA}_i\,\eta_s(T)}\left(-\frac{\partial \ln \eta_s}{\partial \ln T}\right).
\]
\textbf{Proof.} Taking the logarithmic derivative of $D_i(T,\eta_s(T))$ with respect to $\ln T$ yields $\partial_{\ln T}\ln D_i = 1 - \partial_{\ln T}\ln[\gamma_{\mathrm{int},i} + c_h\,\mathrm{rSASA}_i\,\eta_s(T)]$. Defining the hydrodynamic coupling fraction $\phi_{\mathrm{solv},i} \coloneqq c_h\,\mathrm{rSASA}_i\,\eta_s / (\gamma_{\mathrm{int},i} + c_h\,\mathrm{rSASA}_i\,\eta_s) \in [0,1]$ and liquid water's Arrhenius viscosity activation $E_\eta/(k_B T) = -\partial_{\ln T}\ln\eta_s > 0$ ($\approx 6.0$ near $300\text{ K}$) gives $\alpha_i = 1 + \phi_{\mathrm{solv},i}[E_\eta/(k_B T)]$. In deeply buried hydrophobic cores ($\mathrm{rSASA}_i < 0.10$, $\gamma_{\mathrm{int},i} \gg c_h\,\mathrm{rSASA}_i\,\eta_s$), $\phi_{\mathrm{solv},i} \to 0$ and $\alpha_{\mathrm{core}} \to 1$ ($\vFourAlphaBuriedCore{}$ empirically, parameter-free linear Einstein scaling). On solvent-exposed loops ($\mathrm{rSASA}_i > 0.35$) and fully solvated short peptides ($\mathrm{rSASA}_i \to 1$), $\phi_{\mathrm{solv},i}$ rises monotonically toward unity, elevating $\alpha_{\mathrm{loop}} = \vFourAlphaSurfaceLoop{}$ and $\alpha_{\mathrm{peptide}} = \vFourAlphaShortPeptide{}$ within one unified equation. $\blacksquare$

\subsection{Theorem S6: Kinematic Jacobian Projection onto NMR $S^2$ and Crankshaft Cancellation}
\textbf{Proposition.} \textit{Let $\mathbf{u}_i(\bm{\theta}) \in \mathbb{S}^2$ ($\|\mathbf{u}_i\|=1$) denote the amide $\mathrm{N}_i\text{--}\mathrm{H}_i$ bond vector of residue $i$ expressed in the molecular Eckart frame, governed locally to leading order by the flanking torsion quartet $\bm{\vartheta}_i = (\psi_{i-1},\phi_i,\psi_i,\phi_{i+1})^\top \in \mathbb{R}^4$ with local covariance $\bm{\Sigma}_{\mathrm{local},i} = \mathrm{Cov}(\bm{\vartheta}_i)$ and kinematic Jacobian $\mathbf{J}_i = \partial\mathbf{u}_i/\partial\bm{\vartheta}_i\big|_{\bm{\vartheta}_{i,0}} \in \mathbb{R}^{3\times 4}$. Up to $\mathcal{O}(\|\Delta\bm{\vartheta}_i\|^4)$, the isotropic Lipari--Szabo generalized order parameter satisfies}
\[
S^2_i \coloneqq \frac{3}{2}\sum_{\alpha,\beta=1}^3 \langle u_{i,\alpha} u_{i,\beta}\rangle^2 - \frac{1}{2} = 1 - \frac{3}{2}\Tr\!\big(\mathbf{J}_i\bm{\Sigma}_{\mathrm{local},i}\mathbf{J}_i^\top\big) + \mathcal{O}\!\big(\|\Delta\bm{\vartheta}_i\|^4\big),
\]
\textit{and peptide-plane crankshaft anti-correlation $\Sigma_{\psi_{i-1},\phi_i} < 0$ strictly increases $S^2_i$ relative to independent diagonal fluctuations whenever $\mathbf{J}_{i,1}\cdot\mathbf{J}_{i,2} > 0$.}

\textbf{Proof.} Because $\|\mathbf{u}_i(\bm{\vartheta})\|^2 = 1$ identically on $\mathbb{S}^2$, differentiating twice around the equilibrium native geometry $\mathbf{u}_{i,0}$ gives $\mathbf{u}_{i,0}^\top \mathbf{J}_i = \mathbf{0}^\top$ and $\mathbf{u}_i = \mathbf{u}_{i,0} + \mathbf{J}_i\Delta\bm{\vartheta}_i + \frac{1}{2}\mathbf{r}_i(\Delta\bm{\vartheta}_i) + \mathcal{O}(\|\Delta\bm{\vartheta}_i\|^3)$ with $\mathbf{u}_{i,0}^\top\mathbf{r}_i(\Delta\bm{\vartheta}_i) = -\|\mathbf{J}_i\Delta\bm{\vartheta}_i\|^2$. Taking ensemble expectations with $\langle\Delta\bm{\vartheta}_i\rangle = \mathbf{0}$ yields $\langle u_{i,\alpha} u_{i,\beta}\rangle = u_{i,0,\alpha}u_{i,0,\beta}\big(1 - \langle\|\mathbf{J}_i\Delta\bm{\vartheta}_i\|^2\rangle\big) + (\mathbf{J}_i\bm{\Sigma}_{\mathrm{local},i}\mathbf{J}_i^\top)_{\alpha\beta} + \mathcal{O}(\|\Delta\bm{\vartheta}_i\|^4)$. Squaring and summing over $\alpha,\beta \in \{1,2,3\}$ while using $\mathbf{u}_{i,0}^\top\mathbf{J}_i = \mathbf{0}^\top$ cancels cross terms between $\mathbf{u}_{i,0}\mathbf{u}_{i,0}^\top$ and $\mathbf{J}_i\bm{\Sigma}_{\mathrm{local},i}\mathbf{J}_i^\top$ at order $\mathcal{O}(\|\Delta\bm{\vartheta}_i\|^2)$, leaving $\sum_{\alpha,\beta}\langle u_{i,\alpha}u_{i,\beta}\rangle^2 = 1 - 2\langle\|\mathbf{J}_i\Delta\bm{\vartheta}_i\|^2\rangle + \mathcal{O}(\|\Delta\bm{\vartheta}_i\|^4)$, whence $S^2_i = 1 - \frac{3}{2}\Tr(\mathbf{J}_i\bm{\Sigma}_{\mathrm{local},i}\mathbf{J}_i^\top)$. Expanding the trace over the $(\psi_{i-1},\phi_i)$ peptide-plane block gives $\Tr(\mathbf{J}_i\bm{\Sigma}_{\mathrm{local},i}\mathbf{J}_i^\top) = \sum_{a=1}^4 \|\mathbf{J}_{i,a}\|^2 \Sigma_{aa} + 2\mathbf{J}_{i,1}\cdot\mathbf{J}_{i,2}\,\Sigma_{\psi_{i-1},\phi_i} + \dots$; because rigid trans-peptide rotation tilts $\mathbf{u}_i$ co-directionally under $+\Delta\psi_{i-1}$ and $+\Delta\phi_i$ ($\mathbf{J}_{i,1}\cdot\mathbf{J}_{i,2} > 0$) while compensatory crankshaft motion enforces $\Sigma_{\psi_{i-1},\phi_i} < 0$ ($\bar{R} = \vFourMeanPeptideCrankshaftR{}$), the off-diagonal covariance cancels unphysical amide libration and lifts mean per-protein NMR $S^2$ rank correlation from $\vFourNMRMeanProteinSpearmanFlank{}$ to $\vFourNMRMeanProteinSpearmanCrank{}$. $\blacksquare$

\subsection{Theorem S7: Mori--Zwanzig Power-Law Memory Attenuation and Adjoint Committor Kinetics}
\textbf{Proposition.} \textit{Projecting $d$-dimensional rough-landscape dynamics onto reaction coordinate $Q(t) \in [0,1]$ yields a generalized Langevin equation with scale-free orthogonal barrier spectrum $K_{\mathrm{orth}}(t) \sim K_0 t^{-(1-\alpha_K)}$ ($0 < \alpha_K < 1$) on intermediate lags $\tau_{\mathrm{micro}} \ll \tau \ll \tau_{\mathrm{commit}}$. Then (i) the finite-lag effective diffusivity attenuates algebraically as $D_{\mathrm{eff}}(\tau) = \langle\Delta Q(\tau)^2\rangle/(2\tau) \propto \tau^{-\gamma_M}$ with $\gamma_M = 1 - \alpha_K \in (0,1)$; and (ii) at the calibrated coarse-graining lag $\tau^*$ matching orthogonal barrier relaxation, the reversible 1D Smoluchowski generator $\mathcal{L}g = e^{\beta F(Q)}\partial_Q[D(Q)e^{-\beta F(Q)}\partial_Q g]$ yields exact closed-form committors, MFPTs, and reactive transition-path durations $\tau_{\mathrm{TP}}$.}

\textbf{Proof.} By the second fluctuation--dissipation theorem in Laplace space, velocity autocorrelation $\hat{C}_v(s) = k_B T / [m s + \hat{K}_{\mathrm{orth}}(s)]$. For $K_{\mathrm{orth}}(t) \propto t^{-(1-\alpha_K)}$, Tauberian inversion gives $\hat{K}_{\mathrm{orth}}(s) \propto s^{-\alpha_K}$ as $s\to 0$, so mean-square displacement $\langle\Delta Q(\tau)^2\rangle = \mathcal{L}^{-1}[2\hat{C}_v(s)/s^2](\tau) \propto \tau^{1-\gamma_M}$ with $\gamma_M = 1-\alpha_K$, whence $D_{\mathrm{eff}}(\tau) = \langle\Delta Q(\tau)^2\rangle/(2\tau) \propto \tau^{-\gamma_M}$ (empirically $\bar{\gamma}_M = \vFourMemoryGammaMean{} \pm \vFourMemoryGammaSD{}$, $R^2 > 0.99$ across $\tau \in [1,400]\text{ ps}$). Once evaluated at $\tau^*$ where orthogonal non-Markovian modes have relaxed inside metastable cores $(Q_U, Q_F)$, solving $\mathcal{L}q_+(Q) = 0$ with $q_+(Q \le Q_U)=0, q_+(Q \ge Q_F)=1$ and $\mathcal{L}t_{UF}(Q) = -1$ with reflecting boundary at $Q_{\min}$ and absorbing boundary at $Q_F$ yields standard quadratures $q_+(Q) = \int_{Q_U}^Q [D(x)p_{\mathrm{eq}}(x)]^{-1}dx \big/ \int_{Q_U}^{Q_F} [D(x)p_{\mathrm{eq}}(x)]^{-1}dx$, reactive steady flux $J_{UF} = \big[\int_{Q_U}^{Q_F} [D(x)p_{\mathrm{eq}}(x)]^{-1}dx\big]^{-1}$, and Hummer--Szabo mean transition-path time $\tau_{\mathrm{TP}} = (2J_{UF})^{-1}\int_{Q_U}^{Q_F} 2q_+(x)[1-q_+(x)]p_{\mathrm{eq}}(x)\,dx$ obeying microscopic detailed balance to machine precision ($<10^{-16}$). $\blacksquare$
"""
    return tex
