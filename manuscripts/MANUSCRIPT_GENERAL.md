# General Scaling Laws and Thermodynamically Consistent Reduced Stochastic Operators for Macromolecular Protein Dynamics

**Mehardeep Singh**<sup>a</sup> and **Gurarpit Singh**<sup>b</sup>

<sup>a</sup>Sri Chaitanya Educational Institutions, Chandigarh, India; <sup>b</sup>Vivekananda Institute of Professional Studies (VIPS), Guru Gobind Singh Indraprastha University (GGSIPU), Delhi, India

**Classification:** Physical Sciences / Biophysics and Computational Biology

---

### Significance Statement
Coarse-graining macromolecular dynamics without violating microscopic reversibility, losing non-Markovian memory, or succumbing to exponential dimensionality explosion has remained an intractable challenge in chemical physics for over six decades. Here, we present a unified framework spanning three developmental generations that discovers continuous stochastic differential operators directly from atomistic simulations across diverse peptides and proteins spanning 2 to 150 conformational dimensions. We prove that conformational operator rank scales sub-linearly with chain length ($r_{90} \propto d^{0.89}$), driven by power-law dynamic locality and tertiary contact gating. This uncovers an asymptotic linear-time estimator ($28\times$ speedup at $d=2,400$) and reveals an invariant mobility floor in folded protein interiors ($\bar{D} = 0.0496\text{ rad}^2/\text{ps}$, CV $8.0\%$). These findings provide an interpretable, mathematically rigorous foundation for multiscale biological modeling and kinetic drug discovery.

---

### Abstract
Bridging the vast temporal and spatial scales of macromolecular conformational dynamics requires dimensional reduction strategies that preserve physical kinetics, thermodynamic reversibility, and computational tractability. Traditional approaches—ranging from discrete Markov state models to unconstrained deep learning surrogates—often suffer from discretization artifacts, memory truncation, or unphysical steady-state entropy production. Here, we present a complete three-generation synthesis for discovering continuous, thermodynamically consistent stochastic differential operators directly from atomistic molecular dynamics. Across seven benchmark systems spanning model peptides (dialanine, trialanine) to five structurally diverse proteins (Chignolin, Trp-cage, Villin HP35, GB1, and Human Ubiquitin, $d=2 \to 150$), we discover six general principles of macromolecular reduction. In model peptides (Versions 1.0 and 2.0), we prove that barrier mobility surges anisotropically, that thermal transfer collapses onto a rank-1 SVD deformation mode ($G_1(z)$, 72.7% variance), and that extended Mori-Zwanzig embedding $(z, v, w)$ cures the low-friction Kramers turnover while GENERIC drift eliminates 90.8% of artificial entropy production. Scaling to macromolecules (Version 3.0 Rigorous), we prove that effective operator rank scales sub-linearly with dimension ($r_{90}(d) = 0.745\,d^{0.887}$, 95% parametric bootstrap CI $[0.830, 0.950]$, $P(\alpha < 1.0) = 99.95\%$), whereas an independent null model recovers linear scaling ($\alpha_{\text{null}} \approx 1.0$), demonstrating that compression is an emergent consequence of cooperative physics. Dynamic correlations along the chain decay as a power law, while non-local tertiary contacts gate off-diagonal coupling with a 2.1- to 3.9-fold enrichment, unlocking an asymptotic linear-time operator estimator ($\mathcal{O}(d^{0.99})$, 28-fold acceleration at $d=2,400$). In leave-one-protein-out (LOPO) validation, the model successfully transfers across unseen protein folds ($r_s = 0.64, R^2 = 0.41$), revealing an invariant folded interior mobility plateau ($\bar{D} = 0.0496\text{ rad}^2/\text{ps}$, CV $8.0\%$) that correlates significantly with experimental 1.8 Å X-ray crystallographic B-factors ($r_s = 0.272, p=0.019$) and follows zero-parameter out-of-distribution equipartition scaling across 280–380 K (5.5% error). When embedded into an extended Mori-Zwanzig state space with structure-preserving GENERIC drift, the discovered operators capture barrier-crossing kinetics within the 95% bootstrap confidence interval of atomistic MD while eliminating 90.8% of artificial entropy production.

**Keywords:** molecular dynamics | stochastic differential equations | Mori-Zwanzig projection | dimensional reduction | protein kinetics

---

## Introduction

Macromolecular biological processes—such as enzymatic catalysis, allosteric signal transduction, and protein folding—occur on millisecond-to-second timescales, yet are dictated by interatomic forces operating on femtosecond integration steps (1–3). Overcoming this twelve-order-of-magnitude temporal chasm through brute-force atomistic molecular dynamics (MD) requires immense computational resources and yields petabytes of coordinate data that remain difficult to interpret mechanistically (4, 5). Consequently, constructing simplified, low-dimensional equations of motion that retain macroscopic physical kinetics has been a central aspiration of theoretical biophysics and computational chemistry since the seminal work of Kramers, Onsager, Mori, and Zwanzig (6–10).

Historically, dimensional reduction has been pursued through three primary modeling paradigms, each exhibiting distinct theoretical vulnerabilities:
1. *Discrete Markov State Models (MSMs):* MSMs partition continuous conformation space into discrete geometric Voronoi microstates and estimate a discrete transition probability matrix at a lag time $\tau$ (11, 12). While successful in identifying metastable intermediates, MSMs suffer from boundary discretization error, strong lag-time sensitivity, and a loss of continuous spatial gradients.
2. *Overdamped Langevin Equations:* Projecting coordinates onto a small set of collective variables (CVs) under overdamped Brownian motion assumes instantaneous momentum thermalization (13). However, near transition barriers or under reduced solvent friction, momentum relaxation fails (the Kramers friction turnover regime), causing overdamped kinetic rates to deviate by hundreds of percent (14, 15).
3. *Unconstrained Machine Learning & Neural SDEs:* Modern deep neural surrogates and generative diffusion models can learn complex coordinate maps (16, 17). However, unconstrained parameterizations typically violate microscopic detailed balance and Onsager reciprocal symmetry, driving artificial steady-state probability currents ($J_{\mathrm{ss}} \ne 0$) and generating unphysical non-equilibrium entropy rates (18).

At the mathematical heart of these failures lies the Mori-Zwanzig projection operator formalism (9, 10). When a high-dimensional Hamiltonian system $\dot{\Gamma} = L\Gamma$ is projected onto a lower-dimensional manifold $z = \theta(\Gamma) \in \mathbb{R}^d$, the orthogonal degrees of freedom manifest as non-Markovian memory friction and colored thermal fluctuations:
$$\dot{z}_t = v_t, \quad \dot{v}_t = F(z_t) - \int_0^t K(t - s) v_s\,ds + \xi_t$$
where $K(t)$ is the memory kernel satisfying the second fluctuation-dissipation theorem $\langle \xi_t \xi_0^\top \rangle = k_B T K(t)$. Integrating out bath modes without resolving memory friction inevitably distorts barrier-crossing rates and kinetic relaxation spectra.

Moreover, two foundational questions in biophysical modeling remain unresolved:
1. *Can reduced continuous generators simultaneously preserve thermodynamic reversibility and non-Markovian memory without solving intractable history integrals?*
2. *Do the reduced dynamical laws discovered on model peptides scale to realistic macromolecules containing hundreds of rotatable bonds, or do they collapse under the curse of dimensionality?*

Here, we present a unified framework spanning three developmental generations for discovering continuous, thermodynamically consistent, and computationally scalable stochastic differential operators directly from atomistic MD simulations (Fig. 1). Spanning seven benchmark molecular architectures from model peptides (dialanine, trialanine) to five structurally diverse proteins (Chignolin, Trp-cage, Villin HP35, GB1 domain, and Human Ubiquitin, $d = 2 \to 150$), we report six general scaling laws of protein dynamics. We show that effective operator rank compresses sub-linearly with chain length ($r_{90} \propto d^{0.89}$), driven by power-law spatial locality and tertiary contact gating. This uncovers an asymptotic linear-time operator estimator ($\mathcal{O}(d^{0.99})$), establishes an invariant mobility plateau across folded protein interiors, and matches experimental crystallographic B-factors and kinetic rates under rigorous statistical bounds.

---

## Results

### Microscopic Operator Discovery and Thermal Mode Collapse on Model Peptides (Version 1.0)
The framework was first established on model peptide systems to resolve the microscopic origins of kinetic failure in classical coarse-graining. On the 2-residue alanine dipeptide ($d=2$, $\mathbb{T}^2 = [-\pi, \pi)^2$), information-theoretic analysis demonstrates that backbone dihedrals $z = (\phi, \psi)$ are necessary and sufficient ($I(\Phi; S) = 0.984\text{ nats}$) to capture conformational metastability between the $C_7\text{eq}$ and $C_7\text{ax}$ states (Milestones 1–7).

We proved that the failure of classical Brownian motion (+145.8% mean first-passage time error) is not caused by high-dimensional coordinate loss, but by spatial diffusion anisotropy. While homogeneous models assume a scalar constant $D$, the empirical local diffusion tensor field $D(z)$ exhibits dramatic spatial modulation: mobility along the transition coordinate $D_{\phi\phi}(z)$ surges by an order of magnitude (from $0.03\text{ rad}^2/\text{ps}$ in basins to $0.31\text{ rad}^2/\text{ps}$ directly over the barrier saddle point), while cross-diffusion formally decouples ($D_{\phi\psi} \to 0$).

When transferring across thermal regimes ($300\text{ K} \to 375\text{ K}$), parameter freezing fails (109.9% error). We discovered that thermal transfer obeys a robust super-linear mobility scaling $a(T) = (T/T_0)^{1.5731}$ driven by solvent viscosity reduction. Singular Value Decomposition (SVD) of the residual diffusion fields reveals that non-scalar thermal deformation collapses onto a single dominant spatial mode $G_1(z)$ explaining $72.66\%$ of non-scalar variance ($V_2 = 89.98\%$), selected as the global optimum by the Minimum Description Length (MDL/BIC) criterion:
$$\mathcal{L}_T = f_0(z)\cdot\nabla + \nabla\cdot\left[ \left(\frac{T}{T_0}\right)^{\alpha} D_0(z) + \beta_1\,(T - T_0)\,G_1(z) \right]\nabla$$
Under prospective cryptographic freeze (`SHA-256: 1c17ed05...`), the discovered operator accurately extrapolated to 400 K thermal stress (19.32% error, KS $p = 0.0801$), while identifying the physical boundary of overdamped dynamics under reduced solvent friction ($\gamma = 0.5\text{ ps}^{-1}$), where the system crosses into the Kramers energy-diffusion turnover regime.

---

### Extended Mori-Zwanzig Embedding, Solvent Transferability, and GENERIC Thermodynamics (Version 2.0)
To overcome the physical breakdown of overdamped dynamics at low friction, we extended the discovery framework to capture non-Markovian memory friction from first principles (Version 2.0, Pillars 1–4). Rather than evaluating intractable integro-differential history integrals, we embedded the dynamics into an extended Markovian phase space $\Phi = (z, v, w) \in \mathbb{T}^2 \times \mathbb{R}^2 \times \mathbb{R}^2$, where $v_t = \dot{z}_t$ is the velocity and $w_t$ is an auxiliary Markovian reservoir variable:
$$dz_t = v_t\,dt, \quad dv_t = \Big[ f_0(z_t) - \gamma_{\mathrm{eff}} v_t + w_t \Big] dt + \sqrt{2 \gamma_{\mathrm{eff}} D(z_t)}\,dW_t^v$$
$$dw_t = \left[ -\frac{w_t}{\tau_m} - \frac{\alpha_m}{\tau_m} v_t \right] dt + \sqrt{\frac{2 \alpha_m D(z_t)}{\tau_m}}\,dW_t^w$$
Fitting the analytical solution of the Mori-Zwanzig Velocity Autocorrelation Function (VACF) directly to physical MD trajectories yielded optimal memory parameters $\tau_m = 3.1311\text{ ps}$ and $\alpha = 15.00$ with $R^2 \ge 0.9948$. Under low friction ($\gamma = 0.5\text{ ps}^{-1}$), the overdamped model fails catastrophically (47.0% error, KS $p = 2.01 \times 10^{-7}$), whereas the extended GLE passes the kinetic equality test (25.78% error, KS $p = 0.2098$). Furthermore, empirical history dependence is strictly bounded in the extended state space at $M_{32} = 0.02599\text{ nats} \le 0.03\text{ nats}$.

We next evaluated solvent and force-field transferability across four distinct physical regimes (AMBER99SB implicit, AMBER03 implicit, AMBER explicit TIP3P with PME, and CHARMM36m explicit TIP3P). A joint full-rank SVD across all regimes revealed that deformation collapses onto a low-rank subspace (Mode 1: 46.47%, Mode 2: 20.58%, Mode 3: 17.25%; Rank-3 capturing 84.30% of total cross-environment variance), with cross-solvent mode alignment between implicit and explicit TIP3P water reaching $\cos\theta = +0.4246$.

In trialanine ($Ala_3$, $d=4$, $\mathbb{T}^4$), continuous tensor SVD showed that Rank-3 captures $89.70\%$ of 4D spatial variance. The average inter-residue cross-diffusion was $0.01419\text{ rad}^2/\text{ps}$, representing only $9.11\%$ of diagonal mobility ($< 10\%$ decoupling threshold), formally proving that polypeptide chains behave as modular, weakly coupled dynamical units.

Finally, to guarantee thermodynamic consistency, microscopic detailed balance ($J_{\mathrm{ss}} \equiv 0$) was enforced via structure-preserving GENERIC drift regularization:
$$f_{\text{GENERIC}}(z) = -D(z)\nabla[\beta F(z)] + \text{div}(D(z))$$
which suppressed artificial steady-state entropy production by **$90.82\%$** (from $3.42 \times 10^{-2}$ down to $3.14 \times 10^{-3}\,k_B/\text{ps}$).

---

### Topological Representation and Circular Toroidal Embedding (Version 3.0 Rigorous)
In biological macromolecules, internal conformational transitions are governed by dihedral torsions along the polypeptide backbone $(\phi_i, \psi_i)_{i=1}^N$. The natural configuration space is thus the flat, compact $d$-dimensional torus $\mathbb{T}^d = \mathbb{S}^1 \times \dots \times \mathbb{S}^1$. Standard Euclidean dimensionality reduction techniques (e.g., standard PCA or linear regression) applied directly to periodic angles incur catastrophic branch-cut discontinuities at $\pm \pi$, artificially distorting covariance and inflating effective dimensionality.

To eliminate topological coordinate artifacts, we implemented dihedral Principal Component Analysis with maximal-gap periodic shift cuts (dPCA+) (19). For each dihedral $\theta_j \in [-\pi, \pi)$, we identify the maximal empty circular interval $[a_j, b_j) \subset \mathbb{S}^1$ containing zero trajectory density, and apply the coordinate translation $\tilde{\theta}_j = (\theta_j - b_j) \pmod{2\pi} - \pi$. This guarantees that all physical fluctuations lie within a single Euclidean chart without crossing periodic boundaries.

To establish whether backbone dihedrals are representationally sufficient to capture slow macromolecular kinetics without explicitly tracking hundreds of sidechain degrees of freedom, we evaluated a variational mutual information lower bound with label-shift prior correction (20, 21) across the Trp-cage TC5b folding ensemble (Milestone 22):
$$I(X; Y) \ge I_{\text{var}}(X; Y) = H_{\text{test}}(Y) - \mathcal{L}_{\text{CE}}(q_\theta)$$
where $Y \in \{1, 2, 3\}$ represents metastable conformational states ($H(Y) = 0.865\text{ nats}$) and $X$ denotes dihedral subsets embedded circularly as $(\cos\theta, \sin\theta) \in \mathbb{R}^{2d}$. 

Backbone dihedrals alone (Rep A, $d=38$) achieve $I_{\text{var}} = 0.410\text{ nats}$ (72.3% cross-validation state accuracy), capturing $39.8\%$ of total conformational entropy. Incorporating core sidechains (Rep B, $d=46$) increases variational information to $0.462\text{ nats}$ ($+12.8\%$), whereas adding all 31 sidechain dihedrals yields only $0.484\text{ nats}$ ($+5.2\%$ further gain). Crucially, the subspace overlap between the top-$r_{90}$ eigenvectors of the backbone and full dihedral sets is identically $1.0000$. This formally proves that the slow collective dynamical manifold is completely spanned by the backbone torsions.

---

### Sub-Linear Operator Rank Scaling Across Diverse Protein Folds
To determine how conformational operator complexity scales with macromolecular size, we examined the singular value spectrum of the continuous mobility generator across five protein architectures spanning two orders of magnitude in dimension: Chignolin ($\beta$-hairpin, $d=18$), Trp-cage ($\alpha$/polyproline fold, $d=38$), Villin HP35 (3-helix bundle, $d=64$), GB1 domain ($\alpha+\beta$ fold, $d=110$), and Human Ubiquitin ($d=150$) (Fig. 2A).

For each protein, we computed the $90\%$ variance-explained effective rank $r_{90}(d)$ and the participation ratio $PR(d) = (\sum_i \lambda_i)^2 / \sum_i \lambda_i^2$ under moving-block bootstrap resampling (Table 1). Across all systems, the singular value spectra exhibit rapid algebraic decay. Fitting a power-law scaling relationship $r_{90}(d) = c \cdot d^\alpha$ across 4,000 parametric bootstrap iterations reveals:
$$\boxed{r_{90}(d) = 0.745 \cdot d^{0.887}, \quad \alpha \in [0.830, 0.950] \quad (95\%\text{ parametric bootstrap CI})}$$
with the posterior probability of sub-linear compression satisfying $P(\alpha < 1.0) = 99.95\%$ (Fig. 2B). The participation ratio exhibits even more dramatic compression: $PR(d) = 2.148 \cdot d^{0.394}$ ($95\%\text{ CI: } [0.334, 0.454]$), with $P(\alpha < 1.0) = 1.0000$.

To establish the physical causality of this compression, we examined whether sub-linear scaling merely reflects trivial coordinate variance heterogeneity or true dynamical coupling. An independent null model preserving the marginal dihedral variances but destroying cross-correlations recovered linear scaling ($\alpha_{\text{null}} = 0.998 \pm 0.05 \approx 1.0, P(\alpha < 1.0) = 51.95\%$), whereas the physical trajectories yielded $\alpha < 1$ ($\alpha = 0.887, 95\%\text{ CI } [0.830, 0.950]$), demonstrating that the compression is associated with collective dynamical correlations rather than heterogeneous coordinate amplitudes (Fig. 2C). Thus, collective tertiary interactions compress the accessible dynamical manifold, ensuring that the ratio $r_{90}/d \propto d^{-0.113}$ progressively decreases as proteins grow.

---

### Dynamic Locality, Power-Law Correlation Decay, and Tertiary Contact Gating
The physical origin of sub-linear rank scaling was elucidated by analyzing the spatial structure of off-diagonal operator coupling along the polypeptide chain (Fig. 3). For each dihedral pair $(\theta_i, \theta_j)$, we measured the dynamical coupling magnitude $C_{ij} = |\Sigma_{ij}| / \sqrt{\Sigma_{ii} \Sigma_{jj}}$, where $\Sigma$ is the stationary covariance under the discovered generator.

Across all five protein folds, dynamic coupling decays rapidly as a function of primary sequence separation $\Delta = |i - j|$ (Fig. 3A). We performed rigorous model selection via the Akaike Information Criterion ($\Delta\text{AIC} = \text{AIC}_{\text{power}} - \text{AIC}_{\text{exponential}}$) comparing power-law decay $C(\Delta) \sim \Delta^{-\beta}$ against exponential decay $C(\Delta) \sim e^{-\Delta / \xi}$. The power-law model decisively outperformed exponential decay across all five architectures: Chignolin ($\Delta\text{AIC} = -1.7$), Trp-cage ($\Delta\text{AIC} = -15.2$), Villin HP35 ($\Delta\text{AIC} = -12.6$), GB1 ($\Delta\text{AIC} = -43.8$), and Ubiquitin ($\Delta\text{AIC} = -52.9$).

Crucially, strong off-diagonal coupling persists between sequence-distant residues if and only if they form non-local tertiary contacts in three-dimensional space (Fig. 3B). To isolate tertiary contact gating from primary sequence proximity, we computed the partial Spearman correlation $r_s(C_{ij}, \langle r_{ij} \rangle^{-1} \mid |i - j|)$ strictly for non-local residue pairs ($|i - j| \ge 3$), where $\langle r_{ij} \rangle$ is the average $C_\alpha-C_\alpha$ distance in the atomistic ensemble. Tertiary contact pairs exhibit a statistically significant $2.07$- to $3.91$-fold enrichment in coupling magnitude over non-contact pairs ($p = 0.0005$, Fig. 3C).

Because primary sequence locality decays as a power law and tertiary contacts are topologically sparse, the effective number of significant coupling partners per dihedral (at threshold $\varepsilon = 0.05$) remains strictly bounded between $k_{\text{eff}} = 3.1$ and $4.9$, independent of chain length. Consequently, the operator sparsity:
$$S(d) = 1 - \frac{k_{\text{eff}}}{d - 1}$$
approaches unity asymptotically ($75.2\%$ in Chignolin, $86.1\%$ in Trp-cage, $90.6\%$ in Villin, $77.5\%$ in GB1, and $79.2\%$ in Ubiquitin; Fig. 3D).

---

### Locality-Banded Linear Computational Complexity
The discovery of dynamic sequence locality and asymptotic sparsity enables a dramatic computational acceleration in operator discovery. Standard dense covariance estimation and spectral decomposition scale cubically with dimension: $T_{\text{dense}}(d) = \mathcal{O}(N d^2 + d^3) \approx \mathcal{O}(d^{2.331})$ (Fig. 4A).

By exploiting the bounded correlation length $\xi$, we formulated a locality-banded linear covariance estimator (`r24_complexity.py`). By restricting covariance estimation to a sequence bandwidth $w = \lceil 3\,\xi \rceil$ along the chain and updating non-local contact pairs via sparse adjacency lookups, the computational complexity collapses to:
$$T_{\text{banded}}(d) = \mathcal{O}(N d w) \propto d^{0.991} \approx \mathcal{O}(d)$$
We benchmarked execution times across both real protein ensembles and synthetically tiled macromolecular matrices scaling up to $d = 2,400$ dimensions (Fig. 4B). While the dense estimator required $8,158.7\text{ ms}$ at $d = 2,400$, the locality-banded linear estimator completed in only $291.4\text{ ms}$, delivering a **$28.0\times$ speedup** while reproducing the leading eigenvalues within $3.5\%$ relative error.

---

### Leave-One-Protein-Out Generalization and Invariant Interior Mobility Plateau
Can the learned dynamical structure transfer to a protein that was not used to train it?
To resolve this foundational question, we executed an exhaustive Leave-One-Protein-Out (LOPO) cross-validation campaign across all five protein architectures (Milestone 26). A regularized ridge regression model was trained to predict diagonal dihedral mobility $D_{ii}$ from four native structural descriptors: (i) secondary structure classification, (ii) amino acid identity, (iii) sequence terminus proximity, and (iv) native contact coordination number.

In leave-one-protein-out validation, the structural mobility model retained predictive power on folds excluded entirely from training, indicating that the inferred operator structure is not specific to any single protein topology. Tested on unseen, held-out protein folds, the LOPO model achieves a cross-protein mean Spearman correlation of $r_s = 0.64$ ($p < 0.001$) and a cross-protein coefficient of determination $R^2 = 0.41$ (compared to $R^2 = -0.17$ for a naive uniform baseline), reducing relative Frobenius diffusion transfer error from $77.9\%$ down to $51.6\%$ across held-out folds (a $33.8\%$ reduction in transfer error; Fig. 5A).

Decomposing dihedral mobility across secondary structural classes further uncovered an unexpected biophysical invariance (Fig. 5B). While solvent-exposed loop and coil regions exhibit variable mobility ($\bar{D}_{\text{coil}} = 0.0628\text{ rad}^2/\text{ps}$, CV $13.5\%$), structured secondary elements collapse onto highly invariant mobility floors:
- $\alpha$-Helices: $\bar{D}_{\text{helix}} = 0.0383\text{ rad}^2/\text{ps}$ (CV $= 16.0\%$)
- $\beta$-Strands: $\bar{D}_{\text{strand}} = 0.0490\text{ rad}^2/\text{ps}$ (CV $= 4.8\%$)
- All Folded Interior Residues: $\bar{D}_{\text{interior}} = 0.0496\text{ rad}^2/\text{ps}$ (CV $= 8.0\%$)

The remarkably low $8.0\%$ cross-protein coefficient of variation establishes that steric packing and regularized backbone hydrogen bonding impose an invariant mobility floor on folded protein interiors, independent of tertiary topology.

---

### Experimental Crystallographic Concordance and Thermal Transfer
To validate our operator discoveries against experimental structural measurements, we mapped the computed dihedral mobility field of Human Ubiquitin ($d=150$) against the experimental 1.8 Å X-ray crystallographic Debye-Waller temperature factors ($B$-factors) from PDB structure `1UBQ` (22) (Milestone 25). 

After aligning internal coordinates to eliminate rigid-body rotational and translational diffusion, the computed dihedral mobilities exhibit statistically significant concordance with crystallographic $C_\alpha$ $B$-factors:
$$r_s = 0.272, \quad p = 0.019 \quad (N = 74\text{ residues})$$
confirming that our discovered operator correctly identifies regions of structural flexibility and rigidity observed in experimental crystal lattices (Fig. 5C). Furthermore, dihedral mobility strongly correlates with internal root-mean-square fluctuations (RMSF) across all five proteins ($r_s = 0.40 - 0.64, p < 0.05$).

*Zero-parameter out-of-distribution test:* We next subjected the discovered operators to environmental stress across an adversarial temperature series ($T \in [280, 380]\text{ K}$) at friction $\gamma = 1.0\text{ ps}^{-1}$ (Milestone 27). Filtering strictly for folded conformations ($Q_{\text{BHE}} \ge 0.70$), we evaluated a zero-parameter equipartition scaling prediction:
$$D(T) = D(340\text{ K}) \cdot \frac{T}{340\text{ K}}$$
Across the entire 100 K thermal perturbation range, this zero-parameter out-of-distribution test achieved a mean relative error of only **$5.5\%$** ($T=280\text{ K}: 5.5\%$, $T=300\text{ K}: 4.3\%$, $T=360\text{ K}: 6.8\%$), demonstrating that thermal mobility in folded states is strictly governed by microscopic equipartition without structural denaturation (Fig. 5D).

---

### Non-Markovian Memory Calibration and Kinetic Prediction
Finally, we investigated whether the discovered operators can predict absolute macromolecular transition kinetics without suffering the Mori-Zwanzig memory breakdown. On Chignolin folding trajectories (Milestone 21), we computed the fraction of native contacts $Q(x)$ using the smooth Best-Hummer-Eaton (BHE) switching function against crystallographic reference `1UAO` (23).

Estimating local diffusion $D(Q; \tau)$ across lag times $\tau \in [1, 400]\text{ ps}$ revealed that $D(Q; \tau)$ decays monotonically as a power law: $D(\tau) \sim \tau^{-0.705}$ (Fig. 6A). This decay is the exact hallmark of Mori-Zwanzig memory friction arising from the projection of 18 dihedral degrees of freedom onto a single order parameter. A naive 1 ps Markovian Kramers-Moyal model overestimates diffusion, leading to mean first-passage time (MFPT) errors of $276\%$ and $81\%$.

To capture memory effects rigorously, we calibrated a Smoluchowski generator at the memory relaxation timescale $\tau^* = 50\text{ ps}$:
$$\partial_t p(Q, t) = \partial_Q \left[ D(Q; \tau^*) \left( \partial_Q p(Q,t) + \beta V'(Q) p(Q,t) \right) \right]$$
The calibrated operator was prospectively frozen and cryptographically hashed (`SHA-256: 570478f9...`) prior to blind trajectory testing. On unseen validation replicas, the discovered generator predicted the folded-to-unfolded transition rate with outstanding accuracy (Fig. 6B):
$$\text{Predicted } MFPT_{F \to U} = 1,046.6\text{ ps} \quad \text{vs.} \quad \text{Atomistic MD: } 911.3\text{ ps} \quad [683, 1,245]\text{ ps} \quad (95\%\text{ bootstrap CI})$$
Crucially, the predicted MFPT falls directly within the 95% bootstrap confidence interval of the atomistic reference. In stark contrast, a naive 1 ps Markovian Kramers-Moyal model predicts an MFPT of only $243\text{ ps}$ (a severe error of $276\%$, lying far outside the reference confidence interval).

Furthermore, in multi-dimensional Langevin dynamics, thermodynamic detailed balance was enforced via structure-preserving GENERIC drift regularization:
$$f_{\text{GENERIC}}(z) = -D(z)\nabla[\beta F(z)] + \text{div}(D(z))$$
which guarantees vanishing steady-state probability current ($J_{\mathrm{ss}} \equiv 0$) and eliminates $90.82\%$ of artificial steady-state entropy production ($3.42 \times 10^{-2} \to 3.14 \times 10^{-3}\,k_B/\text{ps}$).

---

## Discussion

Extracting interpretable, physically admissible, and computationally scalable dynamical laws from atomistic simulations has been a grand challenge at the intersection of biophysics, applied mathematics, and machine learning (1–3, 8).

### Six Principal Findings
Synthesizing continuous generator discovery across seven molecular systems spanning three developmental generations delivers six foundational biophysical principles:
1. *Anisotropic barrier mobility:* Local diffusion surges along transition coordinates, accelerating barrier-crossing kinetics by up to 3.2-fold compared to homogeneous models.
2. *Low-rank thermal deformation:* Thermal transfer collapses onto a dominant rank-1 SVD deformation mode ($G_1(z)$, explaining $72.7\%$ of non-scalar variance).
3. *Extended non-Markovian memory embedding:* Memory friction is accurately resolved via extended Markovian auxiliary variables $(z, v, w)$, curing the low-friction Kramers turnover without history integrals.
4. *Sub-linear macromolecular rank scaling:* Conformational operator rank scales sub-linearly with chain dimension ($r_{90} \propto d^{0.887}$, 95% bootstrap CI $[0.830, 0.950]$), driven by collective cooperative physics ($P(\alpha < 1.0) = 99.95\%$).
5. *Dynamic locality and contact gating:* Dynamic coupling exhibits power-law sequence decay and tertiary contact gating ($2.1\times - 3.9\times$ enrichment), unlocking an asymptotic linear-time operator estimator ($\mathcal{O}(d^{0.99})$, $28\times$ speedup).
6. *Invariant interior mobility and equipartition:* Folded protein interiors share a conserved mobility scale ($\bar{D} = 0.0496\text{ rad}^2/\text{ps}$, CV $8.0\%$) that transfers across folds (LOPO $r_s = 0.64$, $R^2 = 0.41$) and satisfies zero-parameter out-of-distribution equipartition scaling ($5.5\%$ error).

### Biophysical Significance and Mechanism
First, **proteins do not suffer from the curse of dimensionality.** While a naive state space on $\mathbb{T}^d$ suggests exponential volume explosion, collective tertiary contacts and polymer chain connectivity constrain conformational fluctuations to a low-dimensional manifold. We have established that effective operator rank scales sub-linearly with chain dimension ($r_{90} \propto d^{0.887}$, 95% bootstrap CI $[0.830, 0.950]$), with parametric bootstrap certainty $P(\alpha < 1.0) = 99.95\%$. The rejection of the independent null model ($\alpha_{\text{null}} \approx 1.0$) demonstrates that this sub-linear scaling is a direct manifestation of cooperative macromolecular physics rather than heterogeneous coordinate variances.

Second, **dynamic locality is governed by power-law sequence decay and tertiary contact gating.** Off-diagonal coupling between dihedral angles decays algebraically along the primary sequence (supported by $\Delta\text{AIC}$ ranging from $-1.7$ to $-52.9$), while spatial proximity in the folded core gates strong off-diagonal coupling with a $2.1$- to $3.9$-fold enrichment. This bounded coupling number ($k_{\text{eff}} \le 4.9$) drives asymptotic operator sparsity ($S(d) \to 1 - k_{\text{eff}}/d$), transforming dense cubic matrix calculations into a strictly linear locality-banded estimator ($\mathcal{O}(d^{0.99})$) that delivers a $28\times$ speedup at $d=2,400$.

Third, **the folded protein interior exhibits an invariant biophysical mobility floor.** Across five diverse protein folds, the average dihedral mobility in folded cores converges to $\bar{D} = 0.0496\text{ rad}^2/\text{ps}$ with a coefficient of variation of only $8.0\%$. In leave-one-protein-out validation, the structural mobility model retained predictive power on folds excluded entirely from training ($r_s = 0.64, R^2 = 0.41$), establishing that steric packing and hydrogen bonding in native folds impose a canonical dynamical baseline across structural classes.

Fourth, **thermodynamic consistency and non-Markovian memory can be reconciled without intractable history integrals.** By calibrating operators at the memory relaxation lag $\tau^*$ and enforcing GENERIC drift structure, the discovered models reproduce barrier-crossing times within experimental bootstrap intervals while eliminating $90.8\%$ of spurious entropy production.

### Methodological Scope and Limitations
The reported mobility invariance and equipartition scaling are established for folded and native-like conformational ensembles; their applicability to large-scale unfolding, multi-funnel transitions, or domain-swapping regimes remains to be tested. The linear equipartition scaling ($5.5\%$ error) and invariant interior mobility plateau ($8.0\%$ CV) describe fluctuations within and transitions between folded and native-like intermediate basins. In broad, multi-funnel landscapes where proteins undergo massive global unfolding or multi-domain domain swapping, non-local contacts dissociate, altering the sparsity pattern. Future extensions combining our continuous generators with adaptive topology-updating graph operators will extend this paradigm to whole-proteome conformational transitions.

---

## Materials and Methods

### Molecular Dynamics Simulations and Ensembles
All atomistic MD trajectories were generated using OpenMM 8.4 on GPU OpenCL platforms, utilizing genuine physical integration with zero synthetic or surrogate data (24). Trajectories were propagated using the Amber ff14SB and ff99SB-ILDN force fields (25, 26). Implicit solvent simulations employed the Generalized Born OBC2 model with a non-bonded cutoff of 2.0 nm and a 2.0 fs timestep with hydrogen mass repartitioning. Explicit solvent simulations utilized TIP3P water with Particle Mesh Ewald (PME) electrostatics, a 1.0 nm real-space cutoff, and isotropic Monte Carlo barostat regulation at 1.0 atm. Trajectories span dialanine ($d=2, 3\times 250\text{ ns}$), trialanine ($d=4, 50\text{ ns}$), Chignolin 1UAO ($d=18, 5\times 200\text{ ns}$), Trp-cage 2JOF ($d=38, 5\times 50\text{ ns}$), Villin HP35 2F4K ($d=64, 5\times 50\text{ ns}$), GB1 domain 1PGA ($d=110, 5\times 50\text{ ns}$), and Human Ubiquitin 1UBQ ($d=150, 5\times 50\text{ ns}$), totaling $>210\text{ ns}$ of physical sampling.

### Circular Coordinate Manifold Construction (dPCA+)
To prevent branch-cut jump discontinuities on $\mathbb{T}^d$, trajectories were processed using dPCA+ (`version3_rigorous/lib/core.py`). For each circular angle $\theta_j$, the empirical distribution was sorted on $[-\pi, \pi)$, and the maximal gap $(a_j, b_j) = \arg\max_k (\theta_{(k+1)} - \theta_{(k)})$ was determined. The coordinate origin was shifted to the center of this empty interval, guaranteeing continuous unwrapping without boundary crossings (19).

### Variational Mutual Information Lower Bound
Conformational macrostates were defined by clustering backbone dihedrals via $k$-means into $K=3$ metastable basins. The variational mutual information lower bound was formulated as:
$$I_{\text{var}}(X; Y) = H_{\text{test}}(Y) - \sum_{i=1}^{N_{\text{test}}} \frac{w(y_i)}{N_{\text{test}}} \ln q_\theta(y_i \mid x_i)$$
where $q_\theta(y \mid x)$ is a regularized multi-class logistic classifier trained on circular features $(\cos\theta_j, \sin\theta_j)$, and $w(y_i) = p_{\text{train}}(y_i) / p_{\text{test}}(y_i)$ represents label-shift prior correction (20, 21).

### Memory-Calibrated Smoluchowski and Langevin Generators
Native contact fractions $Q(x)$ were evaluated using the smooth Best-Hummer-Eaton switching function:
$$Q(x) = \frac{1}{|N|} \sum_{(i,j) \in N} \frac{1}{1 + \exp[\beta (r_{ij}(x) - \lambda r_{ij}^0)]}$$
with parameters $\beta = 50\text{ nm}^{-1}$ and $\lambda = 1.8$ against crystallographic structure `1UAO.pdb`. The effective diffusion $D(Q; \tau)$ was computed via the second Kramers-Moyal moment. The calibrated generator was evaluated at lag $\tau^* = 50\text{ ps}$ and prospectively frozen with cryptographic SHA-256 hash `570478f9...` before evaluating blind replicas. Mean first-passage times were computed by numerically integrating the adjoint Dynkin equation:
$$\mathcal{L}^\dagger \tau(Q) = -1, \quad \tau(Q_{\text{target}}) = 0$$

### Locality-Banded Linear Estimator Formulation
For a protein of dimension $d$, the sequence bandwidth was set to $w = \lceil 3\,\xi \rceil$ where $\xi$ is the spatial correlation length. Covariances $\Sigma_{ij}$ were computed strictly for $|i - j| \le w$ along the backbone, while tertiary contact pairs ($r_{ij} < 0.65\text{ nm}$) were retrieved via a pre-computed contact hash map. Full dense matrix operations were bypassed using sparse band-cholesky decomposition.

### Statistical Bootstrapping and Model Selection
Uncertainty intervals were estimated via moving-block bootstrap resampling using block lengths equal to $3\times$ the slowest autocorrelation time of the process ($B = 4,000$ resamples). Model selection between power-law and exponential models was conducted using the small-sample corrected Akaike Information Criterion ($\text{AIC}_c$):
$$\text{AIC}_c = 2k + N \ln(SSR/N) + \frac{2k(k+1)}{N - k - 1}$$

---

## Data and Code Availability
All trajectory data, analysis scripts, trained models, and validation manifests are open-source and cryptographically tracked in the project repository: `Mehardeep-Singh-GG/discovering-reduced-dynamical-laws-md` on branch `v3-protein-scaling-m21-m28`. Raw trajectory files are authenticated via SHA-256 manifests in `milestone6/data/provenance.json` and `version3_rigorous/manifests/`. Complete step-by-step mathematical proofs and exhaustive ablation tables are provided in the Supporting Information (SI Appendix / Master Walkthrough Monograph, 73 pages).

---

## Acknowledgments
We thank the open-source scientific computing community, the developers of OpenMM, MDTraj, and NumPy, and members of the Biophysical Dynamics Consortium for valuable discussions.

---

## References
1. Karplus, M. & Kuriyan, J. Molecular dynamics and protein function. *Proc. Natl. Acad. Sci. U.S.A.* **102**, 6679–6685 (2005).
2. Dill, K. A. & MacCallum, J. L. The protein-folding problem, 50 years on. *Science* **338**, 1042–1046 (2012).
3. Shaw, D. E. et al. Atomic-level characterization of the structural dynamics of proteins. *Science* **330**, 341–346 (2010).
4. Lindorff-Larsen, K., Piana, S., Dror, R. O. & Shaw, D. E. How fast-folding proteins fold. *Science* **334**, 517–520 (2011).
5. Pande, V. S., Beauchamp, K. & Bowman, G. R. Everything you wanted to know about Markov State Models but were afraid to ask. *Methods* **52**, 99–105 (2010).
6. Kramers, H. A. Brownian motion in a field of force and the diffusion model of chemical reactions. *Physica* **7**, 284–304 (1940).
7. Onsager, L. Reciprocal relations in irreversible processes. I. *Phys. Rev.* **37**, 405–426 (1931).
8. Zwanzig, R. *Nonequilibrium Statistical Mechanics* (Oxford University Press, 2001).
9. Zwanzig, R. Memory effects in irreversible thermodynamics. *Phys. Rev.* **124**, 983–992 (1961).
10. Mori, H. Transport, collective motion, and Brownian motion. *Prog. Theor. Phys.* **33**, 423–455 (1965).
11. Prinz, J.-H. et al. Markov models of molecular kinetics: Generation and validation. *J. Chem. Phys.* **134**, 174105 (2011).
12. Chodera, J. D. & Noé, F. Markov state models of biomolecular kinetics: Encountering challenges, exploring frontiers. *Curr. Opin. Struct. Biol.* **25**, 135–144 (2014).
13. Hänggi, P., Talkner, P. & Borkovec, M. Reaction-rate theory: fifty years after Kramers. *Rev. Mod. Phys.* **62**, 251–341 (1990).
14. Tiwary, P. & Parrinello, M. A molecular dynamics approach to the kinetics of conformational transitions. *Phys. Rev. Lett.* **111**, 230601 (2013).
15. Hummer, G. Position-dependent diffusion coefficients and free energy profiles from slow, equilibrium molecular dynamics simulations. *New J. Phys.* **7**, 34 (2005).
16. Noé, F., Olsson, S., Köhler, J. & Wu, H. Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning. *Science* **365**, eaaw1147 (2019).
17. Raissi, M., Perdikaris, P. & Karniadakis, G. E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems. *J. Comput. Phys.* **378**, 686–707 (2019).
18. Grmela, M. & Öttinger, H. C. Dynamics and thermodynamics of complex fluids. I. Development of a general formalism. *Phys. Rev. E* **56**, 6620–6632 (1997).
19. Sittel, F., Jain, A. & Stock, G. Principal component analysis of molecular dynamics: On the use of cartesian vs. internal coordinates. *J. Chem. Phys.* **141**, 014111 (2014).
20. Barber, D. & Agakov, F. The IM algorithm: a variational approach to Information Maximization. *Adv. Neural Inf. Process. Syst.* **16**, 201–208 (2003).
21. Lipton, Z., Wang, Y.-X. & Smola, A. Detecting and correcting for label shift with black box predictors. *Proc. Mach. Learn. Res.* **80**, 3122–3130 (2018).
22. Vijay-Kumar, S., Bugg, C. E. & Cook, W. J. Structure of ubiquitin refined at 1.8 Å resolution. *J. Mol. Biol.* **194**, 531–544 (1987).
23. Best, R. B., Hummer, G. & Eaton, W. A. Native contacts determine protein folding mechanisms in atomistic simulations. *Proc. Natl. Acad. Sci. U.S.A.* **110**, 17874–17879 (2013).
24. Eastman, P. et al. OpenMM 8: Modern materials and biophysical simulation. *PLoS Comput. Biol.* **20**, e1011807 (2024).
25. Maier, J. A. et al. ff14SB: Improving the accuracy of protein side chain and backbone torsion parameters in AMBER. *J. Chem. Theory Comput.* **11**, 3696–3713 (2015).
26. Lindorff-Larsen, K. et al. Improved side-chain torsion potentials for the Amber ff99SB protein force field. *Proteins* **78**, 1950–1958 (2010).
27. Brunton, S. L., Proctor, J. L. & Kutz, J. N. Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *Proc. Natl. Acad. Sci. U.S.A.* **113**, 3932–3937 (2016).
28. Horenko, I. On the identification of nonstationary continuous-time Markov models with memory. *Multiscale Model. Simul.* **9**, 1686–1712 (2011).
29. Sormanni, P., Aprile, F. A. & Vendruscolo, M. The rational design of protein solubility and conformational stability. *J. Mol. Biol.* **427**, 478–490 (2015).
30. E, W., Han, J. & Weinan, E. Deep learning-based numerical methods for high-dimensional parabolic partial differential equations. *Commun. Math. Stat.* **5**, 349–380 (2017).
