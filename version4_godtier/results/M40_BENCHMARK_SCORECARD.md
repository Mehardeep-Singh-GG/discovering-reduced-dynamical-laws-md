# Milestone 40 Benchmark Scorecard: Fair 6-Method Head-to-Head SOTA Evaluation

**Campaign:** Version 4.0 God-Tier (`version4_godtier`, 13 Macromolecular Proteins, $d = 18 \to 738$)  
**Evaluation Protocol:** Systematically tuned on `Rep 1 -> Rep 2`, locked, and evaluated once blindly on `Rep 3`.  

## Table 1: Validation-Tuned Hyperparameters Selected on `Rep 2`
| Protein | $d$ | M1 (VAMP-2 MSM) | M2 (TICA) | M3 (Smol) | M4 (Unconst SDE) | M5 (V4 GENERIC) | M6 (Tuned ENM) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Chignolin | 18 | $k=100, \tau=5$ | $r=5, \tau=2$ | $M=20, \tau^*=50$ | $B=2, \lambda=0.0001$ | $\ell=6.0, \beta_c=0.5$ | $r_c=1.2$ nm |
| Trp-cage | 38 | $k=20, \tau=5$ | $r=5, \tau=10$ | $M=40, \tau^*=20$ | $B=1, \lambda=0.0001$ | $\ell=6.0, \beta_c=0.3$ | $r_c=1.2$ nm |
| BBA5 | 54 | $k=200, \tau=5$ | $r=8, \tau=20$ | $M=30, \tau^*=5$ | $B=2, \lambda=0.1$ | $\ell=6.0, \beta_c=0.5$ | $r_c=1.0$ nm |
| Villin | 64 | $k=20, \tau=5$ | $r=8, \tau=2$ | $M=40, \tau^*=5$ | $B=2, \lambda=0.1$ | $\ell=6.0, \beta_c=0.15$ | $r_c=0.7$ nm |
| WW-FiP35 | 68 | $k=20, \tau=5$ | $r=8, \tau=5$ | $M=40, \tau^*=5$ | $B=2, \lambda=0.0001$ | $\ell=6.0, \beta_c=0.5$ | $r_c=0.8$ nm |
| GB1 | 110 | $k=100, \tau=10$ | $r=5, \tau=2$ | $M=20, \tau^*=5$ | $B=2, \lambda=0.01$ | $\ell=6.0, \beta_c=0.5$ | $r_c=1.2$ nm |
| BPTI | 114 | $k=100, \tau=5$ | $r=8, \tau=2$ | $M=30, \tau^*=50$ | $B=2, \lambda=0.01$ | $\ell=6.0, \beta_c=0.15$ | $r_c=0.7$ nm |
| Ubiquitin | 150 | $k=200, \tau=5$ | $r=12, \tau=20$ | $M=40, \tau^*=10$ | $B=2, \lambda=0.01$ | $\ell=6.0, \beta_c=0.5$ | $r_c=1.0$ nm |
| Barnase | 218 | $k=20, \tau=5$ | $r=5, \tau=2$ | $M=40, \tau^*=50$ | $B=2, \lambda=0.01$ | $\ell=6.0, \beta_c=0.15$ | $r_c=1.0$ nm |
| Lysozyme | 256 | $k=50, \tau=5$ | $r=5, \tau=2$ | $M=20, \tau^*=50$ | $B=2, \lambda=0.01$ | $\ell=6.0, \beta_c=0.3$ | $r_c=1.2$ nm |
| DHFR | 316 | $k=50, \tau=5$ | $r=5, \tau=2$ | $M=20, \tau^*=50$ | $B=2, \lambda=0.01$ | $\ell=6.0, \beta_c=0.5$ | $r_c=0.8$ nm |
| AdK | 426 | $k=20, \tau=5$ | $r=12, \tau=2$ | $M=40, \tau^*=50$ | $B=1, \lambda=0.01$ | $\ell=6.0, \beta_c=0.3$ | $r_c=1.2$ nm |
| MBP | 738 | $k=20, \tau=5$ | $r=5, \tau=2$ | $M=20, \tau^*=50$ | $B=2, \lambda=0.1$ | $\ell=6.0, \beta_c=0.0$ | $r_c=0.7$ nm |

## Table 2: 5-Axis Blind-Test (`Rep 3`) Head-to-Head Performance Summary
| Evaluation Metric | M1 (MSM) | M2 (TICA) | M3 (Smol) | M4 (Unconst SDE) | M6 (Tuned ENM) | M5 (V4 GENERIC) | Empirical Winner |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A: Mean Blind MFPT Log-Err** | 3.683 | 3.336 | **1.764** | 2.186 | N/A | **1.764** | **Statistical Tie (M3_SMOLUCHOWSKI / M5_V4_GENERIC)** |
| **B: Spurious Entropy $\dot{S}_{\rm irr}$ ($k_B$/ns/$d$)** | **0.000** | **0.000** | **0.000** | 346.17 | N/A | **0.000 (Exact)** | **Tie (M1 / M2 / M3 / M5)** |
| **C: Mean RelFrob Err on $\mathbf{D}$ ($13$ prots)** | N/A | 1.759 | 0.754 | 0.497 | N/A | **0.288** | **M5_V4_GENERIC** |
| **D: 13-Fold Zero-Shot LOPO ($r_s$)** | N/A | N/A | N/A | N/A | +0.328 | **+0.706** | **M5_V4_GENERIC** |
| **E: Solution NMR $S^2$ Fidelity ($r_s$)** | +0.307 | +0.145 | -0.110 | +0.250 | +0.378 | **+0.255** | **M6_TUNED_ENM** |
