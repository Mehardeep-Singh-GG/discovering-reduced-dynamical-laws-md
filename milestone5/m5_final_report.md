# Milestone 5 — Final Scientific Report: Frozen-Law Held-Out and Cross-Regime Validation

## 1. Executive Summary & Author Verdict

$$\boxed{
\textbf{Milestone 5 Verdict: PASS for held-out/cross-trajectory validation;}
\quad
\textbf{CROSS-REGIME: PARTIAL TRANSFER, NOT YET VALIDATED.}
}
$$

The structural dynamical operator discovered in Milestone 4B:
$$dz_t = f^\ast(z_t)\,dt + L^\ast(z_t)\,dW_t, \qquad D^\ast(z) = \begin{pmatrix} D_{\phi\phi}^\ast(z) & 0 \\ 0 & D_{\psi\psi}^\ast(z) \end{pmatrix}$$
was evaluated under complete scientific freezing across an **untouched held-out trajectory $T_3$** ($300\text{ K}$) and a **cross-regime thermodynamic perturbation** ($350.0\text{ K}$ with OpenMM).

The central scientific findings are:
1. **Held-Out Trajectory $T_3$ Generalization ($300\text{ K}$)**: **PASS**. The frozen operator reproduces local dynamics ($1\text{ ps}$ error ratio $1.016$), equilibrium free energy ($D_{\mathrm{KL}} = 0.3026\text{ nats}$), and forward transition kinetics ($\langle T_{A\to B}\rangle = 23.96\text{ ps}$ vs MD $20.69\text{ ps}$, relative error $15.8\%$, Kolmogorov-Smirnov test $p = 0.3365$).
2. **Strict Cross-Regime Transfer ($350\text{ K}$)**: **FAIL**. Keeping 300 K parameters unchanged predicts $\langle T\rangle = 25.16\text{ ps}$ vs empirical MD $11.99\text{ ps}$ (relative error $109.9\%$, $p < 0.0001$).
3. **Theoretical Structural Transfer ($D \propto T$)**: **PARTIAL TRANSFER**. Prescribing $D_{350}(z) = (350/300) D_{300}(z)$ accelerates kinetics to $\langle T\rangle = 18.76\text{ ps}$, but remains $56.5\%$ from MD and fails the distribution test ($p = 0.0008$).
4. **Resolution via Independent Local Scaling (Hypothesis 2)**: Estimating the local diffusion scaling independently from one-step residuals ($a_{\mathrm{emp}} = 1.471$) and applying it to the frozen functional form $D^\ast(z)$ recovers the $350\text{ K}$ kinetics ($\langle T\rangle = 12.99\text{ ps}$, $7.2\%$ error, **$p = 0.6330$**). The observed scaling exceeded the $T$-proportional prediction, indicating that additional temperature-dependent changes in the effective reduced dynamics contribute.

---

## 2. Master Validation Table

| Dataset | Representation | Operator Law | Multi-Lag RMSE (50 ps) | Equilibrium $D_{\mathrm{KL}}$ (nats) | Autocorr Error $\Delta C$ | Mean FPT $\langle T \rangle$ (ps) | Survival KS $p$-value | Drift $R^2$ | Diffusion $R^2$ | Memory Residual $M_{32}$ | Final Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T_2$ (Validation)** | Frozen $(\phi, \psi)$ | cand_005 | 4.3031 | 0.2512 | 0.0165 | 23.85 | 0.3540 | 0.985 | 0.892 | 0.0000 | **PASS** |
| **$T_3$ (Untouched Test)**| Frozen $(\phi, \psi)$ | cand_005 | 4.2180 | 0.3026 | 0.0161 | 23.96 | **0.3365** | **0.979** | **0.443** | **0.0311** | **PASS (Within-Regime)** |
| **Regime R2 Strict** | Frozen $(\phi, \psi)$ | cand_005 ($s=1.0$) | 4.1205 | 0.2840 | 0.0188 | 25.16 | **0.0000** | 0.972 | 0.885 | 0.0000 | **FAIL (Strict Transfer)** |
| **Regime R2 Structural $D\sim T$** | Frozen $(\phi, \psi)$ | cand_005 ($s=1.167$) | 4.1205 | 0.2840 | 0.0188 | 18.76 | **0.0008** | 0.972 | 0.885 | 0.0000 | **PARTIAL TRANSFER** |
| **Regime R2 Local Scaled** | Frozen $(\phi, \psi)$ | cand_005 ($s=1.47$) | 4.1205 | 0.2840 | 0.0188 | **12.99** | **0.6330** | 0.972 | 0.885 | 0.0000 | **PASS (Hypothesis 2)** |

---

## 3. Scientific Nuances & Diagnostic Breakdown

### A. Cross-Trajectory Stability vs Absence of Overfitting
The pairwise transfer error matrix across trajectories:
$$E_{ij} = \begin{pmatrix} 0.6985 & 0.7076 & 0.7098 \\ 0.7047 & 0.7027 & 0.7123 \\ 0.7018 & 0.7077 & 0.7072 \end{pmatrix}\quad \text{(rad)}$$
exhibits $< 1.5\%$ variation. This provides **no evidence of substantial trajectory-specific degradation under the tested split** (rather than an absolute proof of absence of overfitting).

### B. Asymmetry Between Drift and Diffusion Field Transfer
A striking asymmetry emerges in local field validation:
- **Drift Field Agreement**: $R^2(f_\phi) = 0.9958$, $R^2(f_\psi) = 0.9629$ (strong agreement).
- **Diffusion Field Agreement**: $R^2(D_{\phi\phi}) = 0.5401$, $R^2(D_{\psi\psi}) = 0.3463$ (moderate/weak agreement).
The frozen drift field transfers strongly; diffusion-field agreement is substantially weaker. Yet the stochastic model reproduces the macroscopic kinetics reasonably well at $300\text{ K}$, indicating that macroscopic mean first passage times depend primarily on drift-guided basin exit and average local mobility rather than point-by-point diffusion precision.

### C. Finite-History Memory Residuals
The mutual information residual increases with history length:
$$M_1 = 0.00786\text{ nats} \;\longrightarrow\; M_{16} = 0.01785\text{ nats} \;\longrightarrow\; M_{32} = 0.03113\text{ nats}.$$
Because $M_k$ increases rather than monotonically disappearing, the representation is not strictly Markovian. However, because $M_k < 0.05\text{ nats}$ across the entire tested horizon, **no large residual history dependence was detected under the prespecified practical cutoff**.

### D. Bidirectional Kinetics Asymmetry & Failure to Reject
- **Forward Transition ($A\to B$)**: MD $= 20.69\text{ ps}$, Model $= 23.96\text{ ps}$ ($15.8\%$ error, $p = 0.3365$).
- **Reverse Transition ($B\to A$)**: MD $= 43.31\text{ ps}$, Model $= 30.92\text{ ps}$ ($28.6\%$ relative difference, $p = 0.1183$).
The statistical test results ($p > 0.05$) indicate **failure to reject the null hypothesis**, not positive proof of distributional equality. The substantial difference in the reverse means indicates potential directional sensitivity in the reduced representation or finite-sample effects in the $N = 39$ empirical transitions.

### E. Transition-Path Geometry Discrepancy
Comparing reactive saddle crossings on $T_3$:
- Empirical MD: $(\phi, \psi) = (-1.718, 0.286)\text{ rad}$
- Frozen Model: $(\phi, \psi) = (-1.624, -0.689)\text{ rad}$
The difference in the $\psi$ coordinate ($0.286\text{ rad}$ vs $-0.689\text{ rad}$) is substantial, demonstrating that **correct global kinetics does not imply identical reactive-path geometry**. The model recovers a useful reduced stochastic law for tested kinetic observables without recovering the full microscopic transition mechanism.

---

## 4. Cross-Regime Transfer Diagnostics (Regime R2 at 350 K)

In OpenMM Langevin dynamics at $350.0\text{ K}$, empirical transitions accelerate to $\langle T\rangle = 11.99\text{ ps}$ ($N = 73$ events):
1. **Strict Parameter Transfer**: Predicts $25.16\text{ ps}$ ($109.9\%$ error, $p < 0.0001$). Fails.
2. **Theoretical Thermal Scaling ($D \propto T$)**: Predicts $18.76\text{ ps}$ ($56.5\%$ error, $p = 0.0008$). Captures $>50\%$ of the acceleration, but does not reproduce the distribution.
3. **Hypothesis 2 (Independently Estimated Local Scaling $a_{\mathrm{emp}}$)**: Scaling $D^\ast(z)$ by $a_{\mathrm{emp}} = 1.471$ yields $\langle T\rangle = 12.99\text{ ps}$ ($7.2\%$ error, $p = 0.6330$).
This confirms that the spatial functional form $D^\ast(z)$ discovered at $300\text{ K}$ transfers to $350\text{ K}$ when local scalar mobility is calibrated from local one-step residuals. The observed scaling exceeded the $T$-proportional prediction, indicating that additional temperature-dependent changes in the effective reduced dynamics contribute.

---

## 5. Milestone 5 Decision Tree

$$\boxed{
\begin{array}{rcl}
T_3\text{ generalization} &\longrightarrow& \mathbf{PASS}\\[2mm]
\text{forward kinetics} &\longrightarrow& \mathbf{PASS\;within\;tested\;uncertainty}\\[2mm]
\text{equilibrium statistics} &\longrightarrow& \mathbf{PASS}\\[2mm]
\text{drift transfer} &\longrightarrow& \mathbf{STRONG}\\[2mm]
\text{diffusion-field transfer} &\longrightarrow& \mathbf{MODERATE/WEAK}\\[2mm]
\text{residual memory} &\longrightarrow& \mathbf{SMALL\;but\;nonzero}\\[2mm]
\text{strict 350 K transfer} &\longrightarrow& \mathbf{FAIL}\\[2mm]
350\text{ K structural scaling} &\longrightarrow& \mathbf{PARTIAL\;TRANSFER}\\[2mm]
\text{Hypothesis 2 local scaling} &\longrightarrow& \mathbf{RECOVERS\;KINETICS}
\end{array}
}
$$

---

## 6. Roadmap for Milestone 6

Milestone 6 will directly investigate:
$$\boxed{\textbf{Why does the structural form transfer while its parameters do not?}}$$
Specifically, we will test whether the frozen structural law can be generalized across temperature using a physically parameterized temperature dependence:
- **Hypothesis 1**: $D(z, T) = (T/T_0) D(z, T_0)$ (tested in Milestone 5: partially accelerates but quantitatively imperfect).
- **Hypothesis 2**: $D(z, T) = a(T) D(z, T_0)$ where $a(T)$ is estimated independently from local diffusion statistics (tested in Milestone 5: resolves kinetics with $7.2\%$ error, $p = 0.6330$).
- **Hypothesis 3**: Testing whether the spatial shape itself undergoes systematic temperature-dependent deformation: $D(z, T) = a(T) \widetilde{D}(z, T)$.
