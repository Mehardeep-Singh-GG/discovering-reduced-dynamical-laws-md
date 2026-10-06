"""
Extended Mori-Zwanzig State Space Embedding
===========================================
Cures the low-friction Kramers turnover and resolves non-Markovian memory friction
without solving computationally intractable history integrals.

When high-dimensional atomistic dynamics are projected onto a lower-dimensional
manifold z in R^d, the Generalized Langevin Equation (GLE) reads:

    dz_t / dt = v_t
    dv_t / dt = f_0(z_t) - int_0^t K(t - s) v_s ds + xi_t

By representing the memory kernel as an exponential Markovian bath:
    K(t) = (alpha_m / tau_m) * exp(-t / tau_m)

we expand the state space into the Markovian triplet (z_t, v_t, w_t) in R^{3d}:

    dz_t = v_t dt
    dv_t = [ f_0(z_t) - gamma_eff * v_t + w_t ] dt + sqrt(2 * gamma_eff * D(z_t)) dW_t^v
    dw_t = [ - w_t / tau_m - (alpha_m / tau_m) * v_t ] dt + sqrt(2 * alpha_m * D(z_t) / tau_m) dW_t^w
"""

import numpy as np


class ExtendedMoriZwanzigIntegrator:
    """
    Markovian state space embedding for non-Markovian macromolecular kinetics.
    """

    def __init__(
        self,
        tau_m: float = 3.1311,       # Memory relaxation time (ps)
        alpha_m: float = 15.00,      # Memory coupling strength
        gamma_eff: float = 1.70,     # Effective friction coefficient (ps^-1)
        dt_ps: float = 0.005         # Numerical integration step (ps)
    ):
        self.tau_m = tau_m
        self.alpha_m = alpha_m
        self.gamma_eff = gamma_eff
        self.dt = dt_ps

    def step(
        self,
        z: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        drift_fn,
        diffusion_tensor: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes a single structure-preserving Euler-Maruyama step on (z, v, w).
        """
        d = len(z)
        dt = self.dt
        sq_dt = np.sqrt(dt)

        f_z = drift_fn(z)
        l_diff = np.linalg.cholesky(diffusion_tensor)

        xi_v = l_diff @ np.random.randn(d) * sq_dt
        xi_w = l_diff @ np.random.randn(d) * sq_dt

        # Update position
        z_new = z + v * dt

        # Update velocity (resolving memory force w)
        v_new = v + (f_z - self.gamma_eff * v + w) * dt + np.sqrt(2.0 * self.gamma_eff) * xi_v

        # Update auxiliary bath mode
        w_new = w + (-w / self.tau_m - (self.alpha_m / self.tau_m) * v) * dt + np.sqrt(2.0 * self.alpha_m / self.tau_m) * xi_w

        return z_new, v_new, w_new
