"""
Stochastic Differential Operator Discovery
==========================================
Estimates continuous drift vector f(z) and state-dependent diffusion tensor D(z)
from multidimensional atomistic coordinate trajectories z(t) in R^d:

    dz_t = f(z_t) dt + sqrt(2 * D(z_t)) dW_t

Discretized Kramers-Moyal generator moments:
    f_i(z) = lim_{tau -> 0} (1 / tau) * E[ z_i(t + tau) - z_i(t) | z(t) = z ]
    D_ij(z) = lim_{tau -> 0} (1 / (2 tau)) * E[ (z_i(t + tau) - z_i(t))(z_j(t + tau) - z_j(t)) | z(t) = z ]
"""

import numpy as np


class StochasticOperatorEstimator:
    """
    Estimator for continuous stochastic differential operators on macromolecular manifolds.
    """

    def __init__(self, lag_time_ps: float = 0.5, periodic: bool = True):
        self.lag_time = lag_time_ps
        self.periodic = periodic

    def displacement(self, z_t: np.ndarray, z_next: np.ndarray) -> np.ndarray:
        """Computes coordinate displacement, handling periodic angular boundaries [-pi, pi)."""
        dz = z_next - z_t
        if self.periodic:
            dz = (dz + np.pi) % (2.0 * np.pi) - np.pi
        return dz

    def estimate_drift_and_diffusion(self, z_traj: np.ndarray, dt_ps: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimates the empirical drift vector and global/local diffusion tensor.

        Parameters
        ----------
        z_traj : np.ndarray, shape (N_frames, d)
            Coordinate trajectory across d conformational dimensions.
        dt_ps : float
            Trajectory sampling interval in picoseconds.

        Returns
        -------
        drift : np.ndarray, shape (d,)
            Mean conformational drift vector.
        diffusion : np.ndarray, shape (d, d)
            Positive semi-definite diffusion tensor.
        """
        assert z_traj.ndim == 2, "Trajectory must be 2D array (N_frames, d)"
        n_frames, d = z_traj.shape

        step = max(1, int(round(self.lag_time / dt_ps)))
        eff_tau = step * dt_ps

        z_t = z_traj[:-step]
        z_next = z_traj[step:]
        dz = self.displacement(z_t, z_next)

        drift = np.mean(dz, axis=0) / eff_tau
        diff_raw = (dz.T @ dz) / (2.0 * eff_tau * len(dz))

        # Enforce exact mathematical symmetry: D = 0.5 * (D + D^T)
        diffusion = 0.5 * (diff_raw + diff_raw.T)

        # Spectral projection to guarantee strict positive semi-definiteness
        eigvals, eigvecs = np.linalg.eigh(diffusion)
        eigvals_pos = np.clip(eigvals, a_min=1e-8, a_max=None)
        diffusion = eigvecs @ np.diag(eigvals_pos) @ eigvecs.T

        return drift, diffusion
