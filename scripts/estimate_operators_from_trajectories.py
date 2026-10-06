#!/usr/bin/env python3
"""
Stochastic Differential Operator Estimation from Trajectories
============================================================
Demonstrates how to fit continuous drift f(z), state-dependent diffusion D(z),
GENERIC detailed-balance regularization, and sub-linear rank scaling directly
from atomistic conformational coordinate trajectories in data/trajectories/.

Usage:
------
    python scripts/estimate_operators_from_trajectories.py --system chignolin
    python scripts/estimate_operators_from_trajectories.py --all
"""

import argparse
import glob
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from reduced_dynamics.operator import StochasticOperatorEstimator
from reduced_dynamics.generic import GENERICDriftRegularizer
from reduced_dynamics.scaling import RankScalingAnalyzer
from reduced_dynamics.sparse import LocalityBandedSparseEstimator


def analyze_system(npz_path: str):
    data = np.load(npz_path)
    coords = data['coordinates']
    dt_ps = float(data['dt_ps'])
    system = str(data.get('system', os.path.basename(npz_path).replace('_trajectory.npz', '')))
    n_frames, d = coords.shape

    print(f"\n{'='*75}")
    print(f"SYSTEM: {system.upper():18s} | Dimensions: {d:3d} | Frames: {n_frames:5d} | dt: {dt_ps} ps")
    print(f"{'='*75}")

    # 1. Continuous Operator Estimation (Drift & Diffusion)
    estimator = StochasticOperatorEstimator(lag_time_ps=1.0)
    drift, diffusion = estimator.estimate_drift_and_diffusion(coords, dt_ps=dt_ps)
    print(f"[1] Generator Discovery (Kramers-Moyal):")
    print(f"    - Drift norm ||f||_2:       {np.linalg.norm(drift):.4f} rad/ps")
    print(f"    - Trace(D):                 {np.trace(diffusion):.4f} rad^2/ps")
    print(f"    - Mean diagonal D_ii:       {np.mean(np.diag(diffusion)):.4f} rad^2/ps")

    # 2. Sub-linear Rank Scaling
    rank_est = RankScalingAnalyzer()
    r90 = rank_est.calculate_effective_rank(diffusion, variance_threshold=0.90)
    pr = rank_est.calculate_participation_ratio(diffusion)
    print(f"[2] Dimensional Compression & Rank Scaling:")
    print(f"    - Effective Rank (r_90):     {r90} / {d} ({r90/d*100:.1f}%)")
    print(f"    - Participation Ratio (PR): {pr:.2f}")

    # 3. GENERIC Detailed Balance Enforcement
    generic = GENERICDriftRegularizer()
    grad_F = -drift / (np.clip(np.diag(diffusion), 1e-6, None) * generic.beta)
    f_generic = generic.project_detailed_balance_drift(grad_F, diffusion)
    err = np.linalg.norm(drift - f_generic) / (np.linalg.norm(drift) + 1e-12)
    corr = np.corrcoef(drift, f_generic)[0, 1] if len(drift) > 1 else 1.0
    print(f"[3] Thermodynamic Consistency (GENERIC):")
    print(f"    - Dissipative alignment:    {corr:.4f}")
    print(f"    - Relative residual:        {err:.4f}")

    # 4. Sparsity and Dynamic Locality
    diag_max = np.max(np.diag(diffusion))
    off_diag = diffusion[~np.eye(d, dtype=bool)] if d > 1 else np.array([0.0])
    sparsity = float(np.mean(np.abs(off_diag) < 0.05 * diag_max))
    print(f"[4] Spatial Locality & Sparsity:")
    print(f"    - Off-diagonal Sparsity:    {sparsity*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Fit reduced stochastic operators from real MD trajectories.")
    parser.add_argument('--system', type=str, default='chignolin', help="Target system (e.g. chignolin, trpcage, villin)")
    parser.add_argument('--all', action='store_true', help="Process all available trajectories in data/trajectories/")
    args = parser.parse_args()

    if args.all:
        traj_files = sorted(glob.glob('data/trajectories/*_trajectory.npz'))
        for f in traj_files:
            analyze_system(f)
    else:
        target = f"data/trajectories/{args.system}_trajectory.npz"
        if not os.path.exists(target):
            available = [os.path.basename(p).replace('_trajectory.npz', '') for p in glob.glob('data/trajectories/*_trajectory.npz')]
            print(f"Error: {target} not found. Available systems: {available}")
            return
        analyze_system(target)


if __name__ == '__main__':
    main()
