"""
Reduced Dynamics: Continuous Stochastic Differential Operators from Atomistic MD
================================================================================
A unified theoretical and computational framework for discovering continuous,
thermodynamically consistent reduced stochastic differential equations directly
from atomistic molecular dynamics trajectories across macromolecular scales.

Modules:
- operator: Continuous drift vector and diffusion tensor estimation.
- generic: Structure-preserving GENERIC decomposition enforcing detailed balance.
- mori_zwanzig: Extended state space (z, v, w) resolving Kramers low-friction turnover.
- sparse: Locality-banded sparse estimation with tertiary contact gating (O(d^0.99)).
- scaling: Effective rank scaling (r90 proportional to d^0.89) and bootstrap uncertainty.
- transfer: Cross-protein transferability, invariant interior mobility, and B-factor concordance.
"""

__version__ = "1.0.0"
__authors__ = ["Mehardeep Singh", "Gurarpit Singh"]
__license__ = "MIT"

from .operator import StochasticOperatorEstimator
from .generic import GENERICDriftRegularizer
from .mori_zwanzig import ExtendedMoriZwanzigIntegrator
from .sparse import LocalityBandedSparseEstimator
from .scaling import RankScalingAnalyzer
from .transfer import MobilityTransferValidator

__all__ = [
    "StochasticOperatorEstimator",
    "GENERICDriftRegularizer",
    "ExtendedMoriZwanzigIntegrator",
    "LocalityBandedSparseEstimator",
    "RankScalingAnalyzer",
    "MobilityTransferValidator",
]
