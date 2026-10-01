=============================================================================
MILESTONE 7 FROZEN REDUCED DYNAMICAL MODEL
=============================================================================
This directory contains the completely frozen and immutable reduced dynamical
operator governing alanine dipeptide conformational transitions across temperature:

    L_T = f_0(z) * grad + div * [ a(T) * D0(z) + b(T) * G1(z) ] * grad

where:
    a(T) = (T / 300.0)^1.5731
    b(T) = -0.09607 * (T - 300.0) rad^2/(ps*K)

All coefficients, basis definitions, spatial grids, and parameters were sealed
prior to executing any Milestone 7 evaluations. No fitting or hyperparameter
tuning is permitted on Milestone 7 test data.

Verification:
    shasum -a 256 -c SHA256SUMS.txt
=============================================================================
