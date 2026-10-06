#!/usr/bin/env python3
"""
Macromolecular Atomistic Molecular Dynamics Simulation Pipeline
===============================================================
Generates physical all-atom molecular dynamics trajectories in implicit/explicit
solvent using OpenMM and extracts collective backbone dihedral angles (phi, psi).

Usage:
------
    python scripts/simulate_trajectories.py --system chignolin --temp 300.0 --ns 10.0
"""

import argparse
import os
import sys
import time
import numpy as np

try:
    import openmm
    from openmm import app, unit
    import mdtraj as md
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False


SYSTEM_PDB_MAP = {
    'alanine_dipeptide': 'data/structures/alanine_dipeptide.pdb',
    'trialanine': 'data/structures/trialanine.pdb',
    'chignolin': 'data/structures/chignolin.pdb',
    'trpcage': 'data/structures/trpcage.pdb',
    'villin': 'data/structures/villin.pdb',
    'gb1': 'data/structures/gb1.pdb',
    'ubiquitin': 'data/structures/ubiquitin.pdb'
}


def run_atomistic_simulation(
    pdb_path: str,
    output_npz_path: str,
    temperature_k: float = 300.0,
    friction_ps: float = 1.0,
    time_step_fs: float = 2.0,
    total_ns: float = 5.0,
    report_stride: int = 500,  # 1.0 ps frame interval at 2 fs time step
    seed: int = 42,
    platform_name: str = 'CPU'
):
    """Executes MD simulation and extracts dihedral conformational coordinates."""
    if not OPENMM_AVAILABLE:
        raise RuntimeError("OpenMM and MDTraj are required to execute new simulations. Install via: conda install -c conda-forge openmm mdtraj")

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB structure not found at {pdb_path}")

    total_steps = int(round((total_ns * 1e6) / time_step_fs))
    n_frames = total_steps // report_stride

    print(f"=== OpenMM Production Simulation ===")
    print(f"PDB Structure: {pdb_path}")
    print(f"Temperature:   {temperature_k} K")
    print(f"Length:        {total_ns:.2f} ns ({n_frames} frames at {report_stride * time_step_fs / 1000.0:.1f} ps stride)")

    pdb = app.PDBFile(pdb_path)
    ff = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)

    integrator = openmm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin,
        friction_ps / unit.picosecond,
        time_step_fs * unit.femtoseconds
    )
    integrator.setRandomNumberSeed(seed)

    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
    except Exception:
        platform = openmm.Platform.getPlatformByName('CPU')

    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)

    print("Energy minimizing structure...")
    simulation.minimizeEnergy(maxIterations=500)

    print(f"Simulating {n_frames} frames...")
    positions_list = []
    time_list = []

    t0 = time.time()
    for frame_idx in range(n_frames):
        simulation.step(report_stride)
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        positions_list.append(pos)
        time_list.append(state.getTime().value_in_unit(unit.picoseconds))

        if (frame_idx + 1) % max(1, n_frames // 10) == 0:
            elapsed = time.time() - t0
            pct = (frame_idx + 1) / n_frames * 100.0
            print(f"  Frame {frame_idx + 1}/{n_frames} ({pct:.0f}%) | Elapsed: {elapsed:.1f}s")

    positions = np.array(positions_list, dtype=np.float32)
    times = np.array(time_list, dtype=np.float32)

    print("Computing backbone conformational dihedrals (phi, psi)...")
    traj_md = md.Trajectory(positions, md.Topology.from_openmm(pdb.topology))
    _, phi = md.compute_phi(traj_md)
    _, psi = md.compute_psi(traj_md)
    coords = np.hstack([phi, psi]).astype(np.float32)

    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    np.savez_compressed(
        output_npz_path,
        coordinates=coords,
        phi=phi,
        psi=psi,
        times=times,
        dt_ps=float(report_stride * time_step_fs / 1000.0),
        temperature_k=temperature_k,
        dim=coords.shape[1],
        n_frames=coords.shape[0]
    )
    print(f"Saved trajectory: {output_npz_path} ({coords.shape[0]} frames, {coords.shape[1]} dimensions)")


def main():
    parser = argparse.ArgumentParser(description="Simulate atomistic MD trajectories using OpenMM.")
    parser.add_argument('--system', choices=list(SYSTEM_PDB_MAP.keys()), default='chignolin', help="Target macromolecular system")
    parser.add_argument('--temp', type=float, default=300.0, help="Simulation temperature in Kelvin")
    parser.add_argument('--ns', type=float, default=5.0, help="Total simulation duration in nanoseconds")
    parser.add_argument('--output', type=str, default=None, help="Output NPZ filepath")
    args = parser.parse_args()

    pdb_file = SYSTEM_PDB_MAP[args.system]
    output_file = args.output or f"data/trajectories/{args.system}_trajectory.npz"
    run_atomistic_simulation(pdb_path=pdb_file, output_npz_path=output_file, temperature_k=args.temp, total_ns=args.ns)


if __name__ == '__main__':
    main()
