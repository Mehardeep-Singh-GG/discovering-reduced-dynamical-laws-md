import os
import sys
import time
import json
import numpy as np
import openmm as mm
from openmm import app, unit
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_single_md(task):
    out_fn = task['file']
    temp_val = task['temperature_K']
    gamma_val = task['friction_ps_inv']
    seed = task['seed']
    frames_to_save = task['frames']

    if os.path.exists(out_fn):
        try:
            d = np.load(out_fn)
            if d['arr_0'].shape == (frames_to_save, 2):
                print(f"  [Skipping Exists] {out_fn}")
                return task
        except Exception:
            pass

    t0 = time.time()
    pdb = app.PDBFile('alanine_dipeptide.pdb')
    forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    system = forcefield.createSystem(pdb.topology, constraints=app.HBonds)

    atom_indices = {}
    for atom in pdb.topology.atoms():
        atom_indices[(atom.residue.name, atom.name)] = atom.index

    phi_atoms = [atom_indices[('ACE', 'C')], atom_indices[('ALA', 'N')], atom_indices[('ALA', 'CA')], atom_indices[('ALA', 'C')]]
    psi_atoms = [atom_indices[('ALA', 'N')], atom_indices[('ALA', 'CA')], atom_indices[('ALA', 'C')], atom_indices[('NME', 'N')]]

    def compute_dihedral(pos, i, j, k, l):
        p0 = pos[i]; p1 = pos[j]; p2 = pos[k]; p3 = pos[l]
        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        b1_norm = np.linalg.norm(b1)
        if b1_norm > 0: b1 /= b1_norm
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return np.arctan2(y, x)

    integrator = mm.LangevinMiddleIntegrator(temp_val * unit.kelvin, gamma_val / unit.picosecond, 2.0 * unit.femtosecond)
    integrator.setRandomNumberSeed(seed)
    sim = app.Simulation(pdb.topology, system, integrator)
    sim.context.setPositions(pdb.positions)
    sim.minimizeEnergy(maxIterations=100)
    sim.context.setVelocitiesToTemperature(temp_val * unit.kelvin, seed)
    sim.step(2500) # 5 ps thermalization

    steps_per_frame = 500
    dihedrals = np.zeros((frames_to_save, 2), dtype=np.float32)
    for frame in range(frames_to_save):
        sim.step(steps_per_frame)
        state = sim.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        dihedrals[frame, 0] = compute_dihedral(pos, *phi_atoms)
        dihedrals[frame, 1] = compute_dihedral(pos, *psi_atoms)

    dirname = os.path.dirname(out_fn)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    np.savez_compressed(out_fn, arr_0=dihedrals)
    dur = time.time() - t0
    print(f"  [Done] {out_fn} in {dur:.1f}s (T={temp_val}K, gamma={gamma_val}/ps, Seed={seed})")
    return task

if __name__ == '__main__':
    print("=== Launching Milestone 7 Parallel Trajectory Generation (4 Workers) ===")
    os.makedirs('milestone7/data/independent_md', exist_ok=True)
    os.makedirs('milestone7/data/adversarial_md', exist_ok=True)
    os.makedirs('milestone7/data/metadata', exist_ok=True)

    tasks = []
    frames = 2500

    # 1. Layer B: Independent Ensembles (4 temperatures x 3 reps)
    for temp in [300.0, 325.0, 350.0, 375.0]:
        for rep in [1, 2, 3]:
            tasks.append({
                'type': 'Layer_B_Independent_Test',
                'file': f"milestone7/data/independent_md/alanine_dipeptide_{int(temp)}K_test_rep{rep}.npz",
                'temperature_K': temp,
                'friction_ps_inv': 1.0,
                'replicate': rep,
                'frames': frames,
                'seed': 90000 + int(temp) * 10 + rep
            })

    # 2. Layer D: Adversarial 400 K Extrapolation (3 reps)
    for rep in [1, 2, 3]:
        tasks.append({
            'type': 'Layer_D_Adversarial_400K',
            'file': f"milestone7/data/adversarial_md/alanine_dipeptide_400K_rep{rep}.npz",
            'temperature_K': 400.0,
            'friction_ps_inv': 1.0,
            'replicate': rep,
            'frames': frames,
            'seed': 94000 + rep
        })

    # 3. Layer D: Adversarial Friction Perturbations (gamma = 0.5, 2.0 ps^-1 at 300 K)
    for gamma_val, label in [(0.5, "gamma0.5"), (2.0, "gamma2.0")]:
        for rep in [1, 2, 3]:
            tasks.append({
                'type': f'Layer_D_Adversarial_Friction_{label}',
                'file': f"milestone7/data/adversarial_md/alanine_dipeptide_{label}_rep{rep}.npz",
                'temperature_K': 300.0,
                'friction_ps_inv': gamma_val,
                'replicate': rep,
                'frames': frames,
                'seed': int(90000 + gamma_val * 1000 + rep)
            })

    print(f"Total tasks scheduled: {len(tasks)}")
    t_start = time.time()
    completed = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_single_md, task): task for task in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            completed.append(res)

    manifest_data = {
        'milestone': 'Milestone 7 Blind Test Data',
        'engine': 'OpenMM 8.x LangevinMiddleIntegrator',
        'forcefield': 'AMBER ff99SB-ILDN with OBC2 implicit solvent',
        'total_trajectories': len(completed),
        'total_simulated_time_ns': len(completed) * frames * 0.001,
        'trajectories': completed
    }
    with open('milestone7/data/metadata/provenance_manifest.json', 'w') as f:
        json.dump(manifest_data, f, indent=4)

    total_dur = time.time() - t_start
    print(f"\nAll {len(completed)} trajectories completed in {total_dur:.1f}s.")
