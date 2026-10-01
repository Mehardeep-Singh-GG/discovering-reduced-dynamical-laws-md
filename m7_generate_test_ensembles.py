import os
import time
import json
import numpy as np
import openmm as mm
from openmm import app, unit

print("=== Generating Milestone 7 Independent and Adversarial MD Ensembles ===")

pdb_file = 'alanine_dipeptide.pdb'
pdb = app.PDBFile(pdb_file)

# Force field: AMBER ff99SB-ILDN with OBC2 implicit solvent
forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
system = forcefield.createSystem(pdb.topology, constraints=app.HBonds)

# Find backbone dihedral atom indices
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
    if b1_norm > 0:
        b1 /= b1_norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)

frames_to_save = 3000 # 3.0 ns per trajectory
steps_per_frame = 500  # 500 * 2 fs = 1.0 ps frame interval
timestep = 2.0 * unit.femtosecond

os.makedirs('milestone7/data/independent_md', exist_ok=True)
os.makedirs('milestone7/data/adversarial_md', exist_ok=True)
os.makedirs('milestone7/data/metadata', exist_ok=True)

manifest_entries = []

# -------------------------------------------------------------
# Part 1: Layer B — Independent Trajectories (300, 325, 350, 375 K)
# -------------------------------------------------------------
print("\n--- Part 1: Generating Layer B Independent Ensembles (4 temperatures x 3 reps) ---")
temperatures = [300.0, 325.0, 350.0, 375.0]
replicates = [1, 2, 3]
friction_std = 1.0 / unit.picosecond

for temp in temperatures:
    temp_u = temp * unit.kelvin
    for rep in replicates:
        out_fn = f"milestone7/data/independent_md/alanine_dipeptide_{int(temp)}K_test_rep{rep}.npz"
        seed = 90000 + int(temp) * 10 + rep
        
        if os.path.exists(out_fn):
            print(f"  [Exists] {out_fn}")
        else:
            t0 = time.time()
            integrator = mm.LangevinMiddleIntegrator(temp_u, friction_std, timestep)
            integrator.setRandomNumberSeed(seed)
            sim = app.Simulation(pdb.topology, system, integrator)
            sim.context.setPositions(pdb.positions)
            sim.minimizeEnergy(maxIterations=100)
            sim.context.setVelocitiesToTemperature(temp_u, seed)
            sim.step(2500) # 5 ps thermalization
            
            dihedrals = np.zeros((frames_to_save, 2), dtype=np.float32)
            for frame in range(frames_to_save):
                sim.step(steps_per_frame)
                state = sim.context.getState(getPositions=True)
                pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                dihedrals[frame, 0] = compute_dihedral(pos, *phi_atoms)
                dihedrals[frame, 1] = compute_dihedral(pos, *psi_atoms)
                
            np.savez_compressed(out_fn, arr_0=dihedrals)
            dur = time.time() - t0
            print(f"  [Generated] {out_fn} in {dur:.2f}s (Frames: {frames_to_save}, Temp: {temp}K, Seed: {seed})")
            
        manifest_entries.append({
            'type': 'Layer_B_Independent_Test',
            'file': out_fn,
            'temperature_K': temp,
            'friction_ps_inv': 1.0,
            'replicate': rep,
            'frames': frames_to_save,
            'length_ns': frames_to_save * 0.001,
            'seed': seed
        })

# -------------------------------------------------------------
# Part 2: Layer D — Adversarial 400 K Extrapolation (3 reps)
# -------------------------------------------------------------
print("\n--- Part 2: Generating Layer D Adversarial 400 K Extrapolation (3 reps) ---")
for rep in [1, 2, 3]:
    out_fn = f"milestone7/data/adversarial_md/alanine_dipeptide_400K_rep{rep}.npz"
    seed = 94000 + rep
    temp_400 = 400.0 * unit.kelvin
    
    if os.path.exists(out_fn):
        print(f"  [Exists] {out_fn}")
    else:
        t0 = time.time()
        integrator = mm.LangevinMiddleIntegrator(temp_400, friction_std, timestep)
        integrator.setRandomNumberSeed(seed)
        sim = app.Simulation(pdb.topology, system, integrator)
        sim.context.setPositions(pdb.positions)
        sim.minimizeEnergy(maxIterations=100)
        sim.context.setVelocitiesToTemperature(temp_400, seed)
        sim.step(2500)
        
        dihedrals = np.zeros((frames_to_save, 2), dtype=np.float32)
        for frame in range(frames_to_save):
            sim.step(steps_per_frame)
            state = sim.context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            dihedrals[frame, 0] = compute_dihedral(pos, *phi_atoms)
            dihedrals[frame, 1] = compute_dihedral(pos, *psi_atoms)
            
        np.savez_compressed(out_fn, arr_0=dihedrals)
        dur = time.time() - t0
        print(f"  [Generated] {out_fn} in {dur:.2f}s (Frames: {frames_to_save}, Temp: 400K, Seed: {seed})")
        
    manifest_entries.append({
        'type': 'Layer_D_Adversarial_400K',
        'file': out_fn,
        'temperature_K': 400.0,
        'friction_ps_inv': 1.0,
        'replicate': rep,
        'frames': frames_to_save,
        'length_ns': frames_to_save * 0.001,
        'seed': seed
    })

# -------------------------------------------------------------
# Part 3: Layer D — Adversarial Friction Perturbations (gamma = 0.5, 2.0 ps^-1 at 300 K)
# -------------------------------------------------------------
print("\n--- Part 3: Generating Layer D Adversarial Friction Perturbations (300 K, gamma = 0.5 & 2.0 ps^-1) ---")
for gamma_val, label in [(0.5, "gamma0.5"), (2.0, "gamma2.0")]:
    friction_pert = (gamma_val / unit.picosecond)
    temp_300 = 300.0 * unit.kelvin
    for rep in [1, 2, 3]:
        out_fn = f"milestone7/data/adversarial_md/alanine_dipeptide_{label}_rep{rep}.npz"
        seed = int(90000 + gamma_val * 1000 + rep)
        
        if os.path.exists(out_fn):
            print(f"  [Exists] {out_fn}")
        else:
            t0 = time.time()
            integrator = mm.LangevinMiddleIntegrator(temp_300, friction_pert, timestep)
            integrator.setRandomNumberSeed(seed)
            sim = app.Simulation(pdb.topology, system, integrator)
            sim.context.setPositions(pdb.positions)
            sim.minimizeEnergy(maxIterations=100)
            sim.context.setVelocitiesToTemperature(temp_300, seed)
            sim.step(2500)
            
            dihedrals = np.zeros((frames_to_save, 2), dtype=np.float32)
            for frame in range(frames_to_save):
                sim.step(steps_per_frame)
                state = sim.context.getState(getPositions=True)
                pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                dihedrals[frame, 0] = compute_dihedral(pos, *phi_atoms)
                dihedrals[frame, 1] = compute_dihedral(pos, *psi_atoms)
                
            np.savez_compressed(out_fn, arr_0=dihedrals)
            dur = time.time() - t0
            print(f"  [Generated] {out_fn} in {dur:.2f}s (Frames: {frames_to_save}, Gamma: {gamma_val}/ps, Seed: {seed})")
            
        manifest_entries.append({
            'type': f'Layer_D_Adversarial_Friction_{label}',
            'file': out_fn,
            'temperature_K': 300.0,
            'friction_ps_inv': gamma_val,
            'replicate': rep,
            'frames': frames_to_save,
            'length_ns': frames_to_save * 0.001,
            'seed': seed
        })

# Save metadata manifest
with open('milestone7/data/metadata/provenance_manifest.json', 'w') as f:
    json.dump({
        'milestone': 'Milestone 7 Blind Test Data',
        'engine': 'OpenMM 8.x LangevinMiddleIntegrator',
        'forcefield': 'AMBER ff99SB-ILDN with OBC2 implicit solvent',
        'total_trajectories': len(manifest_entries),
        'total_simulated_time_ns': len(manifest_entries) * frames_to_save * 0.001,
        'trajectories': manifest_entries
    }, f, indent=4)

print(f"\nAll {len(manifest_entries)} Milestone 7 MD trajectories generated and recorded successfully.")

