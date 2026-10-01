import os
import time
import json
import numpy as np
import pandas as pd
import openmm as mm
from openmm import app, unit

print("=== Generating Multi-Temperature Alanine Dipeptide Ensembles for Milestone 6 ===")

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

temperatures = [300.0, 325.0, 350.0, 375.0]
replicates = [1, 2, 3] # 1: Calibration, 2: Validation, 3: Held-Out Test
frames_to_save = 2500
steps_per_frame = 500  # 500 steps * 2.0 fs = 1.0 ps
timestep = 2.0 * unit.femtosecond
friction = 1.0 / unit.picosecond

manifest_rows = []
provenance_dict = {
    'milestone': 'Milestone 6',
    'simulation_engine': 'OpenMM',
    'forcefield': 'AMBER ff99SB-ILDN with OBC2 implicit solvent (amber99sbildn.xml, amber99_obc.xml)',
    'system': 'Alanine Dipeptide (ACE-ALA-NME)',
    'integrator': 'LangevinMiddleIntegrator',
    'friction_per_ps': 1.0,
    'timestep_fs': 2.0,
    'steps_per_frame': steps_per_frame,
    'frame_spacing_ps': 1.0,
    'frames_per_trajectory': frames_to_save,
    'trajectory_length_ns': frames_to_save * 1.0 / 1000.0,
    'temperatures_K': temperatures,
    'role_mapping': {
        'traj1': 'Calibration (local field and scaling parameter estimation)',
        'traj2': 'Validation (intermediate check and variance monitoring)',
        'traj3': 'Held-Out Test (strictly untouched blind kinetic prediction)'
    },
    'trajectories': {}
}

os.makedirs('milestone6/data', exist_ok=True)
total_start = time.time()

for temp in temperatures:
    temp_unit = temp * unit.kelvin
    print(f"\n--- Temperature {temp} K ---")
    
    for rep in replicates:
        role = 'Calibration' if rep == 1 else ('Validation' if rep == 2 else 'Held_Out_Test')
        seed = int(temp * 10) + rep
        out_filename = f"milestone6/data/alanine_dipeptide_{int(temp)}K_traj{rep}.npz"
        
        # Check if file exists
        if os.path.exists(out_filename):
            print(f"  {out_filename} already exists, loading and verifying...")
            d = np.load(out_filename)
            arr = d['arr_0']
            assert arr.shape == (frames_to_save, 2), f"Corrupted shape {arr.shape}"
            sim_time = 0.0
        else:
            t0 = time.time()
            integrator = mm.LangevinMiddleIntegrator(temp_unit, friction, timestep)
            integrator.setRandomNumberSeed(seed)
            
            simulation = app.Simulation(pdb.topology, system, integrator)
            simulation.context.setPositions(pdb.positions)
            simulation.minimizeEnergy(maxIterations=200)
            simulation.context.setVelocitiesToTemperature(temp_unit, seed)
            simulation.step(2500) # 5 ps thermalization
            
            dihedrals = np.zeros((frames_to_save, 2), dtype=np.float32)
            for frame in range(frames_to_save):
                simulation.step(steps_per_frame)
                state = simulation.context.getState(getPositions=True)
                pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
                phi = compute_dihedral(pos, *phi_atoms)
                psi = compute_dihedral(pos, *psi_atoms)
                dihedrals[frame] = [phi, psi]
            
            np.savez_compressed(out_filename, arr_0=dihedrals, dihedrals=dihedrals)
            sim_time = time.time() - t0
            print(f"  Generated {out_filename} (Seed={seed}, Role={role}) in {sim_time:.1f}s")
        
        manifest_rows.append({
            'Temperature_K': temp,
            'Replicate': rep,
            'Role': role,
            'File_Path': out_filename,
            'Frames': frames_to_save,
            'Spacing_ps': 1.0,
            'Length_ns': frames_to_save * 1.0 / 1000.0,
            'Random_Seed': seed
        })
        
        provenance_dict['trajectories'][f"{int(temp)}K_traj{rep}"] = {
            'file': out_filename,
            'temperature_K': temp,
            'replicate': rep,
            'role': role,
            'seed': seed,
            'frames': frames_to_save,
            'spacing_ps': 1.0,
            'duration_ns': 2.5
        }

# Save manifest and provenance
df_manifest = pd.DataFrame(manifest_rows)
df_manifest.to_csv('milestone6/data/trajectory_manifest.csv', index=False)

with open('milestone6/data/provenance.json', 'w') as f_prov:
    json.dump(provenance_dict, f_prov, indent=4)

temp_meta = [
    {'Temperature_K': 300.0, 'Thermal_Scale_T_over_T0': 1.0000, 'kT_kcal_mol': 0.5962, 'Inverse_kT_mol_kcal': 1.6773},
    {'Temperature_K': 325.0, 'Thermal_Scale_T_over_T0': 1.0833, 'kT_kcal_mol': 0.6459, 'Inverse_kT_mol_kcal': 1.5482},
    {'Temperature_K': 350.0, 'Thermal_Scale_T_over_T0': 1.1667, 'kT_kcal_mol': 0.6955, 'Inverse_kT_mol_kcal': 1.4378},
    {'Temperature_K': 375.0, 'Thermal_Scale_T_over_T0': 1.2500, 'kT_kcal_mol': 0.7452, 'Inverse_kT_mol_kcal': 1.3419},
]
pd.DataFrame(temp_meta).to_csv('milestone6/data/temperature_metadata.csv', index=False)

print(f"\nAll 12 ensemble trajectories successfully verified and saved in {time.time() - total_start:.1f}s.")
print("Saved manifest to milestone6/data/trajectory_manifest.csv")
print("Saved provenance to milestone6/data/provenance.json")
print("Saved temperature metadata to milestone6/data/temperature_metadata.csv")
