import os
import time
import numpy as np
import openmm as mm
from openmm import app, unit

print("=== Generating Authentic Alanine Dipeptide MD Trajectory for Regime R2 (T = 350.0 K) ===")

pdb_file = 'alanine_dipeptide.pdb'
pdb = app.PDBFile(pdb_file)

# Force field: AMBER ff99SB-ILDN with OBC2 implicit solvent
forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
system = forcefield.createSystem(pdb.topology, constraints=app.HBonds)

# Find backbone dihedral atom indices
atom_indices = {}
for atom in pdb.topology.atoms():
    atom_indices[(atom.residue.name, atom.name)] = atom.index

# phi: C(ACE 1) - N(ALA 2) - CA(ALA 2) - C(ALA 2)
phi_atoms = [atom_indices[('ACE', 'C')], atom_indices[('ALA', 'N')], atom_indices[('ALA', 'CA')], atom_indices[('ALA', 'C')]]
# psi: N(ALA 2) - CA(ALA 2) - C(ALA 2) - N(NME 3)
psi_atoms = [atom_indices[('ALA', 'N')], atom_indices[('ALA', 'CA')], atom_indices[('ALA', 'C')], atom_indices[('NME', 'N')]]

def compute_dihedral(pos, i, j, k, l):
    p0 = pos[i]
    p1 = pos[j]
    p2 = pos[k]
    p3 = pos[l]
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

frames_to_save = 2500
steps_per_frame = 500  # 500 steps * 2.0 fs = 1.0 ps frame spacing
temperature = 350.0 * unit.kelvin
friction = 1.0 / unit.picosecond
timestep = 2.0 * unit.femtosecond

integrator = mm.LangevinMiddleIntegrator(temperature, friction, timestep)
integrator.setRandomNumberSeed(5000)

simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

# Minimize energy
simulation.minimizeEnergy(maxIterations=200)

# Thermalize to 350 K
simulation.context.setVelocitiesToTemperature(temperature, 5000)
simulation.step(2500) # 5 ps thermalization

dihedrals = np.zeros((frames_to_save, 2), dtype=np.float32)

t0 = time.time()
print("Starting 2.5 ns simulation at 350 K...")
for frame in range(frames_to_save):
    simulation.step(steps_per_frame)
    state = simulation.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    phi = compute_dihedral(pos, *phi_atoms)
    psi = compute_dihedral(pos, *psi_atoms)
    dihedrals[frame] = [phi, psi]
    
    if (frame + 1) % 500 == 0:
        print(f"  Frame {frame + 1}/{frames_to_save} completed (Elapsed: {time.time() - t0:.1f}s)")

out_file = 'alanine-dipeptide-regime2-350K.npz'
np.savez_compressed(out_file, arr_0=dihedrals)
print(f"Successfully saved Regime R2 trajectory to {out_file} ({time.time() - t0:.1f}s total).")
