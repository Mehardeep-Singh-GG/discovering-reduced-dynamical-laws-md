import sys, os, time
import numpy as np
import openmm as mm
from openmm import app, unit

out_fn = sys.argv[1]
temp_val = float(sys.argv[2])
gamma_val = float(sys.argv[3])
seed = int(sys.argv[4])
frames_to_save = int(sys.argv[5]) if len(sys.argv) > 5 else 2500

if os.path.exists(out_fn):
    try:
        d = np.load(out_fn)
        if d['arr_0'].shape == (frames_to_save, 2):
            print(f"[Exists] {out_fn}")
            sys.exit(0)
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
sim.step(2500)

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
print(f"[Done] {out_fn} in {dur:.1f}s")
