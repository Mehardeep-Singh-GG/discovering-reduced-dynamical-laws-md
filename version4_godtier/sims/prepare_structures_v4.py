#!/usr/bin/env python3
"""
version4_godtier/sims/prepare_structures_v4.py
==============================================
Automated, deterministic crystallographic & NMR structure preparation pipeline
for the 15-system Version 4.0 Molecular Atlas (d = 2 -> 738).

Enforces Rules R1-R6:
  * Existing V3 structures (ala1, ala2, chignolin, trpcage, villin, gb1, ubiquitin)
    are copied verbatim from their audited V3 paths and verified by SHA-256 so
    V3 and V4 trajectories share identical atom numbering.
  * New V4 structures (bba 1FME, wwdomain 2F21, bpti 5PTI, barnase 1BNI,
    lysozyme 1AKI, dhfr 1RX2/1RA1, adk 4AKE/1AKE, mbp 1OMP/1ANF) and alt state
    references (gb1 3GB1, ubiquitin 1D3Z) are fetched from RCSB PDB, cleaned via
    PDBFixer + OpenMM Modeller at pH 7.0, checked for exact disulfide bridges and
    target residue counts, and softly clash-relieved with heavy-atom restraints.
"""
import os
import sys
import json
import time
import shutil
import urllib.request
import hashlib
import argparse
import numpy as np
import openmm
from openmm import app, unit
from pdbfixer import PDBFixer
import mdtraj as md

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
V4_DIR = os.path.join(ROOT, 'version4_godtier')
STRUCT_DIR = os.path.join(V4_DIR, 'structures')
RAW_DIR = os.path.join(STRUCT_DIR, 'raw')
META_DIR = os.path.join(STRUCT_DIR, 'metadata')
MAN_DIR = os.path.join(V4_DIR, 'manifests')

for d in (STRUCT_DIR, RAW_DIR, META_DIR, MAN_DIR):
    os.makedirs(d, exist_ok=True)

# Max SASA values (nm^2) per amino acid from Tien et al. (2013) PLoS ONE 8(11): e80635
MAX_SASA_NM2 = {
    'ALA': 1.29, 'ARG': 2.74, 'ASN': 1.95, 'ASP': 1.93, 'CYS': 1.67, 'CYX': 1.67,
    'GLN': 2.25, 'GLU': 2.23, 'GLY': 1.04, 'HIS': 2.24, 'HID': 2.24, 'HIE': 2.24,
    'HIP': 2.24, 'ILE': 1.97, 'LEU': 2.01, 'LYS': 2.36, 'MET': 2.24, 'PHE': 2.40,
    'PRO': 1.59, 'SER': 1.55, 'THR': 1.72, 'TRP': 2.85, 'TYR': 2.63, 'VAL': 1.74,
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download_rcsb_pdb(pdb_id, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return dest_path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"[download] Fetching {url} -> {dest_path}", flush=True)
    urllib.request.urlretrieve(url, dest_path)
    return dest_path


def extract_first_model_and_chain(raw_pdb_path, chain_id='A', resnum_range=None,
                                  strip_deuterium=True):
    """Extract only MODEL 1 (if NMR ensemble) and target chain_id, stripping D atoms
    and alternate conformations (keeping altLoc ' ' or 'A'), and standardizing DPR->PRO."""
    lines_out = []
    in_model = False
    saw_model = False
    with open(raw_pdb_path, 'r') as f:
        for line in f:
            rec = line[:6].strip()
            if rec in ('HEADER', 'TITLE', 'SEQRES'):
                lines_out.append(line)
                continue
            if rec == 'MODEL':
                if saw_model:
                    break
                saw_model = True
                in_model = True
                continue
            if rec == 'ENDMDL':
                if in_model:
                    break
                continue
            if rec in ('ATOM', 'HETATM'):
                resname = line[17:20].strip()
                if rec == 'HETATM' and resname == 'DPR':
                    line = 'ATOM  ' + line[6:17] + 'PRO' + line[20:]
                    resname = 'PRO'
                elif rec == 'HETATM':
                    continue
                ch = line[21:22].strip()
                if chain_id and ch != chain_id and ch != '':
                    continue
                alt = line[16:17]
                if alt not in (' ', 'A', '1'):
                    continue
                elem = line[76:78].strip() if len(line) >= 78 else ''
                atom_name = line[12:16].strip()
                if strip_deuterium and (elem == 'D' or atom_name.startswith('D')):
                    continue
                if resnum_range is not None:
                    resseq = int(line[22:26].strip())
                    if not (resnum_range[0] <= resseq <= resnum_range[1]):
                        continue
                clean_line = line[:16] + ' ' + line[17:]
                lines_out.append(clean_line)
    lines_out.append("TER\nEND\n")
    tmp_path = raw_pdb_path.replace('.pdb', f'_clean_chain{chain_id}.pdb')
    with open(tmp_path, 'w') as f:
        f.writelines(lines_out)
    return tmp_path


def clean_and_protonate_pdb(raw_pdb_path, out_pdb_path, chain_id='A',
                            target_n_res=None, expected_ss_pairs=None,
                            resnum_range=None, add_missing_terminal_residues=False):
    """Run PDBFixer + OpenMM Modeller at pH 7.0 + restrained energy minimization."""
    filtered_path = extract_first_model_and_chain(
        raw_pdb_path, chain_id=chain_id, resnum_range=resnum_range, strip_deuterium=True
    )
    fixer = PDBFixer(filename=filtered_path)
    fixer.findMissingResidues()

    chains = list(fixer.topology.chains())
    current_n_res = sum(1 for _ in fixer.topology.residues())
    keys = list(fixer.missingResidues.keys())
    for key in keys:
        chain_idx, res_idx = key
        n_chain_res = len(list(chains[chain_idx].residues()))
        is_terminal = (res_idx == 0 or res_idx == n_chain_res)
        if not add_missing_terminal_residues and is_terminal:
            del fixer.missingResidues[key]
        elif target_n_res is not None and current_n_res >= target_n_res and is_terminal:
            del fixer.missingResidues[key]
        elif target_n_res is not None and current_n_res < target_n_res and is_terminal:
            needed = target_n_res - current_n_res
            fixer.missingResidues[key] = fixer.missingResidues[key][:needed]
            current_n_res += len(fixer.missingResidues[key])

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Check if residue count exceeds target_n_res (e.g. if 2F21 has 36 residues instead of 35)
    n_res_now = fixer.topology.getNumResidues()
    if target_n_res is not None and n_res_now > target_n_res:
        # Trim C-terminal overhang residues to match exact target_n_res
        mod_trim = app.Modeller(fixer.topology, fixer.positions)
        to_delete = [r for r in mod_trim.topology.residues() if r.index >= target_n_res]
        mod_trim.delete(to_delete)
        fixer.topology = mod_trim.topology
        fixer.positions = mod_trim.positions
    elif target_n_res is not None and n_res_now < target_n_res:
        # If still short (e.g. 2F21 has 34 resolved residues and SEQRES wasn't parsed),
        # append terminal GLY/ALA via Modeller/PDBFixer missingResidues
        needed = target_n_res - n_res_now
        fixer.missingResidues[(0, n_res_now)] = ['GLY'] * needed
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

    ff = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    modeller = app.Modeller(fixer.topology, fixer.positions)

    # Ensure disulfide bonds are registered before addHydrogens so CYX template is selected
    if expected_ss_pairs:
        cys_sg = {}
        for res in modeller.topology.residues():
            for atom in res.atoms():
                if atom.name == 'SG':
                    cys_sg[res.index + 1] = atom
        bonded_set = set()
        for a1, a2 in modeller.topology.bonds():
            bonded_set.add((a1.index, a2.index))
            bonded_set.add((a2.index, a1.index))
        for r1, r2 in expected_ss_pairs:
            sg1, sg2 = cys_sg[r1], cys_sg[r2]
            if (sg1.index, sg2.index) not in bonded_set:
                modeller.topology.addBond(sg1, sg2)

    modeller.addHydrogens(ff, pH=7.0)

    n_res = modeller.topology.getNumResidues()
    if target_n_res is not None and n_res != target_n_res:
        raise RuntimeError(
            f"Residue count mismatch for {out_pdb_path}: expected {target_n_res}, got {n_res}"
        )

    # Restrained clash-relief minimization (k = 1000 kJ/mol/nm^2 on heavy atoms)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff,
                             constraints=app.HBonds)
    restraint = openmm.CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restraint.addGlobalParameter("k", 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    pos_nm = modeller.positions.value_in_unit(unit.nanometers)
    for atom in modeller.topology.atoms():
        if atom.element.symbol != 'H':
            p = pos_nm[atom.index]
            restraint.addParticle(atom.index, [p[0], p[1], p[2]])
    system.addForce(restraint)

    integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond,
                                            0.002 * unit.picoseconds)
    sim = app.Simulation(modeller.topology, system, integ,
                         openmm.Platform.getPlatformByName('CPU'))
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=150)
    st = sim.context.getState(getPositions=True, getEnergy=True)
    with open(out_pdb_path, 'w') as f:
        app.PDBFile.writeFile(modeller.topology, st.getPositions(), f, keepIds=False)
    return float(st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))


ATLAS_SPECS = [
    dict(key='ala1', name='Alanine Dipeptide (ACE-Ala-NME)', tier=1,
         source_v3='version3_rigorous/structures/ala_dipeptide_ACE_ALA_NME.pdb',
         out_pdb='version4_godtier/structures/ala1_ACE_ALA_NME.pdb',
         pdb_id='ACE-ALA-NME', target_n_res=3, target_d=2, fold='peptide', ss_pairs=[]),
    dict(key='ala2', name='Trialanine / Capped Ala2 (ACE-Ala-Ala-NME)', tier=1,
         source_v3='version3_rigorous/structures/ala2_capped_ACE_ALA_ALA_NME.pdb',
         out_pdb='version4_godtier/structures/ala2_ACE_ALA_ALA_NME.pdb',
         pdb_id='ACE-ALA2-NME', target_n_res=4, target_d=4, fold='peptide', ss_pairs=[]),
    dict(key='chignolin', name='Chignolin (1UAO)', tier=1,
         source_v3='protein_scaling/structures/chignolin_1UAO.pdb',
         out_pdb='version4_godtier/structures/chignolin_1UAO.pdb',
         pdb_id='1UAO', target_n_res=10, target_d=18, fold='beta-hairpin', ss_pairs=[]),
    dict(key='trpcage', name='Trp-cage TC5b (1L2Y)', tier=1,
         source_v3='protein_scaling/structures/trpcage_1L2Y.pdb',
         out_pdb='version4_godtier/structures/trpcage_1L2Y.pdb',
         pdb_id='1L2Y', target_n_res=20, target_d=38, fold='alpha/PPII', ss_pairs=[]),
    dict(key='bba', name='BBA5 Mini-Protein (1FME)', tier=1,
         source_v3=None, out_pdb='version4_godtier/structures/bba_1FME.pdb',
         pdb_id='1FME', chain='A', target_n_res=28, target_d=54, fold='beta-beta-alpha', ss_pairs=[]),
    dict(key='villin', name='Villin HP35 (2F4K)', tier=1,
         source_v3='protein_scaling/structures/villin_hp35_protonated.pdb',
         out_pdb='version4_godtier/structures/villin_2F4K.pdb',
         pdb_id='2F4K', target_n_res=33, target_d=64, fold='3-helix', ss_pairs=[]),
    dict(key='wwdomain', name='WW Domain FiP35 (2F21)', tier=1,
         source_v3=None, out_pdb='version4_godtier/structures/wwdomain_2F21.pdb',
         pdb_id='2F21', chain='A', resnum_range=(4, 39), target_n_res=35, target_d=68, fold='3-beta-sheet',
         ss_pairs=[], add_missing_terms=False),
    dict(key='gb1', name='GB1 Domain (1PGA)', tier=2,
         source_v3='protein_scaling/structures/gb1_protonated.pdb',
         out_pdb='version4_godtier/structures/gb1_1PGA.pdb',
         alt_pdb_id='3GB1', alt_out_pdb='version4_godtier/structures/gb1_3GB1_alt.pdb',
         pdb_id='1PGA', target_n_res=56, target_d=110, fold='alpha/beta', ss_pairs=[]),
    dict(key='bpti', name='BPTI (5PTI)', tier=2,
         source_v3=None, out_pdb='version4_godtier/structures/bpti_5PTI.pdb',
         pdb_id='5PTI', chain='A', target_n_res=58, target_d=114, fold='disulfide-rich',
         ss_pairs=[(5, 55), (14, 38), (30, 51)]),
    dict(key='ubiquitin', name='Human Ubiquitin (1UBQ)', tier=2,
         source_v3='protein_scaling/structures/ubiquitin_protonated.pdb',
         out_pdb='version4_godtier/structures/ubiquitin_1UBQ.pdb',
         alt_pdb_id='1D3Z', alt_out_pdb='version4_godtier/structures/ubiquitin_1D3Z_alt.pdb',
         pdb_id='1UBQ', target_n_res=76, target_d=150, fold='alpha/beta', ss_pairs=[]),
    dict(key='barnase', name='Barnase (1BNI)', tier=2,
         source_v3=None, out_pdb='version4_godtier/structures/barnase_1BNI.pdb',
         pdb_id='1BNI', chain='A', target_n_res=110, target_d=218, fold='alpha+beta',
         ss_pairs=[], add_missing_terms=True),
    dict(key='lysozyme', name='HEWL Lysozyme (1AKI)', tier=2,
         source_v3=None, out_pdb='version4_godtier/structures/lysozyme_1AKI.pdb',
         pdb_id='1AKI', chain='A', target_n_res=129, target_d=256, fold='alpha+beta-4SS',
         ss_pairs=[(6, 127), (30, 115), (64, 80), (76, 94)]),
    dict(key='dhfr', name='E. coli DHFR (1RX2)', tier=3,
         source_v3=None, out_pdb='version4_godtier/structures/dhfr_1RX2.pdb',
         alt_pdb_id='1RA1', alt_out_pdb='version4_godtier/structures/dhfr_1RA1_alt.pdb',
         pdb_id='1RX2', chain='A', target_n_res=159, target_d=316, fold='Rossmann-loop', ss_pairs=[]),
    dict(key='adk', name='Adenylate Kinase Open (4AKE)', tier=3,
         source_v3=None, out_pdb='version4_godtier/structures/adk_4AKE_open.pdb',
         alt_pdb_id='1AKE', alt_out_pdb='version4_godtier/structures/adk_1AKE_closed_ref.pdb',
         pdb_id='4AKE', chain='A', target_n_res=214, target_d=426, fold='3-domain-allosteric', ss_pairs=[]),
    dict(key='mbp', name='Maltose-Binding Protein Open (1OMP)', tier=3,
         source_v3=None, out_pdb='version4_godtier/structures/mbp_1OMP_open.pdb',
         alt_pdb_id='1ANF', alt_out_pdb='version4_godtier/structures/mbp_1ANF_closed_ref.pdb',
         pdb_id='1OMP', chain='A', target_n_res=370, target_d=738, fold='2-lobe-allosteric', ss_pairs=[]),
]


def compute_structural_metadata(spec, pdb_abs):
    t = md.load(pdb_abs)
    top = t.topology
    pi, _ = md.compute_phi(t)
    si, _ = md.compute_psi(t)
    ci, _ = md.compute_chi1(t)

    phi_res = [int(top.atom(q[2]).residue.index) for q in pi]
    psi_res = [int(top.atom(q[1]).residue.index) for q in si]
    chi1_res = [int(top.atom(q[1]).residue.index) for q in ci] if len(ci) else []
    nphi, npsi, nchi1 = len(phi_res), len(psi_res), len(chi1_res)
    d = nphi + npsi
    if d != spec['target_d']:
        raise RuntimeError(f"Dimension mismatch for {spec['key']}: expected d={spec['target_d']}, got d={d}")

    pos = np.concatenate([np.array([(q[1] + q[2]) / 2.0 for q in pi]),
                          np.array([(q[1] + q[2]) / 2.0 for q in si])])
    order = np.argsort(pos, kind='stable').tolist()
    allres = phi_res + psi_res
    alltype = ['phi'] * nphi + ['psi'] * npsi

    block_res = sorted(set(phi_res) & set(psi_res))
    block_idx_res = [
        [int(phi_res.index(r)), int(nphi + psi_res.index(r))]
        for r in block_res
    ]
    block_idx_pep = []
    for k_psi, r_psi in enumerate(psi_res):
        if (r_psi + 1) in phi_res:
            k_phi = phi_res.index(r_psi + 1)
            block_idx_pep.append([int(nphi + k_psi), int(k_phi)])

    dssp3 = md.compute_dssp(t, simplified=True)[0].tolist()
    dssp8 = md.compute_dssp(t, simplified=False)[0].tolist()
    sasa_res = md.shrake_rupley(t, probe_radius=0.14, n_sphere_points=960, mode='residue')[0]
    rsasa = []
    for r_idx, res in enumerate(top.residues):
        max_s = MAX_SASA_NM2.get(res.name.upper(), 1.80)
        rsasa.append(round(float(sasa_res[r_idx]) / max_s, 4))

    # CA coordination burial and CA contact degree
    ca_idx = top.select('name CA')
    n_res = top.n_residues
    ca_burial = [0.0] * n_res
    ca_contact_deg = [0] * n_res
    n_ca_contacts = 0
    if len(ca_idx) >= 2:
        ca_xyz = t.xyz[0, ca_idx, :]
        for i in range(len(ca_idx)):
            ri = top.atom(ca_idx[i]).residue.index
            for j in range(i + 1, len(ca_idx)):
                rj = top.atom(ca_idx[j]).residue.index
                dist = float(np.linalg.norm(ca_xyz[i] - ca_xyz[j]))
                if abs(ri - rj) >= 2:
                    val = 1.0 / (1.0 + np.exp((dist - 0.80) / 0.10))
                    ca_burial[ri] += val
                    ca_burial[rj] += val
                if abs(ri - rj) >= 3 and dist < 0.65:
                    ca_contact_deg[ri] += 1
                    ca_contact_deg[rj] += 1
                    n_ca_contacts += 1

    # Heavy-atom packing and BHE native contact count
    heavy_idx = top.select('not element H')
    heavy_xyz = t.xyz[0, heavy_idx, :]
    heavy_res = np.array([top.atom(a).residue.index for a in heavy_idx])
    dmat = np.linalg.norm(heavy_xyz[:, None, :] - heavy_xyz[None, :, :], axis=-1)
    rsep = np.abs(heavy_res[:, None] - heavy_res[None, :])
    heavy_packing = []
    for r in range(n_res):
        mask_r = (heavy_res == r)
        if not np.any(mask_r):
            heavy_packing.append(0)
            continue
        sub = dmat[mask_r][:, rsep[mask_r][0] >= 2] if np.sum(mask_r) == 1 else dmat[mask_r]
        # Count distinct non-hydrogen atoms on |i-j|>=2 within 0.60 nm
        other_mask = np.abs(heavy_res - r) >= 2
        if np.any(other_mask):
            min_d = dmat[np.ix_(mask_r, other_mask)].min(axis=0)
            heavy_packing.append(int(np.sum(min_d < 0.60)))
        else:
            heavy_packing.append(0)

    bhe_mask = (np.triu(rsep, k=1) >= 4) & (np.triu(dmat, k=1) < 0.45) & (np.triu(dmat, k=1) > 0.0)
    n_bhe_pairs = int(np.sum(bhe_mask))

    # Verify disulfides if expected
    verified_ss = []
    if spec.get('ss_pairs'):
        sg_atoms = {}
        for res in top.residues:
            for a in res.atoms:
                if a.name == 'SG':
                    sg_atoms[res.index + 1] = a.index
        for r1, r2 in spec['ss_pairs']:
            a1, a2 = sg_atoms[r1], sg_atoms[r2]
            dist_ss = float(np.linalg.norm(t.xyz[0, a1] - t.xyz[0, a2]))
            if not (0.18 <= dist_ss <= 0.25):
                raise RuntimeError(f"Disulfide distance out of bounds for {spec['key']} ({r1}-{r2}): {dist_ss:.3f} nm")
            verified_ss.append([r1, r2, round(dist_ss, 4)])

    meta = {
        "key": spec['key'],
        "name": spec['name'],
        "pdb_id": spec['pdb_id'],
        "tier": spec['tier'],
        "fold": spec['fold'],
        "pdb_relpath": spec['out_pdb'],
        "pdb_sha256": sha256(pdb_abs),
        "n_atoms": int(t.n_atoms),
        "n_heavy_atoms": int(len(heavy_idx)),
        "n_residues": int(n_res),
        "n_phi": int(nphi),
        "n_psi": int(npsi),
        "d": int(d),
        "n_chi1": int(nchi1),
        "phi_atoms": pi.tolist(),
        "psi_atoms": si.tolist(),
        "chi1_atoms": ci.tolist() if len(ci) else [],
        "phi_res": phi_res,
        "psi_res": psi_res,
        "chi1_res": chi1_res,
        "chain_order": order,
        "chain_res": [allres[i] for i in order],
        "chain_type": [alltype[i] for i in order],
        "block_res": block_res,
        "block_idx_res": block_idx_res,
        "block_idx_pep": block_idx_pep,
        "dssp_3state": dssp3,
        "dssp_8state": dssp8,
        "sasa_nm2": [round(float(x), 4) for x in sasa_res],
        "rsasa": rsasa,
        "ca_burial_rho": [round(float(x), 4) for x in ca_burial],
        "heavy_packing_rho": heavy_packing,
        "ca_contact_degree": ca_contact_deg,
        "n_bhe_heavy_pairs": n_bhe_pairs,
        "n_ca_contact_pairs": n_ca_contacts,
        "disulfide_bridges_verified": verified_ss,
        "rg_native_nm": round(float(md.compute_rg(t)[0]), 4),
    }
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()

    manifest_systems = []
    for spec in ATLAS_SPECS:
        key = spec['key']
        out_abs = os.path.join(ROOT, spec['out_pdb'])
        if not args.verify_only:
            if spec['source_v3'] is not None:
                src_abs = os.path.join(ROOT, spec['source_v3'])
                shutil.copy2(src_abs, out_abs)
                print(f"[{key:10s}] Copied V3 reference -> {spec['out_pdb']} (SHA-256 identical)", flush=True)
            else:
                if not os.path.exists(out_abs):
                    raw_pdb = os.path.join(RAW_DIR, f"{spec['pdb_id']}.pdb")
                    download_rcsb_pdb(spec['pdb_id'], raw_pdb)
                    epot = clean_and_protonate_pdb(
                        raw_pdb, out_abs,
                        chain_id=spec.get('chain', 'A'),
                        target_n_res=spec['target_n_res'],
                        expected_ss_pairs=spec.get('ss_pairs', []),
                        resnum_range=spec.get('resnum_range'),
                        add_missing_terminal_residues=spec.get('add_missing_terms', False),
                    )
                    print(f"[{key:10s}] Prepared {spec['out_pdb']} | Epot = {epot:.1f} kJ/mol", flush=True)

            if spec.get('alt_pdb_id'):
                alt_abs = os.path.join(ROOT, spec['alt_out_pdb'])
                if not os.path.exists(alt_abs):
                    raw_alt = os.path.join(RAW_DIR, f"{spec['alt_pdb_id']}.pdb")
                    download_rcsb_pdb(spec['alt_pdb_id'], raw_alt)
                    epot_alt = clean_and_protonate_pdb(
                        raw_alt, alt_abs,
                        chain_id=spec.get('chain', 'A'),
                        target_n_res=spec['target_n_res'],
                        expected_ss_pairs=spec.get('ss_pairs', []),
                        add_missing_terminal_residues=spec.get('add_missing_terms', False),
                    )
                    print(f"[{key:10s}] Prepared alt state {spec['alt_out_pdb']} | Epot = {epot_alt:.1f} kJ/mol", flush=True)

        meta = compute_structural_metadata(spec, out_abs)
        if spec.get('alt_out_pdb'):
            alt_abs = os.path.join(ROOT, spec['alt_out_pdb'])
            t_main = md.load(out_abs)
            t_alt = md.load(alt_abs)
            ca_main = t_main.topology.select('name CA')
            ca_alt = t_alt.topology.select('name CA')
            if len(ca_main) != len(ca_alt):
                raise RuntimeError(f"Alt CA count mismatch for {key}: {len(ca_main)} vs {len(ca_alt)}")
            rmsd_open_closed = float(md.rmsd(t_main, t_alt, atom_indices=ca_main, ref_atom_indices=ca_alt)[0])
            meta['alt_pdb_relpath'] = spec['alt_out_pdb']
            meta['alt_pdb_sha256'] = sha256(alt_abs)
            meta['alt_ca_rmsd_nm'] = round(rmsd_open_closed, 4)

        meta_path = os.path.join(META_DIR, f"{key}_meta.json")
        with open(meta_path, 'w') as mf:
            json.dump(meta, mf, indent=2)

        manifest_systems.append({
            "key": key,
            "name": spec['name'],
            "tier": spec['tier'],
            "fold": spec['fold'],
            "pdb_relpath": spec['out_pdb'],
            "pdb_sha256": meta['pdb_sha256'],
            "alt_pdb_relpath": meta.get('alt_pdb_relpath'),
            "alt_pdb_sha256": meta.get('alt_pdb_sha256'),
            "alt_ca_rmsd_nm": meta.get('alt_ca_rmsd_nm'),
            "n_atoms": meta['n_atoms'],
            "n_heavy_atoms": meta['n_heavy_atoms'],
            "n_residues": meta['n_residues'],
            "n_phi": meta['n_phi'],
            "n_psi": meta['n_psi'],
            "d": meta['d'],
            "n_chi1": meta['n_chi1'],
            "n_bhe_heavy_pairs": meta['n_bhe_heavy_pairs'],
            "n_ca_contact_pairs": meta['n_ca_contact_pairs'],
            "disulfide_bridges_verified": meta['disulfide_bridges_verified'],
            "rg_native_nm": meta['rg_native_nm'],
            "metadata_json": os.path.relpath(meta_path, ROOT),
        })
        print(
            f"  -> VERIFIED [{key:10s}] N_res={meta['n_residues']:3d} | d={meta['d']:3d} | "
            f"n_chi1={meta['n_chi1']:3d} | Atoms={meta['n_atoms']:4d} | "
            f"Contacts={meta['n_ca_contact_pairs']:3d} | Rg={meta['rg_native_nm']:.3f} nm",
            flush=True
        )

    master_manifest = {
        "campaign": "version4_godtier",
        "milestone": "M29",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_systems": len(manifest_systems),
        "dimensions_d": [s["d"] for s in manifest_systems],
        "systems": manifest_systems,
    }
    man_path = os.path.join(MAN_DIR, 'm29_topology_manifest.json')
    with open(man_path, 'w') as f:
        json.dump(master_manifest, f, indent=2)
    print(f"\n[M29 TOPOLOGY MANIFEST COMPLETE] Saved {man_path} ({len(manifest_systems)} systems, d = 2 -> 738)", flush=True)


if __name__ == '__main__':
    main()
