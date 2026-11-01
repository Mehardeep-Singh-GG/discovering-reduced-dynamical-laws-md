#!/usr/bin/env python3
"""
version4_godtier/sims/md_engine_v4_gpu.py
=========================================
High-throughput V4-GodTier OpenMM GPU / Multi-Platform Molecular Dynamics Engine.

Supports:
  1. Explicit-solvent PME ('explicit_pme'): AMBER ff99SB-ILDN + TIP3P-FB water
     in a 1.2 nm rhombic dodecahedron with 0.15 M NaCl, 1.0 nm PME cutoff,
     HBonds constraints, 1.5 amu Hydrogen Mass Repartitioning (HMR), 4.0 fs step,
     and 4-stage Minimization -> NVT -> NPT -> Barostat-Removed NVT equilibration.
  2. Implicit-solvent Neck-GB ('implicit_gbn2') and OBC2 ('implicit_obc2'):
     AMBER ff99SB-ILDN + GBn2/OBC2, NoCutoff, HBonds constraints, HMR 1.5 amu (4 fs)
     or standard 1.008 amu (2 fs).
  3. On-the-fly solute extraction every 1.0 ps: saves float32 heavy-atom and CA
     coordinates, exact backbone (phi, psi) and side-chain (chi1) angles,
     Best-Hummer-Eaton Q_BHE(t), CA RMSD(t), Rg(t), and allosteric domain metrics.
  4. Seamless 10-ns chunked XML state serialization (sim.saveState / sim.loadState)
     and per-chunk SHA-256 manifest generation.
"""
import os
import sys
import json
import time
import hashlib
import argparse
import numpy as np
import openmm
from openmm import app, unit
import mdtraj as md

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def select_best_platform(requested_platform=None, threads=None):
    available = [openmm.Platform.getPlatform(i).getName()
                 for i in range(openmm.Platform.getNumPlatforms())]
    if requested_platform and requested_platform in available:
        plat_name = requested_platform
    elif 'CUDA' in available:
        plat_name = 'CUDA'
    elif 'OpenCL' in available:
        plat_name = 'OpenCL'
    else:
        plat_name = 'CPU'
    platform = openmm.Platform.getPlatformByName(plat_name)
    props = {}
    if plat_name == 'CUDA':
        props['Precision'] = 'mixed'
    elif plat_name == 'OpenCL':
        props['Precision'] = 'single'
    elif plat_name == 'CPU' and threads:
        props['Threads'] = str(threads)
    return platform, plat_name, props


def native_contact_pairs_from_traj(ref_traj, cutoff_nm=0.45, min_seq_sep=4):
    heavy = ref_traj.topology.select('not element H')
    pairs = []
    for ii, a in enumerate(heavy):
        ra = ref_traj.topology.atom(a).residue.index
        for b in heavy[ii + 1:]:
            if abs(ref_traj.topology.atom(b).residue.index - ra) >= min_seq_sep:
                pairs.append((a, b))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    pairs = np.array(pairs, dtype=np.int32)
    r0 = md.compute_distances(ref_traj, pairs)[0]
    sel = r0 < cutoff_nm
    return pairs[sel], r0[sel].astype(np.float32)


def compute_allosteric_metrics(system_key, traj_solute):
    """Compute domain-specific hinge / loop coordinates for Tier 3 enzymes."""
    out = {}
    top = traj_solute.topology
    pos = traj_solute.xyz  # (T, N_solute, 3) in nm
    if system_key == 'adk':
        core_idx = [a.index for a in top.atoms if a.name == 'CA' and (
            (0 <= a.residue.index <= 28) or (59 <= a.residue.index <= 120) or (159 <= a.residue.index <= 213)
        )]
        lid_idx = [a.index for a in top.atoms if a.name == 'CA' and (121 <= a.residue.index <= 158)]
        nmp_idx = [a.index for a in top.atoms if a.name == 'CA' and (29 <= a.residue.index <= 58)]
        if core_idx and lid_idx and nmp_idx:
            com_core = pos[:, core_idx, :].mean(axis=1)
            com_lid = pos[:, lid_idx, :].mean(axis=1)
            com_nmp = pos[:, nmp_idx, :].mean(axis=1)
            out['adk_d_lid_core_nm'] = np.linalg.norm(com_lid - com_core, axis=1).astype(np.float32)
            out['adk_d_nmp_core_nm'] = np.linalg.norm(com_nmp - com_core, axis=1).astype(np.float32)
    elif system_key == 'mbp':
        ntd_idx = [a.index for a in top.atoms if a.name == 'CA' and (
            (0 <= a.residue.index <= 108) or (263 <= a.residue.index <= 308)
        )]
        ctd_idx = [a.index for a in top.atoms if a.name == 'CA' and (
            (113 <= a.residue.index <= 257) or (315 <= a.residue.index <= 369)
        )]
        if ntd_idx and ctd_idx:
            com_ntd = pos[:, ntd_idx, :].mean(axis=1)
            com_ctd = pos[:, ctd_idx, :].mean(axis=1)
            out['mbp_d_ntd_ctd_nm'] = np.linalg.norm(com_ntd - com_ctd, axis=1).astype(np.float32)
    elif system_key == 'dhfr':
        met20_idx = [a.index for a in top.atoms if a.name == 'CA' and (8 <= a.residue.index <= 23)]
        core_idx = [a.index for a in top.atoms if a.name == 'CA' and not (8 <= a.residue.index <= 23)]
        if met20_idx and core_idx:
            com_met20 = pos[:, met20_idx, :].mean(axis=1)
            com_core = pos[:, core_idx, :].mean(axis=1)
            out['dhfr_d_met20_core_nm'] = np.linalg.norm(com_met20 - com_core, axis=1).astype(np.float32)
    return out


def run_chunk(pdb_path, out_npz, out_xml, out_manifest, system_key, solvent='explicit_pme',
              temperature_k=300.0, friction_ps=1.0, time_step_fs=4.0, hmr_amu=1.5,
              stride_ps=1.0, ns=10.0, chunk_index=0, replica_index=1, seed=42001,
              platform_name='CUDA', threads=None, resume_xml=None, alt_pdb_path=None,
              skip_equil=False, cache_dir=None):
    solute_pdb = app.PDBFile(pdb_path)
    n_solute_atoms = solute_pdb.topology.getNumAtoms()
    solute_md_top = md.Topology.from_openmm(solute_pdb.topology)
    ref_solute_traj = md.load(pdb_path)
    heavy_idx = solute_md_top.select('not element H')
    ca_idx = solute_md_top.select('name CA')

    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_xml)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_manifest)), exist_ok=True)

    _cdir = os.path.abspath(cache_dir) if cache_dir else os.path.dirname(os.path.abspath(out_xml))
    os.makedirs(_cdir, exist_ok=True)
    sys_xml_cache = os.path.join(_cdir, f"{system_key}_{solvent}_rep{replica_index}_system.xml")
    solv_pdb_cache = os.path.join(_cdir, f"{system_key}_{solvent}_rep{replica_index}_solvated.pdb")

    hmr_mass = (hmr_amu * unit.amu) if (hmr_amu and hmr_amu > 1.01) else None

    if solvent == 'explicit_pme':
        ff = app.ForceField('amber99sbildn.xml', 'tip3pfb.xml')
        if chunk_index == 0 or not (os.path.exists(sys_xml_cache) and os.path.exists(solv_pdb_cache)):
            # 1.2 nm padding, enlarged only if needed so that every box height
            # exceeds 2 x 1.0 nm cutoff (+0.2 nm margin) under NPT fluctuations.
            pad_nm = 1.2
            while True:
                modeller = app.Modeller(solute_pdb.topology, solute_pdb.positions)
                modeller.addSolvent(
                    ff, model='tip3p', padding=pad_nm * unit.nanometers,
                    boxShape='dodecahedron', ionicStrength=0.15 * unit.molar,
                    positiveIon='Na+', negativeIon='Cl-'
                )
                bv = np.array(modeller.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometers))
                vol = abs(np.linalg.det(bv))
                heights = [vol / np.linalg.norm(np.cross(bv[(i + 1) % 3], bv[(i + 2) % 3])) for i in range(3)]
                if min(heights) >= 2.2 or pad_nm > 3.0:
                    break
                pad_nm += 0.2
            with open(solv_pdb_cache, 'w') as spf:
                app.PDBFile.writeFile(modeller.topology, modeller.positions, spf)
            system = ff.createSystem(
                modeller.topology, nonbondedMethod=app.PME,
                nonbondedCutoff=1.0 * unit.nanometers, ewaldErrorTolerance=1.0e-4,
                constraints=app.HBonds, rigidWater=True, hydrogenMass=hmr_mass
            )
            with open(sys_xml_cache, 'w') as sxf:
                sxf.write(openmm.XmlSerializer.serialize(system))
            sim_top = modeller.topology
            init_pos = modeller.positions
        else:
            solv_pdb = app.PDBFile(solv_pdb_cache)
            sim_top = solv_pdb.topology
            init_pos = solv_pdb.positions
            with open(sys_xml_cache, 'r') as sxf:
                system = openmm.XmlSerializer.deserialize(sxf.read())
        ff_desc = f"amber99sbildn.xml + tip3pfb.xml (PME 1.0nm, dodecahedron 1.2nm, 0.15M NaCl, HMR={hmr_amu}amu)"
    elif solvent == 'implicit_gbn2':
        ff = app.ForceField('amber99sbildn.xml', 'implicit/gbn2.xml')
        system = ff.createSystem(solute_pdb.topology, nonbondedMethod=app.NoCutoff,
                                 constraints=app.HBonds, hydrogenMass=hmr_mass)
        sim_top = solute_pdb.topology
        init_pos = solute_pdb.positions
        ff_desc = f"amber99sbildn.xml + implicit/gbn2.xml (GBn2, NoCutoff, HMR={hmr_amu}amu)"
    elif solvent == 'implicit_obc2':
        ff = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
        system = ff.createSystem(solute_pdb.topology, nonbondedMethod=app.NoCutoff,
                                 constraints=app.HBonds, hydrogenMass=hmr_mass)
        sim_top = solute_pdb.topology
        init_pos = solute_pdb.positions
        ff_desc = f"amber99sbildn.xml + amber99_obc.xml (OBC2, NoCutoff, HMR={hmr_amu}amu)"
    else:
        raise ValueError(f"Unsupported solvent mode: {solvent}")

    platform, plat_name, props = select_best_platform(platform_name, threads)
    integ = openmm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin,
        friction_ps / unit.picosecond,
        time_step_fs * unit.femtoseconds
    )
    integ.setRandomNumberSeed(int(seed) + int(chunk_index))
    sim = app.Simulation(sim_top, system, integ, platform, props)

    if chunk_index == 0 and resume_xml is None:
        sim.context.setPositions(init_pos)
        sim.minimizeEnergy(maxIterations=500 if not skip_equil else 50)
        sim.context.setVelocitiesToTemperature(temperature_k * unit.kelvin, int(seed))
        if not skip_equil:
            if solvent == 'explicit_pme':
                sim.step(int(round(50000.0 / time_step_fs)))
                baro_idx = system.addForce(
                    openmm.MonteCarloBarostat(1.0 * unit.bar, temperature_k * unit.kelvin, 25)
                )
                sim.context.reinitialize(preserveState=True)
                sim.step(int(round(150000.0 / time_step_fs)))
                system.removeForce(baro_idx)
                sim.context.reinitialize(preserveState=True)
                with open(sys_xml_cache, 'w') as sxf:
                    sxf.write(openmm.XmlSerializer.serialize(system))
            else:
                sim.step(int(round(10000.0 / time_step_fs)))
        sim.context.setTime(0.0 * unit.picoseconds)
    else:
        if not resume_xml or not os.path.exists(resume_xml):
            raise RuntimeError(f"chunk_index={chunk_index} requires existing resume_xml={resume_xml}")
        sim.loadState(resume_xml)

    stride_steps = max(1, int(round(stride_ps * 1000.0 / time_step_fs)))
    total_steps = max(stride_steps, int(round(ns * 1e6 / time_step_fs)))
    n_frames = total_steps // stride_steps

    pos_solute_all = np.empty((n_frames, n_solute_atoms, 3), dtype=np.float32)
    times_ps = np.empty(n_frames, dtype=np.float32)

    t0 = time.time()
    start_time_ps = float(sim.context.getState().getTime().value_in_unit(unit.picoseconds))

    for k in range(n_frames):
        sim.step(stride_steps)
        st = sim.context.getState(getPositions=True, enforcePeriodicBox=(solvent == 'explicit_pme'))
        coords_nm = st.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        pos_solute_all[k] = coords_nm[:n_solute_atoms]
        times_ps[k] = st.getTime().value_in_unit(unit.picoseconds)
        if (k + 1) % 1000 == 0 or k + 1 == n_frames:
            el = time.time() - t0
            rate = (k + 1) * stride_steps * time_step_fs / 1e6 / max(el / 86400.0, 1e-12)
            print(f"[{os.path.basename(out_npz)}] frame {k+1}/{n_frames} | {el:.1f}s | {rate:.1f} ns/day", flush=True)

    wall = time.time() - t0
    sim.saveState(out_xml)

    traj_solute = md.Trajectory(pos_solute_all, solute_md_top)
    phi_idx, phi = md.compute_phi(traj_solute)
    psi_idx, psi = md.compute_psi(traj_solute)
    chi1_idx, chi1 = md.compute_chi1(traj_solute)

    bhe_pairs, bhe_r0 = native_contact_pairs_from_traj(ref_solute_traj)
    if len(bhe_pairs) > 0:
        r_dist = md.compute_distances(traj_solute, bhe_pairs)
        q_bhe = np.mean(1.0 / (1.0 + np.exp(50.0 * (r_dist - 1.8 * bhe_r0))), axis=1).astype(np.float32)
    else:
        q_bhe = np.ones(n_frames, dtype=np.float32)

    rmsd_ca_nm = md.rmsd(traj_solute, ref_solute_traj, atom_indices=ca_idx).astype(np.float32)
    rg_nm = md.compute_rg(traj_solute).astype(np.float32)
    allo_metrics = compute_allosteric_metrics(system_key, traj_solute)

    save_dict = dict(
        positions_heavy=pos_solute_all[:, heavy_idx, :],
        positions_ca=pos_solute_all[:, ca_idx, :],
        heavy_atom_indices=heavy_idx.astype(np.int32),
        ca_atom_indices=ca_idx.astype(np.int32),
        times=times_ps,
        phi=phi.astype(np.float32),
        psi=psi.astype(np.float32),
        chi1=chi1.astype(np.float32) if len(chi1_idx) else np.zeros((n_frames, 0), dtype=np.float32),
        phi_atoms=phi_idx.astype(np.int32),
        psi_atoms=psi_idx.astype(np.int32),
        chi1_atoms=chi1_idx.astype(np.int32) if len(chi1_idx) else np.zeros((0, 4), dtype=np.int32),
        q_bhe=q_bhe,
        rmsd_ca_nm=rmsd_ca_nm,
        rg_nm=rg_nm,
        temperature_k=np.float32(temperature_k),
        friction_ps=np.float32(friction_ps),
        time_step_fs=np.float32(time_step_fs),
        hmr_amu=np.float32(hmr_amu if hmr_amu else 1.008),
        seed=np.int32(seed),
        chunk_index=np.int32(chunk_index),
        replica_index=np.int32(replica_index),
    )
    if n_solute_atoms <= 1800:
        save_dict['positions'] = pos_solute_all
    if alt_pdb_path and os.path.exists(alt_pdb_path):
        alt_ref = md.load(alt_pdb_path)
        if len(alt_ref.topology.select('name CA')) == len(ca_idx):
            save_dict['rmsd_alt_ca_nm'] = md.rmsd(traj_solute, alt_ref, atom_indices=ca_idx).astype(np.float32)
    save_dict.update(allo_metrics)

    np.savez_compressed(out_npz, **save_dict)

    role_map = {1: 'discovery', 2: 'validation', 3: 'test'}
    rec = {
        'campaign': 'version4_godtier',
        'milestone': 'M30',
        'system_key': system_key,
        'replica_index': int(replica_index),
        'replica_role': role_map.get(int(replica_index), f'rep{replica_index}'),
        'chunk_index': int(chunk_index),
        'solvent': solvent,
        'trajectory_file': os.path.relpath(os.path.abspath(out_npz), ROOT),
        'sha256': sha256(out_npz),
        'state_xml_file': os.path.relpath(os.path.abspath(out_xml), ROOT),
        'state_xml_sha256': sha256(out_xml),
        'input_pdb': os.path.relpath(os.path.abspath(pdb_path), ROOT),
        'input_pdb_sha256': sha256(pdb_path),
        'resume_xml_sha256': sha256(resume_xml) if (resume_xml and os.path.exists(resume_xml)) else None,
        'force_field': ff_desc,
        'integrator': 'LangevinMiddleIntegrator (BAOAB)',
        'temperature_k': float(temperature_k),
        'friction_ps': float(friction_ps),
        'time_step_fs': float(time_step_fs),
        'hmr_amu': float(hmr_amu if hmr_amu else 1.008),
        'frame_stride_ps': float(stride_ps),
        'seed': int(seed),
        'start_time_ps': round(start_time_ps, 3),
        'end_time_ps': round(float(times_ps[-1]), 3),
        'total_frames': int(n_frames),
        'total_time_ns': round(n_frames * stride_ps / 1000.0, 6),
        'n_total_particles': int(system.getNumParticles()),
        'n_solute_atoms': int(n_solute_atoms),
        'n_solute_heavy_atoms': int(len(heavy_idx)),
        'n_phi': int(phi.shape[1]),
        'n_psi': int(psi.shape[1]),
        'd': int(phi.shape[1] + psi.shape[1]),
        'n_chi1': int(chi1.shape[1]) if len(chi1_idx) else 0,
        'mean_q_bhe': round(float(np.mean(q_bhe)), 4),
        'mean_rmsd_ca_nm': round(float(np.mean(rmsd_ca_nm)), 4),
        'mean_rg_nm': round(float(np.mean(rg_nm)), 4),
        'wall_time_seconds': round(wall, 3),
        'ns_per_day': round((n_frames * stride_ps / 1000.0) / max(wall / 86400.0, 1e-12), 2),
        'platform': plat_name,
        'openmm_version': openmm.__version__,
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    with open(out_manifest, 'w') as mf:
        json.dump(rec, mf, indent=2)
    return rec


def run_batch_3replicas(pdb_path, out_dir_traj, out_dir_man, system_key,
                        temperature_k=300.0, friction_ps=1.0, time_step_fs=4.0, hmr_amu=1.5,
                        stride_ps=1.0, ns=6.0, chunk_index=0, seeds=(42001, 42002, 42003),
                        platform_name='CUDA', threads=None, alt_pdb_path=None, skip_equil=False):
    """
    Single-Context 3-Replica GPU Batch Engine for implicit_gbn2:
    Places 3 independent copies of the protein centered at (-L, 0, 0), (0, 0, 0), (+L, 0, 0)
    inside ONE OpenMM Simulation Context with CutoffNonPeriodic (cutoff > protein diameter,
    L > 3 * cutoff) and a translational anchor on the first CA atom of each replica.
    Integrates all 3 replicas in a single GPU kernel stream with independent Maxwell-Boltzmann
    velocities (seeds[0], seeds[1], seeds[2]), then extracts and writes the 3 independent
    .npz trajectories and .json manifests.
    """
    solute_pdb = app.PDBFile(pdb_path)
    n_solute_atoms = solute_pdb.topology.getNumAtoms()
    solute_md_top = md.Topology.from_openmm(solute_pdb.topology)
    ref_solute_traj = md.load(pdb_path)
    heavy_idx = solute_md_top.select('not element H')
    ca_idx = solute_md_top.select('name CA')

    os.makedirs(out_dir_traj, exist_ok=True)
    os.makedirs(out_dir_man, exist_ok=True)

    pos_raw = np.array(solute_pdb.positions.value_in_unit(unit.nanometers), dtype=np.float64)
    pos_centered = pos_raw - pos_raw.mean(axis=0, keepdims=True)
    max_span = float(np.max(np.linalg.norm(pos_centered - pos_centered[:, None, :], axis=-1)))
    cutoff_nm = max(5.5, round(max_span + 1.5, 1))
    sep_nm = max(16.0, float(2 ** int(np.ceil(np.log2(3.0 * cutoff_nm)))))

    offsets = [
        np.array([-sep_nm, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([sep_nm, 0.0, 0.0]),
    ]

    mod = app.Modeller(solute_pdb.topology, (pos_centered + offsets[0]) * unit.nanometers)
    mod.add(solute_pdb.topology, (pos_centered + offsets[1]) * unit.nanometers)
    mod.add(solute_pdb.topology, (pos_centered + offsets[2]) * unit.nanometers)

    hmr_mass = (hmr_amu * unit.amu) if (hmr_amu and hmr_amu > 1.01) else None
    ff = app.ForceField('amber99sbildn.xml', 'implicit/gbn2.xml')
    system = ff.createSystem(
        mod.topology, nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=cutoff_nm * unit.nanometers,
        constraints=app.HBonds, hydrogenMass=hmr_mass
    )

    # Translational anchor on first CA atom of each replica to prevent Brownian COM drift
    anchor_force = openmm.CustomExternalForce('0.5*k_anchor*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    anchor_force.addGlobalParameter('k_anchor', 10.0)
    anchor_force.addPerParticleParameter('x0')
    anchor_force.addPerParticleParameter('y0')
    anchor_force.addPerParticleParameter('z0')
    first_ca = int(ca_idx[0]) if len(ca_idx) > 0 else 0
    for r in range(3):
        p_idx = r * n_solute_atoms + first_ca
        xyz0 = pos_centered[first_ca] + offsets[r]
        anchor_force.addParticle(p_idx, [float(xyz0[0]), float(xyz0[1]), float(xyz0[2])])
    system.addForce(anchor_force)

    platform, plat_name, props = select_best_platform(platform_name, threads)
    integ = openmm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin,
        friction_ps / unit.picosecond,
        time_step_fs * unit.femtoseconds
    )
    integ.setRandomNumberSeed(int(seeds[0]))
    sim = app.Simulation(mod.topology, system, integ, platform, props)
    sim.context.setPositions(mod.positions)
    sim.minimizeEnergy(maxIterations=400 if not skip_equil else 50)

    # Assign independent Maxwell-Boltzmann velocities per replica using each replica's seed
    kB_T = (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature_k * unit.kelvin).value_in_unit(unit.kilojoules_per_mole)
    vel_all = np.zeros((3 * n_solute_atoms, 3), dtype=np.float64)
    for r in range(3):
        rng = np.random.RandomState(int(seeds[r]))
        for i in range(n_solute_atoms):
            p_idx = r * n_solute_atoms + i
            m_amu = system.getParticleMass(p_idx).value_in_unit(unit.amu)
            if m_amu > 0.0:
                sigma_v = np.sqrt(kB_T / m_amu)
                vel_all[p_idx] = rng.normal(0.0, sigma_v, size=3)
    sim.context.setVelocities(vel_all * (unit.nanometer / unit.picosecond))

    if not skip_equil:
        sim.step(int(round(10000.0 / time_step_fs)))
    sim.context.setTime(0.0 * unit.picoseconds)

    stride_steps = max(1, int(round(stride_ps * 1000.0 / time_step_fs)))
    total_steps = max(stride_steps, int(round(ns * 1e6 / time_step_fs)))
    n_frames = total_steps // stride_steps

    pos_batched = np.empty((n_frames, 3 * n_solute_atoms, 3), dtype=np.float32)
    times_ps = np.empty(n_frames, dtype=np.float32)

    t0 = time.time()
    for k in range(n_frames):
        sim.step(stride_steps)
        st = sim.context.getState(getPositions=True)
        pos_batched[k] = st.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        times_ps[k] = st.getTime().value_in_unit(unit.picoseconds)
        if (k + 1) % 1000 == 0 or k + 1 == n_frames:
            el = time.time() - t0
            eff_rate = 3.0 * (k + 1) * stride_steps * time_step_fs / 1e6 / max(el / 86400.0, 1e-12)
            print(f"[BATCH-3REP {system_key}] frame {k+1}/{n_frames} (x3 reps) | {el:.1f}s | {eff_rate:.1f} eff ns/day", flush=True)

    wall = time.time() - t0
    bhe_pairs, bhe_r0 = native_contact_pairs_from_traj(ref_solute_traj)
    role_map = {1: 'discovery', 2: 'validation', 3: 'test'}
    manifests = []

    for r in range(3):
        rep_idx = r + 1
        pos_rep = pos_batched[:, r * n_solute_atoms:(r + 1) * n_solute_atoms, :] - offsets[r].astype(np.float32)
        traj_solute = md.Trajectory(pos_rep, solute_md_top)
        phi_idx, phi = md.compute_phi(traj_solute)
        psi_idx, psi = md.compute_psi(traj_solute)
        chi1_idx, chi1 = md.compute_chi1(traj_solute)

        if len(bhe_pairs) > 0:
            r_dist = md.compute_distances(traj_solute, bhe_pairs)
            q_bhe = np.mean(1.0 / (1.0 + np.exp(50.0 * (r_dist - 1.8 * bhe_r0))), axis=1).astype(np.float32)
        else:
            q_bhe = np.ones(n_frames, dtype=np.float32)

        rmsd_ca_nm = md.rmsd(traj_solute, ref_solute_traj, atom_indices=ca_idx).astype(np.float32)
        rg_nm = md.compute_rg(traj_solute).astype(np.float32)
        allo_metrics = compute_allosteric_metrics(system_key, traj_solute)

        base_name = f"{system_key}_rep{rep_idx}_chunk{chunk_index:03d}"
        out_npz = os.path.join(out_dir_traj, f"{base_name}.npz")
        out_xml = os.path.join(out_dir_traj, f"{base_name}_state.xml")
        out_man = os.path.join(out_dir_man, f"{base_name}_implicit_gbn2.json")
        sim.saveState(out_xml)

        save_dict = dict(
            positions_heavy=pos_rep[:, heavy_idx, :],
            positions_ca=pos_rep[:, ca_idx, :],
            heavy_atom_indices=heavy_idx.astype(np.int32),
            ca_atom_indices=ca_idx.astype(np.int32),
            times=times_ps,
            phi=phi.astype(np.float32),
            psi=psi.astype(np.float32),
            chi1=chi1.astype(np.float32) if len(chi1_idx) else np.zeros((n_frames, 0), dtype=np.float32),
            phi_atoms=phi_idx.astype(np.int32),
            psi_atoms=psi_idx.astype(np.int32),
            chi1_atoms=chi1_idx.astype(np.int32) if len(chi1_idx) else np.zeros((0, 4), dtype=np.int32),
            q_bhe=q_bhe,
            rmsd_ca_nm=rmsd_ca_nm,
            rg_nm=rg_nm,
            temperature_k=np.float32(temperature_k),
            friction_ps=np.float32(friction_ps),
            time_step_fs=np.float32(time_step_fs),
            hmr_amu=np.float32(hmr_amu if hmr_amu else 1.008),
            seed=np.int32(seeds[r]),
            chunk_index=np.int32(chunk_index),
            replica_index=np.int32(rep_idx),
        )
        if n_solute_atoms <= 1800:
            save_dict['positions'] = pos_rep
        if alt_pdb_path and os.path.exists(alt_pdb_path):
            alt_ref = md.load(alt_pdb_path)
            if len(alt_ref.topology.select('name CA')) == len(ca_idx):
                save_dict['rmsd_alt_ca_nm'] = md.rmsd(traj_solute, alt_ref, atom_indices=ca_idx).astype(np.float32)
        save_dict.update(allo_metrics)
        np.savez_compressed(out_npz, **save_dict)

        rec = {
            'campaign': 'version4_godtier',
            'milestone': 'M30',
            'system_key': system_key,
            'replica_index': int(rep_idx),
            'replica_role': role_map.get(int(rep_idx), f'rep{rep_idx}'),
            'chunk_index': int(chunk_index),
            'solvent': 'implicit_gbn2',
            'execution_mode': f'batched_3replica_single_context (sep={sep_nm}nm, cutoff={cutoff_nm}nm)',
            'trajectory_file': os.path.relpath(os.path.abspath(out_npz), ROOT),
            'sha256': sha256(out_npz),
            'state_xml_file': os.path.relpath(os.path.abspath(out_xml), ROOT),
            'state_xml_sha256': sha256(out_xml),
            'input_pdb': os.path.relpath(os.path.abspath(pdb_path), ROOT),
            'input_pdb_sha256': sha256(pdb_path),
            'resume_xml_sha256': None,
            'force_field': f"amber99sbildn.xml + implicit/gbn2.xml (3-Replica Batched, HMR={hmr_amu}amu)",
            'integrator': 'LangevinMiddleIntegrator (BAOAB)',
            'temperature_k': float(temperature_k),
            'friction_ps': float(friction_ps),
            'time_step_fs': float(time_step_fs),
            'hmr_amu': float(hmr_amu if hmr_amu else 1.008),
            'frame_stride_ps': float(stride_ps),
            'seed': int(seeds[r]),
            'start_time_ps': 0.0,
            'end_time_ps': round(float(times_ps[-1]), 3),
            'total_frames': int(n_frames),
            'total_time_ns': round(n_frames * stride_ps / 1000.0, 6),
            'n_total_particles': int(n_solute_atoms),
            'n_solute_atoms': int(n_solute_atoms),
            'n_solute_heavy_atoms': int(len(heavy_idx)),
            'n_phi': int(phi.shape[1]),
            'n_psi': int(psi.shape[1]),
            'd': int(phi.shape[1] + psi.shape[1]),
            'n_chi1': int(chi1.shape[1]) if len(chi1_idx) else 0,
            'mean_q_bhe': round(float(np.mean(q_bhe)), 4),
            'mean_rmsd_ca_nm': round(float(np.mean(rmsd_ca_nm)), 4),
            'mean_rg_nm': round(float(np.mean(rg_nm)), 4),
            'wall_time_seconds': round(wall / 3.0, 3),
            'ns_per_day': round(3.0 * (n_frames * stride_ps / 1000.0) / max(wall / 86400.0, 1e-12), 2),
            'platform': plat_name,
            'openmm_version': openmm.__version__,
            'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        with open(out_man, 'w') as mf:
            json.dump(rec, mf, indent=2)
        manifests.append(rec)

    return manifests


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb', required=True)
    ap.add_argument('--out-npz', default=None)
    ap.add_argument('--out-xml', default=None)
    ap.add_argument('--out-manifest', default=None)
    ap.add_argument('--batch-3reps', action='store_true', help='Run all 3 replicas simultaneously in 1 GPU context')
    ap.add_argument('--out-dir-traj', default=None)
    ap.add_argument('--out-dir-man', default=None)
    ap.add_argument('--seeds-3rep', default='42001,42002,42003')
    ap.add_argument('--system-key', required=True)
    ap.add_argument('--solvent', default='implicit_gbn2')
    ap.add_argument('--temperature-k', type=float, default=300.0)
    ap.add_argument('--friction-ps', type=float, default=1.0)
    ap.add_argument('--dt-fs', type=float, default=4.0)
    ap.add_argument('--hmr-amu', type=float, default=1.5)
    ap.add_argument('--stride-ps', type=float, default=1.0)
    ap.add_argument('--ns', type=float, default=10.0)
    ap.add_argument('--chunk-index', type=int, default=0)
    ap.add_argument('--replica-index', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42001)
    ap.add_argument('--platform', default='CUDA')
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--resume-xml', default=None)
    ap.add_argument('--alt-pdb', default=None)
    ap.add_argument('--skip-equil', action='store_true')
    ap.add_argument('--cache-dir', default=None)
    args = ap.parse_args()
    if args.batch_3reps:
        seeds_tuple = tuple(int(x) for x in args.seeds_3rep.split(','))
        run_batch_3replicas(
            pdb_path=args.pdb,
            out_dir_traj=args.out_dir_traj,
            out_dir_man=args.out_dir_man,
            system_key=args.system_key,
            temperature_k=args.temperature_k,
            friction_ps=args.friction_ps,
            time_step_fs=args.dt_fs,
            hmr_amu=args.hmr_amu,
            stride_ps=args.stride_ps,
            ns=args.ns,
            chunk_index=args.chunk_index,
            seeds=seeds_tuple,
            platform_name=args.platform,
            threads=args.threads,
            alt_pdb_path=args.alt_pdb,
            skip_equil=args.skip_equil,
        )
    else:
        run_chunk(
            pdb_path=args.pdb,
            out_npz=args.out_npz,
            out_xml=args.out_xml,
            out_manifest=args.out_manifest,
            system_key=args.system_key,
            solvent=args.solvent,
            temperature_k=args.temperature_k,
            friction_ps=args.friction_ps,
            time_step_fs=args.dt_fs,
            hmr_amu=args.hmr_amu,
            stride_ps=args.stride_ps,
            ns=args.ns,
            chunk_index=args.chunk_index,
            replica_index=args.replica_index,
            seed=args.seed,
            platform_name=args.platform,
            threads=args.threads,
            resume_xml=args.resume_xml,
            alt_pdb_path=args.alt_pdb,
            skip_equil=args.skip_equil,
            cache_dir=args.cache_dir,
        )
