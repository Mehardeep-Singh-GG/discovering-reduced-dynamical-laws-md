#!/usr/bin/env python3
"""
version4_godtier/lib/core_v4.py
===============================
Audited numerical & topological core for the Version 4.0 campaign (15 systems, d = 2 -> 738).
Strict superset of version3_rigorous/lib/core.py enforcing Rules R1-R8.
"""
import os
import glob
import json
import hashlib
import numpy as np
import mdtraj as md

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
V4_DIR = os.path.join(ROOT, 'version4_godtier')
STRUCT_DIR = os.path.join(V4_DIR, 'structures')
META_DIR = os.path.join(STRUCT_DIR, 'metadata')
MAN_DIR = os.path.join(V4_DIR, 'manifests')
RES_DIR = os.path.join(V4_DIR, 'results')
FIG_DIR = os.path.join(V4_DIR, 'figures')

for _d in (STRUCT_DIR, META_DIR, MAN_DIR, RES_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

ORDER_V4 = [
    'ala1', 'ala2', 'chignolin', 'trpcage', 'bba', 'villin', 'wwdomain',
    'gb1', 'bpti', 'ubiquitin', 'barnase', 'lysozyme', 'dhfr', 'adk', 'mbp'
]
PROTEINS_V4 = ORDER_V4[2:]
DT_PS = 1.0

PS = 'protein_scaling'
SYSTEMS_V4 = {
    'ala1': dict(
        name='Alanine Dipeptide (ACE-Ala-NME)', short='Ala1',
        pdb='version4_godtier/structures/ala1_ACE_ALA_NME.pdb', d=2, tier=1, fold='peptide',
        v3_reps=[f'version3_rigorous/trajectories/ala_dipeptide/ala1_rep{r}.npz' for r in (1, 2, 3)],
    ),
    'ala2': dict(
        name='Trialanine / Capped Ala2 (ACE-Ala-Ala-NME)', short='Ala2',
        pdb='version4_godtier/structures/ala2_ACE_ALA_ALA_NME.pdb', d=4, tier=1, fold='peptide',
        v3_reps=[f'version3_rigorous/trajectories/ala2_capped/ala2_rep{r}.npz' for r in (1, 2, 3)],
    ),
    'chignolin': dict(
        name='Chignolin (1UAO)', short='Chignolin',
        pdb='version4_godtier/structures/chignolin_1UAO.pdb', d=18, tier=1, fold='beta-hairpin',
        v3_reps=[f'{PS}/trajectories/chignolin/chignolin_rep1_discovery.npz',
                 f'{PS}/trajectories/chignolin/chignolin_rep2_cv.npz',
                 f'{PS}/trajectories/chignolin/chignolin_rep3_test.npz'],
    ),
    'trpcage': dict(
        name='Trp-cage TC5b (1L2Y)', short='Trp-cage',
        pdb='version4_godtier/structures/trpcage_1L2Y.pdb', d=38, tier=1, fold='alpha/PPII',
        v3_reps=[f'{PS}/trajectories/trpcage/trpcage_rep1_discovery.npz',
                 f'{PS}/trajectories/trpcage/trpcage_rep2_cv.npz',
                 f'{PS}/trajectories/trpcage/trpcage_rep3_test.npz'],
    ),
    'bba': dict(
        name='BBA5 Mini-Protein (1FME)', short='BBA5',
        pdb='version4_godtier/structures/bba_1FME.pdb', d=54, tier=1, fold='beta-beta-alpha',
        v3_reps=[],
    ),
    'villin': dict(
        name='Villin HP35 (2F4K)', short='Villin',
        pdb='version4_godtier/structures/villin_2F4K.pdb', d=64, tier=1, fold='3-helix',
        v3_reps=[f'{PS}/trajectories/villin/villin_rep1_discovery.npz',
                 f'{PS}/trajectories/villin/villin_rep2_cv.npz',
                 f'{PS}/trajectories/villin/villin_rep3_test.npz'],
    ),
    'wwdomain': dict(
        name='WW Domain FiP35 (2F21)', short='WW-FiP35',
        pdb='version4_godtier/structures/wwdomain_2F21.pdb', d=68, tier=1, fold='3-beta-sheet',
        v3_reps=[],
    ),
    'gb1': dict(
        name='GB1 Domain (1PGA)', short='GB1',
        pdb='version4_godtier/structures/gb1_1PGA.pdb',
        alt_pdb='version4_godtier/structures/gb1_3GB1_alt.pdb', d=110, tier=2, fold='alpha/beta',
        v3_reps=[f'{PS}/trajectories/gb1/gb1_rep1_discovery.npz',
                 f'{PS}/trajectories/gb1/gb1_rep2_cv.npz',
                 f'{PS}/trajectories/gb1/gb1_rep3_test.npz'],
    ),
    'bpti': dict(
        name='BPTI (5PTI)', short='BPTI',
        pdb='version4_godtier/structures/bpti_5PTI.pdb', d=114, tier=2, fold='disulfide-rich',
        v3_reps=[],
    ),
    'ubiquitin': dict(
        name='Human Ubiquitin (1UBQ)', short='Ubiquitin',
        pdb='version4_godtier/structures/ubiquitin_1UBQ.pdb',
        alt_pdb='version4_godtier/structures/ubiquitin_1D3Z_alt.pdb', d=150, tier=2, fold='alpha/beta',
        v3_reps=[f'{PS}/trajectories/ubiquitin/ubiquitin_rep1_discovery.npz',
                 f'{PS}/trajectories/ubiquitin/ubiquitin_rep2_cv.npz',
                 f'{PS}/trajectories/ubiquitin/ubiquitin_rep3_test.npz'],
    ),
    'barnase': dict(
        name='Barnase (1BNI)', short='Barnase',
        pdb='version4_godtier/structures/barnase_1BNI.pdb', d=218, tier=2, fold='alpha+beta',
        v3_reps=[],
    ),
    'lysozyme': dict(
        name='HEWL Lysozyme (1AKI)', short='Lysozyme',
        pdb='version4_godtier/structures/lysozyme_1AKI.pdb', d=256, tier=2, fold='alpha+beta-4SS',
        v3_reps=[],
    ),
    'dhfr': dict(
        name='E. coli DHFR (1RX2)', short='DHFR',
        pdb='version4_godtier/structures/dhfr_1RX2.pdb',
        alt_pdb='version4_godtier/structures/dhfr_1RA1_alt.pdb', d=316, tier=3, fold='Rossmann-loop',
        v3_reps=[],
    ),
    'adk': dict(
        name='Adenylate Kinase Open (4AKE)', short='AdK',
        pdb='version4_godtier/structures/adk_4AKE_open.pdb',
        alt_pdb='version4_godtier/structures/adk_1AKE_closed_ref.pdb', d=426, tier=3, fold='3-domain-allosteric',
        v3_reps=[],
    ),
    'mbp': dict(
        name='Maltose-Binding Protein Open (1OMP)', short='MBP',
        pdb='version4_godtier/structures/mbp_1OMP_open.pdb',
        alt_pdb='version4_godtier/structures/mbp_1ANF_closed_ref.pdb', d=738, tier=3, fold='2-lobe-allosteric',
        v3_reps=[],
    ),
}


def P(rel):
    return os.path.join(ROOT, rel)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def expected_hashes_v4():
    exp = {}
    top_man = os.path.join(MAN_DIR, 'm29_topology_manifest.json')
    if os.path.exists(top_man):
        for entry in json.load(open(top_man)).get('systems', []):
            exp[entry['pdb_relpath']] = entry['pdb_sha256']
            if entry.get('alt_pdb_relpath'):
                exp[entry['alt_pdb_relpath']] = entry['alt_pdb_sha256']
    runs_dir = os.path.join(MAN_DIR, 'runs')
    if os.path.isdir(runs_dir):
        for fn in os.listdir(runs_dir):
            if fn.endswith('.json'):
                r = json.load(open(os.path.join(runs_dir, fn)))
                exp[r['trajectory_file']] = r['sha256']
                if 'system_key' in r and 'solvent' in r:
                    local_rel = os.path.join(
                        'version4_godtier', 'trajectories',
                        r['system_key'], r['solvent'],
                        os.path.basename(r['trajectory_file'])
                    )
                    exp[local_rel] = r['sha256']
    # Also include V3 manifests for backward compatibility
    v3_runs = os.path.join(ROOT, 'version3_rigorous', 'manifests', 'runs')
    if os.path.isdir(v3_runs):
        for fn in os.listdir(v3_runs):
            if fn.endswith('.json'):
                r = json.load(open(os.path.join(v3_runs, fn)))
                exp[r['trajectory_file']] = r['sha256']
    for idx, key in enumerate(('chignolin', 'trpcage', 'villin', 'gb1', 'ubiquitin'), start=21):
        m = P(f'{PS}/manifests/m{idx}_execution_manifest.json')
        if os.path.exists(m):
            for r in json.load(open(m)).get('replicas', []):
                exp[r['trajectory_file']] = r['sha256']
    return exp


_HASH_CACHE = {}


def verify(rel):
    if rel in _HASH_CACHE:
        return _HASH_CACHE[rel]
    exp = expected_hashes_v4().get(rel)
    act = sha256(P(rel))
    if exp is None or exp != act:
        raise RuntimeError(f"SHA-256 verification FAILED for {rel}: expected {exp}, got {act}")
    _HASH_CACHE[rel] = act
    return act


_TOP = {}


def topology(key):
    if key not in _TOP:
        _TOP[key] = md.load(P(SYSTEMS_V4[key]['pdb']))
    return _TOP[key]


def dihedral_meta(key):
    """Return full backbone + chi1 metadata including BOTH within-residue (block_idx_res)
    and peptide-bond crankshaft (block_idx_pep) pairings."""
    t = topology(key)
    pi, _ = md.compute_phi(t)
    si, _ = md.compute_psi(t)
    ci, _ = md.compute_chi1(t)
    top = t.topology
    phi_res = np.array([top.atom(q[2]).residue.index for q in pi], dtype=int)
    psi_res = np.array([top.atom(q[1]).residue.index for q in si], dtype=int)
    chi1_res = np.array([top.atom(q[1]).residue.index for q in ci], dtype=int) if len(ci) else np.zeros(0, dtype=int)
    nphi, npsi, nchi1 = len(phi_res), len(psi_res), len(chi1_res)

    pos = np.concatenate([np.array([(q[1] + q[2]) / 2.0 for q in pi]),
                          np.array([(q[1] + q[2]) / 2.0 for q in si])])
    order = np.argsort(pos, kind='stable')
    allres = np.concatenate([phi_res, psi_res])
    alltype = np.array(['phi'] * nphi + ['psi'] * npsi)

    # 1. Within-residue Ramachandran blocks: (phi_r, psi_r) on same residue r
    block_res = sorted(set(phi_res.tolist()) & set(psi_res.tolist()))
    block_idx_res = [
        (int(np.where(phi_res == r)[0][0]), nphi + int(np.where(psi_res == r)[0][0]))
        for r in block_res
    ]

    # 2. Peptide-bond crankshaft blocks: (psi_r, phi_{r+1}) flanking peptide bond r -> r+1
    block_pep_res = []
    block_idx_pep = []
    for k_psi, r_psi in enumerate(psi_res):
        matches = np.where(phi_res == (r_psi + 1))[0]
        if len(matches) > 0:
            k_phi = int(matches[0])
            block_pep_res.append((int(r_psi), int(r_psi + 1)))
            block_idx_pep.append((nphi + k_psi, k_phi))

    resnames = [top.residue(int(r)).name for r in allres]
    return dict(
        phi_res=phi_res, psi_res=psi_res, chi1_res=chi1_res,
        nphi=nphi, npsi=npsi, nchi1=nchi1, d=nphi + npsi,
        chain_order=order, chain_res=allres[order], chain_type=alltype[order],
        chain_resname=np.array(resnames)[order],
        block_res=np.array(block_res, dtype=int),
        block_idx=np.array(block_idx_res, dtype=int),       # alias for V3 compatibility
        block_idx_res=np.array(block_idx_res, dtype=int),
        block_pep_res=np.array(block_pep_res, dtype=int),
        block_idx_pep=np.array(block_idx_pep, dtype=int),
        all_res=allres, all_type=alltype,
    )


def load_angles(rel, verify_hash=True):
    if verify_hash:
        verify(rel)
    z = np.load(P(rel))
    return np.concatenate([z['phi'], z['psi']], axis=1).astype(np.float64)


def load_replica_chunks(key, replica_index=1, solvent='explicit_pme', verify_hash=True):
    """Load and cryptographically stitch all continuous chunks for one replica."""
    pattern = os.path.join(
        V4_DIR, 'trajectories', key, solvent, f"{key}_rep{replica_index}_chunk*.npz"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No chunks found matching {pattern}")
    angles_list, chi1_list, times_list = [], [], []
    prev_end_ps = None
    for fpath in files:
        rel = os.path.relpath(fpath, ROOT)
        if verify_hash:
            verify(rel)
        z = np.load(fpath)
        t = z['times'].astype(np.float64)
        if prev_end_ps is not None:
            dt_gap = float(t[0] - prev_end_ps)
            if abs(dt_gap - DT_PS) > 1e-2:
                raise RuntimeError(
                    f"Broken trajectory time continuity in {rel}: prev_end={prev_end_ps}, start={t[0]}"
                )
        prev_end_ps = float(t[-1])
        ang = np.concatenate([z['phi'], z['psi']], axis=1).astype(np.float64)
        angles_list.append(ang)
        if 'chi1' in z and z['chi1'].shape[1] > 0:
            chi1_list.append(z['chi1'].astype(np.float64))
        times_list.append(t)
    return dict(
        angles=np.concatenate(angles_list, axis=0),
        chi1=np.concatenate(chi1_list, axis=0) if chi1_list else np.zeros((sum(len(a) for a in angles_list), 0)),
        times=np.concatenate(times_list, axis=0),
        n_chunks=len(files),
    )


def atlas_solvent_and_files(key, prefer_v3=False):
    """Return (solvent_tag, [rep1_paths, rep2_paths, rep3_paths]) for any of the 15 Atlas systems."""
    if prefer_v3 and SYSTEMS_V4[key].get('v3_reps'):
        v3_paths = [P(r) for r in SYSTEMS_V4[key]['v3_reps']]
        if len(v3_paths) == 3 and all(os.path.exists(p) for p in v3_paths):
            sol = 'explicit_pme_v3' if key in ('ala1', 'ala2') else 'implicit_obc2_v3'
            return sol, [[p] for p in v3_paths]

    for sol in ('explicit_pme', 'implicit_gbn2'):
        rep_lists = []
        for r_idx in (1, 2, 3):
            pat = os.path.join(V4_DIR, 'trajectories', key, sol, f"{key}_rep{r_idx}_chunk*.npz")
            fls = sorted(glob.glob(pat))
            if fls:
                rep_lists.append(fls)
        if len(rep_lists) == 3:
            return sol, rep_lists

    if SYSTEMS_V4[key].get('v3_reps'):
        v3_paths = [P(r) for r in SYSTEMS_V4[key]['v3_reps']]
        if len(v3_paths) == 3 and all(os.path.exists(p) for p in v3_paths):
            sol = 'explicit_pme_v3' if key in ('ala1', 'ala2') else 'implicit_obc2_v3'
            return sol, [[p] for p in v3_paths]

    raise FileNotFoundError(f"Incomplete 3-replica trajectory set on disk for {key}")


def load_atlas_replicas_v4(key, verify_hash=True, max_frames_per_rep=None, load_3d=False, prefer_v3=False):
    """Load 3 cryptographically SHA-256 verified independent replicas for any of the 15 Atlas systems."""
    solvent, rep_file_lists = atlas_solvent_and_files(key, prefer_v3=prefer_v3)
    top_obj = topology(key)
    ca_sel = top_obj.topology.select('name CA')
    heavy_sel = top_obj.topology.select('not element H')

    reps_out = []
    for r_idx, fls in enumerate(rep_file_lists, start=1):
        ang_l, chi1_l, t_l, q_l, rg_l, rmsd_l = [], [], [], [], [], []
        ca_l, heavy_l, full_l = [], [], []
        heavy_idx_ref = None
        has_full = False
        frames_loaded = 0

        for fpath in fls:
            rel = os.path.relpath(fpath, ROOT)
            if verify_hash:
                verify(rel)
            z = np.load(fpath)
            ang = np.concatenate([z['phi'], z['psi']], axis=1).astype(np.float64)
            n_take = len(ang)
            if max_frames_per_rep is not None:
                rem = max_frames_per_rep - frames_loaded
                if rem <= 0:
                    break
                n_take = min(n_take, rem)
                ang = ang[:n_take]

            ang_l.append(ang)
            t_l.append(z['times'][:n_take].astype(np.float64))
            if 'chi1' in z and z['chi1'].shape[1] > 0:
                chi1_l.append(z['chi1'][:n_take].astype(np.float64))

            if 'q_bhe' in z:
                q_l.append(z['q_bhe'][:n_take].astype(np.float64))
            elif 'q_contacts' in z:
                q_l.append(z['q_contacts'][:n_take].astype(np.float64))

            if 'rg_nm' in z:
                rg_l.append(z['rg_nm'][:n_take].astype(np.float64))
            elif 'rg' in z:
                rg_l.append(z['rg'][:n_take].astype(np.float64))

            if 'rmsd_ca_nm' in z:
                rmsd_l.append(z['rmsd_ca_nm'][:n_take].astype(np.float64))
            elif 'rmsd_ca' in z:
                rmsd_l.append(z['rmsd_ca'][:n_take].astype(np.float64))

            if load_3d:
                if 'positions_ca' in z:
                    ca_l.append(z['positions_ca'][:n_take].astype(np.float64))
                elif 'positions' in z and len(ca_sel) > 0:
                    ca_l.append(z['positions'][:n_take][:, ca_sel, :].astype(np.float64))

                if 'positions_heavy' in z:
                    heavy_l.append(z['positions_heavy'][:n_take].astype(np.float64))
                    heavy_idx_ref = z['heavy_atom_indices'].astype(int)
                elif 'positions' in z and len(heavy_sel) > 0:
                    heavy_l.append(z['positions'][:n_take][:, heavy_sel, :].astype(np.float64))
                    heavy_idx_ref = heavy_sel

                if 'positions' in z:
                    has_full = True
                    full_l.append(z['positions'][:n_take].astype(np.float64))

            frames_loaded += n_take

        angles_cat = np.concatenate(ang_l, axis=0)
        rec = dict(
            replica=r_idx,
            solvent=solvent,
            files=[os.path.relpath(p, ROOT) for p in fls],
            angles=angles_cat,
            chi1=np.concatenate(chi1_l, axis=0) if chi1_l else np.zeros((len(angles_cat), 0)),
            times=np.concatenate(t_l, axis=0),
            q_bhe=np.concatenate(q_l, axis=0) if q_l else None,
            rg_nm=np.concatenate(rg_l, axis=0) if rg_l else None,
            rmsd_ca_nm=np.concatenate(rmsd_l, axis=0) if rmsd_l else None,
        )
        if load_3d:
            rec['positions_ca'] = np.concatenate(ca_l, axis=0) if ca_l else None
            rec['positions_heavy'] = np.concatenate(heavy_l, axis=0) if heavy_l else None
            rec['heavy_atom_indices'] = heavy_idx_ref
            rec['has_full_positions'] = has_full
            rec['positions_full'] = np.concatenate(full_l, axis=0) if full_l else None
        reps_out.append(rec)
    return reps_out


def reconstruct_heavy_mdtraj(key, rep_dict, max_frames=5000):
    """Build an md.Trajectory with full or heavy-atom coordinates placed at exact topology indices."""
    top_obj = topology(key)
    if rep_dict.get('has_full_positions') and rep_dict.get('positions_full') is not None:
        xyz = rep_dict['positions_full'][:max_frames]
        return md.Trajectory(xyz, top_obj.topology), True
    elif rep_dict.get('positions_heavy') is not None and rep_dict.get('heavy_atom_indices') is not None:
        ph = rep_dict['positions_heavy'][:max_frames]
        h_idx = rep_dict['heavy_atom_indices']
        xyz = np.zeros((len(ph), top_obj.topology.n_atoms, 3), dtype=np.float32)
        xyz[:, h_idx, :] = ph
        return md.Trajectory(xyz, top_obj.topology), False
    return None, False


def wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def increments(ang, lag=1):
    return wrap(ang[lag:] - ang[:-lag])


def maxgap_shift(ang_list):
    pooled = np.concatenate(ang_list, axis=0)
    d = pooled.shape[1]
    offs = np.zeros(d)
    bins = np.linspace(-np.pi, np.pi, 361)
    for k in range(d):
        h, _ = np.histogram(pooled[:, k], bins=bins)
        hm = h <= max(1, int(0.0002 * len(pooled)))
        hh = np.concatenate([hm, hm])
        best_len, best_mid, run = 0, None, 0
        for i, v in enumerate(hh):
            run = run + 1 if v else 0
            if run > best_len and run <= len(h):
                best_len, best_mid = run, i - run / 2.0
        if best_mid is None:
            best_mid = float(np.argmin(h))
        centre = bins[0] + (best_mid % len(h) + 0.5) * (bins[1] - bins[0])
        offs[k] = wrap(centre + np.pi)
    return offs


def dpca_plus_coords(ang_list, offs=None):
    if offs is None:
        offs = maxgap_shift(ang_list)
    return [wrap(a - offs) for a in ang_list], offs


def diffusion_tensor(ang_list, lag=1, dt=DT_PS, return_counts=False):
    S1 = 0.0
    S2 = 0.0
    n = 0
    for a in ang_list:
        dx = increments(a, lag)
        S1 = S1 + dx.sum(0)
        S2 = S2 + dx.T @ dx
        n += len(dx)
    mu = S1 / n
    C = (S2 / n - np.outer(mu, mu)) * (n / (n - 1.0))
    D = 0.5 * (C + C.T) / (2.0 * lag * dt)
    return (D, n) if return_counts else D


def corr_from_cov(C):
    s = np.sqrt(np.clip(np.diag(C), 1e-300, None))
    return C / np.outer(s, s)


def block_bootstrap_indices(T, block, rng):
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, max(1, T - block + 1), size=nb)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T]
    return idx


def bootstrap_increment_stat(ang_list, stat_fn, lag=1, n_boot=200, block=200, seed=0):
    rng = np.random.default_rng(seed)
    incs = [increments(a, lag) for a in ang_list]
    point = stat_fn(incs)
    samples = []
    for b in range(n_boot):
        bs = [x[block_bootstrap_indices(len(x), block, rng)] for x in incs]
        samples.append(stat_fn(bs))
    return point, np.array(samples)


def cov_from_increments(incs, lag=1, dt=DT_PS):
    S1 = sum(x.sum(0) for x in incs)
    S2 = sum(x.T @ x for x in incs)
    n = sum(len(x) for x in incs)
    mu = S1 / n
    C = (S2 / n - np.outer(mu, mu)) * n / (n - 1.0)
    return 0.5 * (C + C.T) / (2.0 * lag * dt)


def ci(samples, axis=0, level=0.95):
    lo = np.nanpercentile(samples, 100 * (1 - level) / 2, axis=axis)
    hi = np.nanpercentile(samples, 100 * (1 + level) / 2, axis=axis)
    return lo, hi


def rank_measures(C, thresholds=(0.8, 0.9, 0.95)):
    ev = np.clip(np.sort(np.linalg.eigvalsh(C))[::-1], 0, None)
    cum = np.cumsum(ev) / ev.sum()
    out = {f'r{int(q*100)}': int(np.searchsorted(cum, q - 1e-12) + 1) for q in thresholds}
    out['participation_ratio'] = float(ev.sum() ** 2 / np.sum(ev ** 2))
    out['eigenvalues'] = ev.tolist()
    return out


def native_contact_pairs(key, cutoff_nm=0.45, min_seq_sep=4):
    t = topology(key)
    heavy = t.topology.select('not element H')
    pairs = []
    for ii, a in enumerate(heavy):
        ra = t.topology.atom(a).residue.index
        for b in heavy[ii + 1:]:
            if abs(t.topology.atom(b).residue.index - ra) >= min_seq_sep:
                pairs.append((a, b))
    if not pairs:
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float)
    pairs = np.array(pairs, dtype=int)
    r0 = md.compute_distances(t, pairs)[0]
    sel = r0 < cutoff_nm
    return pairs[sel], r0[sel]


def q_best_hummer(key, positions, beta=50.0, lam=1.8):
    pairs, r0 = native_contact_pairs(key)
    if len(pairs) == 0:
        return np.ones(len(positions), dtype=float)
    t = topology(key)
    traj = md.Trajectory(positions, t.topology)
    r = md.compute_distances(traj, pairs)
    return np.mean(1.0 / (1.0 + np.exp(np.clip(beta * (r - lam * r0), -50.0, 50.0))), axis=1)


def save_json(obj, name):
    path = os.path.join(RES_DIR, name)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=conv)
    return path
