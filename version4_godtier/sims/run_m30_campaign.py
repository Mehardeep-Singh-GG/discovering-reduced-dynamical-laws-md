#!/usr/bin/env python3
"""
version4_godtier/sims/run_m30_campaign.py  (FULL-SPEC / MAX-RIGOR MODE)
======================================================================
Implements the M30 spec exactly: 15 systems x 3 independent replicas x 2 solvents
(implicit_gbn2 and explicit_pme TIP3P-FB/0.15 M NaCl), 50 ns per replica, executed as
5 x 10 ns chunks chained by OpenMM XML state (sim.loadState), 1.0 ps stride, 4 fs HMR.

No 3-replica batching / anchor restraints are used here: every replica is a plain,
unrestrained, independent simulation (batching remains available only as a speed tool).

Resumable: every chunk is SHA-256 verified on disk and skipped on restart. A replica's
chunk k+1 is only dispatched from the verified chunk k state XML (Rule R4 continuity).
"""
import os
import sys
import json
import time
import zlib
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from version4_godtier.sims.agy_gpu_scheduler import ColabBridgeClient, run_remote_job, sha256_file

PROGRESS_PATH = os.path.join(ROOT, "version4_godtier", "manifests", "m30_campaign_progress.json")
M29_MANIFEST_PATH = os.path.join(ROOT, "version4_godtier", "manifests", "m29_topology_manifest.json")
LOCK = threading.Lock()

SYSTEMS = ["ala1", "ala2", "chignolin", "trpcage", "bba", "villin", "wwdomain", "gb1",
           "bpti", "ubiquitin", "barnase", "lysozyme", "dhfr", "adk", "mbp"]
NS_PER_REPLICA = 50.0
CHUNK_NS = 10.0
N_CHUNKS = int(round(NS_PER_REPLICA / CHUNK_NS))
STRIDE_PS = 1.0


def deterministic_seed(system_key, solvent, replica_index):
    tag = f"v4_m30_{system_key}_{solvent}_rep{replica_index}".encode("utf-8")
    return 100000 + (zlib.crc32(tag) % 800000) + replica_index * 1000


def paths(sys_key, solvent, rep, chk):
    base = f"{sys_key}_rep{rep}_chunk{chk:03d}"
    npz = os.path.join(ROOT, "version4_godtier", "trajectories", sys_key, solvent, f"{base}.npz")
    xml = os.path.join(ROOT, "version4_godtier", "trajectories", sys_key, solvent, f"{base}_state.xml")
    man = os.path.join(ROOT, "version4_godtier", "manifests", "runs", f"{base}_{solvent}.json")
    return npz, xml, man


def verified(sys_key, solvent, rep, chk, min_frames):
    npz, xml, man = paths(sys_key, solvent, rep, chk)
    if not (os.path.exists(npz) and os.path.exists(xml) and os.path.exists(man)):
        return None
    try:
        m = json.load(open(man))
        if int(m.get("total_frames", 0)) >= min_frames and sha256_file(npz) == m.get("sha256") \
                and sha256_file(xml) == m.get("state_xml_sha256"):
            return m
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--solvents", default="implicit_gbn2,explicit_pme")
    ap.add_argument("--concurrency", type=int, default=6, help="Concurrent replica chains (MPS-shared GPU)")
    ap.add_argument("--reverse", action="store_true", help="Process largest explicit-water systems first")
    ap.add_argument("--worker-id", required=True, help="Unique id for this GPU endpoint (claims)")
    ap.add_argument("--adopt-json", default=None, help="JSON {chain_key: {chunk: job_id}} of in-flight jobs to adopt")
    args = ap.parse_args()
    solvents = [s.strip() for s in args.solvents.split(",") if s.strip()]

    sys_lookup = {s["key"]: s for s in json.load(open(M29_MANIFEST_PATH))["systems"]}
    client = ColabBridgeClient(args.endpoint, args.token)
    gpu = client.gpu_status()
    platform = "CUDA" if "CUDA" in gpu.get("openmm_platforms", []) else "CPU"
    print(f"[M30-FULL] {gpu.get('nvidia_smi')} | OpenMM {gpu.get('openmm_version')} | platform={platform}", flush=True)
    print(f"[M30-FULL] {len(SYSTEMS)} systems x 3 reps x {len(solvents)} solvents x {NS_PER_REPLICA} ns "
          f"({N_CHUNKS} x {CHUNK_NS} ns chunks)", flush=True)

    up, up_alt = {}, {}
    for k in SYSTEMS:
        s = sys_lookup[k]
        up[k] = client.upload_pdb(os.path.join(ROOT, s["pdb_relpath"]))
        if s.get("alt_pdb_relpath") and os.path.exists(os.path.join(ROOT, s["alt_pdb_relpath"])):
            up_alt[k] = client.upload_pdb(os.path.join(ROOT, s["alt_pdb_relpath"]))
    print(f"[M30-FULL] Uploaded+verified {len(up)} primary / {len(up_alt)} alt PDBs", flush=True)

    min_frames = int(round(CHUNK_NS * 1000.0 / STRIDE_PS))
    total_chunks = len(SYSTEMS) * 3 * len(solvents) * N_CHUNKS
    done = {"n": 0}
    t0 = time.time()

    claim_dir = os.path.join(ROOT, "version4_godtier", "manifests", "claims")
    os.makedirs(claim_dir, exist_ok=True)
    adopt = json.load(open(args.adopt_json)) if args.adopt_json else {}

    def claim(sys_key, solvent, rep):
        """Atomic per-chain claim: a chain (and its solvated-box cache) stays on one GPU."""
        cp = os.path.join(claim_dir, f"{sys_key}_{solvent}_rep{rep}.claim")
        try:
            fd = os.open(cp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, args.worker_id.encode()); os.close(fd)
            return True
        except FileExistsError:
            return open(cp).read().strip() == args.worker_id

    def chain(sys_key, solvent, rep):
        if not claim(sys_key, solvent, rep):
            return
        s = sys_lookup[sys_key]
        ckey = f"{sys_key}|{solvent}|{rep}"
        seed = deterministic_seed(sys_key, solvent, rep)
        for chk in range(N_CHUNKS):
            m = verified(sys_key, solvent, rep, chk, min_frames)
            if m is None:
                resume = None
                if chk > 0:
                    _, prev_xml, _ = paths(sys_key, solvent, rep, chk - 1)
                    resume = prev_xml
                print(f">>> {sys_key}(d={s['d']}) {solvent} rep{rep} chunk{chk} ({CHUNK_NS} ns)", flush=True)
                m = run_remote_job(
                    client=client, system_key=sys_key, pdb_relpath=s["pdb_relpath"], solvent=solvent,
                    replica_index=rep, chunk_index=chk, chunk_ns=CHUNK_NS, dt_fs=4.0, hmr_amu=1.5,
                    stride_ps=STRIDE_PS, seed=seed, platform=platform, resume_xml_path=resume,
                    alt_pdb_relpath=s.get("alt_pdb_relpath"), uploaded_pdb=up[sys_key],
                    uploaded_alt_pdb=up_alt.get(sys_key), skip_equil=False, poll_sec=5.0,
                    adopt_job_id=adopt.get(ckey, {}).get(str(chk)))
            with LOCK:
                done["n"] += 1
                hrs = (time.time() - t0) / 3600.0
                print(f"[Progress] {done['n']}/{total_chunks} chunks verified | {hrs:.2f} h elapsed | "
                      f"{sys_key} {solvent} rep{rep} c{chk} @ {m.get('ns_per_day', 0):.0f} ns/day", flush=True)
                os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
                json.dump({"updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "chunks_done": done["n"], "chunks_total": total_chunks},
                          open(PROGRESS_PATH, "w"), indent=2)

    # Small systems first per solvent so analysis milestones can start early
    jobs = [(k, sv, r) for sv in solvents for k in SYSTEMS for r in (1, 2, 3)]
    if args.reverse:
        jobs = jobs[::-1]
    # Chains already claimed by this worker (incl. adopted in-flight jobs) go first
    def _mine(j):
        cp = os.path.join(ROOT, "version4_godtier", "manifests", "claims", f"{j[0]}_{j[1]}_rep{j[2]}.claim")
        return 0 if (os.path.exists(cp) and open(cp).read().strip() == args.worker_id) else 1
    jobs.sort(key=_mine)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(chain, *j): j for j in jobs}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"[CHAIN-ERROR] {futs[f]}: {e}", flush=True)

    print(f"[M30-FULL] Finished: {done['n']}/{total_chunks} chunks verified.", flush=True)


if __name__ == "__main__":
    main()
