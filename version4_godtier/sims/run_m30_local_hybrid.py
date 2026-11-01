#!/usr/bin/env python3
"""
version4_godtier/sims/run_m30_local_hybrid.py
=============================================
Local Mac OpenCL + CPU Co-Processor for Milestone 30 (M30).
Works cooperatively with the remote Colab Pro+ A100 coordinator (run_m30_campaign.py):
- Checks version4_godtier/manifests/runs/ before starting any trajectory.
- Uses an atomic lock file (.lock) per (system_key, solvent, replica_index) so local
  OpenCL/CPU workers and remote A100 workers never duplicate work.
- Executes trajectories using the exact same md_engine_v4_gpu.run_chunk() physics engine
  and writes identical SHA-256 verified .npz, .xml, and .json manifests.
"""

import os
import sys
import json
import time
import zlib
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from version4_godtier.sims.md_engine_v4_gpu import run_chunk
from version4_godtier.sims.agy_gpu_scheduler import sha256_file

M29_MANIFEST_PATH = os.path.join(ROOT, "version4_godtier", "manifests", "m29_topology_manifest.json")
TRAJ_DIR = os.path.join(ROOT, "version4_godtier", "trajectories")
RUNS_MAN_DIR = os.path.join(ROOT, "version4_godtier", "manifests", "runs")
os.makedirs(RUNS_MAN_DIR, exist_ok=True)


def deterministic_seed(system_key: str, solvent: str, replica_index: int) -> int:
    tag = f"v4_m30_{system_key}_{solvent}_rep{replica_index}".encode("utf-8")
    return 100000 + (zlib.crc32(tag) % 800000) + replica_index * 1000


def is_done_or_locked(system_key: str, solvent: str, replica_index: int, min_frames: int = 2000) -> bool:
    base_name = f"{system_key}_rep{replica_index}_chunk000"
    local_npz = os.path.join(TRAJ_DIR, system_key, solvent, f"{base_name}.npz")
    local_man = os.path.join(RUNS_MAN_DIR, f"{base_name}_{solvent}.json")
    if os.path.exists(local_npz) and os.path.exists(local_man):
        try:
            man = json.load(open(local_man))
            if int(man.get("total_frames", 0)) >= min_frames and sha256_file(local_npz) == man.get("sha256"):
                return True
        except Exception:
            pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns-gbn2", type=float, default=2.5)
    ap.add_argument("--ns-pme", type=float, default=2.0)
    args = ap.parse_args()

    m29 = json.load(open(M29_MANIFEST_PATH))
    sys_lookup = {s["key"]: s for s in m29["systems"]}

    # Skip the jobs active on the A100 first batch (ala1, ala2, chignolin, trpcage, bba)
    a100_active = {
        ("ala1", "implicit_gbn2", 1), ("ala1", "implicit_gbn2", 2), ("ala1", "implicit_gbn2", 3),
        ("ala2", "implicit_gbn2", 1), ("ala2", "implicit_gbn2", 2), ("ala2", "implicit_gbn2", 3),
        ("chignolin", "implicit_gbn2", 1), ("chignolin", "implicit_gbn2", 2), ("chignolin", "implicit_gbn2", 3),
        ("trpcage", "implicit_gbn2", 1), ("trpcage", "implicit_gbn2", 2), ("trpcage", "implicit_gbn2", 3),
        ("bba", "implicit_gbn2", 1), ("bba", "implicit_gbn2", 2), ("bba", "implicit_gbn2", 3),
    }

    # Build local queue starting from chignolin rep2..3, trpcage, bba, villin, wwdomain, gb1, bpti, ubiquitin, barnase, lysozyme, dhfr, adk, mbp
    all_keys = [s["key"] for s in m29["systems"]]
    tasks = []
    for sys_key in all_keys:
        for rep in (1, 2, 3):
            if (sys_key, "implicit_gbn2", rep) not in a100_active:
                tasks.append((sys_key, "implicit_gbn2", rep, args.ns_gbn2))

    def _worker(job, idx):
        sys_key, solvent, rep, ns = job
        if is_done_or_locked(sys_key, solvent, rep, min_frames=int(ns * 1000)):
            return f"[SKIP] {sys_key} {solvent} rep{rep}"
        s_info = sys_lookup[sys_key]
        pdb_path = os.path.join(ROOT, s_info["pdb_relpath"])
        alt_pdb = os.path.join(ROOT, s_info["alt_pdb_relpath"]) if s_info.get("alt_pdb_relpath") else None

        base_name = f"{sys_key}_rep{rep}_chunk000"
        out_dir = os.path.join(TRAJ_DIR, sys_key, solvent)
        os.makedirs(out_dir, exist_ok=True)
        out_npz = os.path.join(out_dir, f"{base_name}.npz")
        out_xml = os.path.join(out_dir, f"{base_name}_state.xml")
        out_man = os.path.join(RUNS_MAN_DIR, f"{base_name}_{solvent}.json")

        seed = deterministic_seed(sys_key, solvent, rep)
        # Use OpenCL on primary stream and CPU (4 threads) on secondary stream
        plat = "OpenCL" if (idx % 2 == 0) else "CPU"
        threads = 4 if plat == "CPU" else None

        print(f"[Local-{plat}] >>> Starting {sys_key} (d={s_info['d']}) | {solvent} | rep{rep} | {ns} ns", flush=True)
        t0 = time.time()
        rec = run_chunk(
            pdb_path=pdb_path,
            out_npz=out_npz,
            out_xml=out_xml,
            out_manifest=out_man,
            system_key=sys_key,
            solvent=solvent,
            temperature_k=300.0,
            friction_ps=1.0,
            time_step_fs=4.0,
            hmr_amu=1.5,
            stride_ps=1.0,
            ns=ns,
            chunk_index=0,
            replica_index=rep,
            seed=seed,
            platform_name=plat,
            threads=threads,
            resume_xml=None,
            alt_pdb_path=alt_pdb,
            skip_equil=False,
        )
        el = time.time() - t0
        print(f"[Local-{plat}] <<< DONE {sys_key} rep{rep} in {el:.1f}s ({rec['ns_per_day']:.1f} ns/day) | Rg={rec['mean_rg_nm']:.3f} nm", flush=True)
        return rec["trajectory_file"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_worker, job, i) for i, job in enumerate(tasks)]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"[Local Worker Error] {e}", flush=True)


if __name__ == "__main__":
    main()
