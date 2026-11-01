#!/usr/bin/env python3
"""
Milestone 29 / 30 Queue Manifest Builder & Colab Pro+ One-Paste Cell Generator
==============================================================================
Generates:
1. version4_godtier/manifests/queue_tier1_fastfolders.json
2. version4_godtier/manifests/queue_tier2_globular_nmr.json
3. version4_godtier/manifests/queue_tier3_allosteric.json
4. version4_godtier/sims/COLAB_PRO_PLUS_PASTE_CELL.py (Self-contained 1-cell Colab Pro+ bootstrap)
"""

import json
import zlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V4_DIR = PROJECT_ROOT / "version4_godtier"
MANIFESTS_DIR = V4_DIR / "manifests"
SIMS_DIR = V4_DIR / "sims"


def build_queues():
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    # Tier 1: Fast-Folders & Benchmark Systems (Explicit TIP3P-FB PME + Matched GBn2)
    tier1_systems = [
        ("chignolin", "chignolin_1UAO.pdb", 18, 200.0, 4),
        ("trpcage", "trpcage_2JOF.pdb", 38, 200.0, 4),
        ("bba", "bba_1FME.pdb", 54, 200.0, 4),
        ("villin", "villin_2F4K.pdb", 64, 200.0, 4),
        ("wwdomain", "wwdomain_2F21.pdb", 68, 200.0, 4),
    ]
    tier1_jobs = []
    for sys_id, pdb_file, d, total_ns, n_chunks in tier1_systems:
        chunk_ns = total_ns / n_chunks
        for solvent in ["explicit_pme", "implicit_gbn2"]:
            for rep in [1, 2, 3]:
                for chunk in range(1, n_chunks + 1):
                    tier1_jobs.append({
                        "system_id": sys_id,
                        "pdb_filename": pdb_file,
                        "d_expected": d,
                        "solvent_model": solvent,
                        "replica_idx": rep,
                        "chunk_idx": chunk,
                        "chunk_ns": chunk_ns,
                        "dt_fs": 4.0 if solvent == "explicit_pme" else 2.0,
                        "save_interval_ps": 1.0,
                        "seed": 2026000 + (zlib.crc32(f"{sys_id}_{solvent}".encode()) % 10000) + rep * 100 + chunk,
                    })

    # Tier 2: Globular NMR & Disulfide Cross-Linked Systems
    tier2_systems = [
        ("gb1", "gb1_1PGB.pdb", 110, 150.0, 3),
        ("bpti", "bpti_5PTI.pdb", 114, 150.0, 3),
        ("ubiquitin", "ubiquitin_1UBQ.pdb", 150, 200.0, 4),
        ("barnase", "barnase_1BNI.pdb", 218, 150.0, 3),
        ("lysozyme", "lysozyme_2LZT.pdb", 256, 150.0, 3),
        ("dhfr", "dhfr_1RX2.pdb", 316, 200.0, 4),
    ]
    tier2_jobs = []
    for sys_id, pdb_file, d, total_ns, n_chunks in tier2_systems:
        chunk_ns = total_ns / n_chunks
        solvents = ["explicit_pme", "implicit_gbn2"] if sys_id in ("gb1", "bpti", "ubiquitin") else ["explicit_pme"]
        for solvent in solvents:
            for rep in [1, 2, 3]:
                for chunk in range(1, n_chunks + 1):
                    tier2_jobs.append({
                        "system_id": sys_id,
                        "pdb_filename": pdb_file,
                        "d_expected": d,
                        "solvent_model": solvent,
                        "replica_idx": rep,
                        "chunk_idx": chunk,
                        "chunk_ns": chunk_ns,
                        "dt_fs": 4.0 if solvent == "explicit_pme" else 2.0,
                        "save_interval_ps": 1.0,
                        "seed": 3026000 + (zlib.crc32(f"{sys_id}_{solvent}".encode()) % 10000) + rep * 100 + chunk,
                    })

    # Tier 3: Large Multi-Domain Allosteric Machines (AdK d=426, MBP d=738)
    tier3_systems = [
        ("adk", "adk_4AKE_open.pdb", 426, 250.0, 5),
        ("mbp", "mbp_1OMP_open.pdb", 738, 250.0, 5),
    ]
    tier3_jobs = []
    for sys_id, pdb_file, d, total_ns, n_chunks in tier3_systems:
        chunk_ns = total_ns / n_chunks
        for solvent in ["explicit_pme", "implicit_gbn2"]:
            for rep in [1, 2, 3]:
                for chunk in range(1, n_chunks + 1):
                    tier3_jobs.append({
                        "system_id": sys_id,
                        "pdb_filename": pdb_file,
                        "d_expected": d,
                        "solvent_model": solvent,
                        "replica_idx": rep,
                        "chunk_idx": chunk,
                        "chunk_ns": chunk_ns,
                        "dt_fs": 4.0 if solvent == "explicit_pme" else 2.0,
                        "save_interval_ps": 2.0,
                        "seed": 4026000 + (zlib.crc32(f"{sys_id}_{solvent}".encode()) % 10000) + rep * 100 + chunk,
                    })

    q1 = MANIFESTS_DIR / "queue_tier1_fastfolders.json"
    q2 = MANIFESTS_DIR / "queue_tier2_globular_nmr.json"
    q3 = MANIFESTS_DIR / "queue_tier3_allosteric.json"

    q1.write_text(json.dumps({"tier": "tier1_fastfolders", "n_jobs": len(tier1_jobs), "jobs": tier1_jobs}, indent=2))
    q2.write_text(json.dumps({"tier": "tier2_globular_nmr", "n_jobs": len(tier2_jobs), "jobs": tier2_jobs}, indent=2))
    q3.write_text(json.dumps({"tier": "tier3_allosteric", "n_jobs": len(tier3_jobs), "jobs": tier3_jobs}, indent=2))

    print(f"Wrote {q1.name} ({len(tier1_jobs)} jobs)")
    print(f"Wrote {q2.name} ({len(tier2_jobs)} jobs)")
    print(f"Wrote {q3.name} ({len(tier3_jobs)} jobs)")


def build_colab_paste_cell():
    """Creates a single self-contained Python cell to paste into Google Colab Pro+."""
    engine_code = (SIMS_DIR / "md_engine_v4_gpu.py").read_text()
    worker_code = (SIMS_DIR / "colab_worker_bootstrap.py").read_text()

    paste_cell = f'''# ==============================================================================
# ANTIGRAVITY <-> GOOGLE COLAB PRO+ REMOTE GPU WORKER (ONE-PASTE BOOTSTRAP CELL)
# ==============================================================================
# Instructions:
# 1. Open Google Colab Pro+ (https://colab.research.google.com)
# 2. Select Runtime -> Change runtime type -> A100 GPU or L4 GPU + High-RAM
# 3. Paste this entire cell into the first cell and click Run.
# 4. Copy the printed ANTIGRAVITY_COLAB_ENDPOINT and ANTIGRAVITY_COLAB_TOKEN
#    back into the Antigravity chat!
# ==============================================================================

import os, subprocess, sys, secrets
from pathlib import Path

print("[1/3] Installing OpenMM 8 CUDA + MDTraj + Cloudflared...")
subprocess.run('pip install -q "openmm[cuda12]" mdtraj scipy numpy || pip install -q openmm mdtraj scipy numpy', shell=True, check=False)

WORKDIR = Path("/content/nexus_v4_worker")
SIMS_DIR = WORKDIR / "version4_godtier" / "sims"
SIMS_DIR.mkdir(parents=True, exist_ok=True)

print("[2/3] Writing md_engine_v4_gpu.py and colab_worker_bootstrap.py...")
(SIMS_DIR / "md_engine_v4_gpu.py").write_text({repr(engine_code)})
(SIMS_DIR / "colab_worker_bootstrap.py").write_text({repr(worker_code)})

token = os.environ.get("ANTIGRAVITY_COLAB_TOKEN", "nexus-v4-" + secrets.token_hex(8))
print("[3/3] Launching REST GPU Worker + Cloudflare Tunnel...", flush=True)
proc = subprocess.Popen(
    [
        sys.executable, "-u",
        str(SIMS_DIR / "colab_worker_bootstrap.py"),
        "--port", "8765",
        "--token", token,
        "--workdir", str(WORKDIR),
        "--tunnel",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in proc.stdout:
    print(line, end="", flush=True)
'''
    cell_path = SIMS_DIR / "COLAB_PRO_PLUS_PASTE_CELL.py"
    cell_path.write_text(paste_cell)
    print(f"Wrote self-contained Colab Pro+ cell: {cell_path} ({len(paste_cell)} bytes)")


if __name__ == "__main__":
    build_queues()
    build_colab_paste_cell()
