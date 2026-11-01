#!/usr/bin/env python3
"""
Sequential, fail-closed execution of V4 milestones r34 -> r41 on the 100% verified 15-system dataset.
Zero concurrency so CPU/memory benchmarks (e.g. r40) have dedicated single-process resources.
"""
import os
import sys
import time
import subprocess

ROOT = "/Users/mac/molecular dynamics"
SCRIPTS = [
    "version4_godtier/analysis/r34_solvent_internal_friction_v4.py",
    "version4_godtier/analysis/r35_nmr_s2_bfactors_v4.py",
    "version4_godtier/analysis/r36_lopo_cross_protein_v4.py",
    "version4_godtier/analysis/r37_allosteric_hinges_adk_mbp_v4.py",
    "version4_godtier/analysis/r38_distal_allostery_dhfr_ubq_v4.py",
    "version4_godtier/analysis/r39_fast_folding_kinetics_v4.py",
    "version4_godtier/analysis/r40_fair_sota_benchmark_v4.py",
    "version4_godtier/analysis/r41_master_synthesis.py",
]


def main():
    t_total = time.time()
    for idx, rel_path in enumerate(SCRIPTS, 1):
        full_path = os.path.join(ROOT, rel_path)
        print(f"\n{'='*76}", flush=True)
        print(f"[{idx}/{len(SCRIPTS)}] EXECUTING {rel_path}", flush=True)
        print(f"{'='*76}", flush=True)
        t0 = time.time()
        proc = subprocess.run([sys.executable, full_path], cwd=ROOT)
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"\n[FAIL-CLOSED] {rel_path} exited with code {proc.returncode} after {dt:.1f}s!", flush=True)
            sys.exit(proc.returncode)
        print(f"[COMPLETED] {rel_path} in {dt:.1f}s (exit 0)", flush=True)

    print(f"\n{'#'*76}", flush=True)
    print(f"ALL MILESTONES r34 -> r41 COMPLETED CLEANLY IN {time.time() - t_total:.1f}s!", flush=True)
    print(f"{'#'*76}", flush=True)


if __name__ == "__main__":
    main()
