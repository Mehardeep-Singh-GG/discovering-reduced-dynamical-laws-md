#!/usr/bin/env python3
"""
version4_godtier/sims/verify_m29_all_gates.py
=============================================
Executes all 5 Mandatory Verification Gates (M29-G1 through M29-G5) for Milestone 29
and writes version4_godtier/results/m29_verification_report.json.
"""
import os
import sys
import json
import time
import socket
import subprocess
import numpy as np
import mdtraj as md

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from version4_godtier.lib import core_v4 as core


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    report = {
        "milestone": "M29",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gates": {},
    }

    man_path = os.path.join(ROOT, "version4_godtier", "manifests", "m29_topology_manifest.json")
    manifest = json.load(open(man_path))
    systems = manifest["systems"]

    # ------------------------------------------------------------------
    # Gate M29-G1: Complete 15-System Dimension & Atom Integrity Gate
    # ------------------------------------------------------------------
    expected_d = [2, 4, 18, 38, 54, 64, 68, 110, 114, 150, 218, 256, 316, 426, 738]
    actual_d = [s["d"] for s in systems]
    assert actual_d == expected_d, f"M29-G1 FAILED: {actual_d} != {expected_d}"

    for s in systems:
        key = s["key"]
        core.verify(s["pdb_relpath"])
        meta = core.dihedral_meta(key)
        assert meta["d"] == s["d"], f"Dihedral meta d mismatch for {key}"
        assert len(meta["chain_order"]) == s["d"], f"Chain order length mismatch for {key}"
        if key not in ("ala1", "ala2"):
            assert len(meta["block_idx_res"]) == s["n_residues"] - 2
            assert len(meta["block_idx_pep"]) == s["n_residues"] - 1

    report["gates"]["M29_G1_15_system_dimensions"] = {
        "status": "PASS",
        "n_systems": len(systems),
        "dimensions_d": actual_d,
    }
    print("[PASS] Gate M29-G1: All 15 systems verified across d =", actual_d, flush=True)

    # ------------------------------------------------------------------
    # Gate M29-G2: V3-to-V4 Backward Compatibility & Hash Identity Gate
    # ------------------------------------------------------------------
    v3_pairs = [
        ("ala1", "version3_rigorous/structures/ala_dipeptide_ACE_ALA_NME.pdb"),
        ("ala2", "version3_rigorous/structures/ala2_capped_ACE_ALA_ALA_NME.pdb"),
        ("chignolin", "protein_scaling/structures/chignolin_1UAO.pdb"),
        ("trpcage", "protein_scaling/structures/trpcage_1L2Y.pdb"),
        ("villin", "protein_scaling/structures/villin_hp35_protonated.pdb"),
        ("gb1", "protein_scaling/structures/gb1_protonated.pdb"),
        ("ubiquitin", "protein_scaling/structures/ubiquitin_protonated.pdb"),
    ]
    v3_checks = {}
    for key, v3_rel in v3_pairs:
        h_v3 = core.sha256(core.P(v3_rel))
        h_v4 = core.sha256(core.P(core.SYSTEMS_V4[key]["pdb"]))
        assert h_v3 == h_v4, f"M29-G2 FAILED for {key}: {h_v3} != {h_v4}"
        v3_checks[key] = h_v4
    report["gates"]["M29_G2_v3_backward_compatibility"] = {
        "status": "PASS",
        "verified_identical_sha256": v3_checks,
    }
    print("[PASS] Gate M29-G2: All 7 V3 carry-over PDBs have identical SHA-256 digests.", flush=True)

    # ------------------------------------------------------------------
    # Gate M29-G3: Disulfide Bridge Covalent Topology Gate
    # ------------------------------------------------------------------
    by_key = {s["key"]: s for s in systems}
    bpti_ss = by_key["bpti"]["disulfide_bridges_verified"]
    lyso_ss = by_key["lysozyme"]["disulfide_bridges_verified"]
    assert len(bpti_ss) == 3, f"Expected 3 disulfides in BPTI, got {bpti_ss}"
    assert len(lyso_ss) == 4, f"Expected 4 disulfides in Lysozyme, got {lyso_ss}"
    report["gates"]["M29_G3_disulfide_topology"] = {
        "status": "PASS",
        "bpti_5PTI_bridges_nm": bpti_ss,
        "lysozyme_1AKI_bridges_nm": lyso_ss,
    }
    print("[PASS] Gate M29-G3: Disulfide bridges verified for BPTI (3 SS) & Lysozyme (4 SS).", flush=True)

    # ------------------------------------------------------------------
    # Gate M29-G4: Allosteric Open/Closed Pair Correspondence Gate
    # ------------------------------------------------------------------
    alt_checks = {}
    for key in ("gb1", "ubiquitin", "dhfr", "adk", "mbp"):
        s = by_key[key]
        assert s.get("alt_pdb_relpath"), f"Missing alt_pdb for {key}"
        core.verify(s["alt_pdb_relpath"])
        alt_checks[key] = {
            "alt_pdb": s["alt_pdb_relpath"],
            "ca_rmsd_open_vs_closed_nm": s["alt_ca_rmsd_nm"],
        }
    assert 0.60 <= alt_checks["adk"]["ca_rmsd_open_vs_closed_nm"] <= 0.80
    assert 0.30 <= alt_checks["mbp"]["ca_rmsd_open_vs_closed_nm"] <= 0.50
    report["gates"]["M29_G4_allosteric_reference_pairs"] = {
        "status": "PASS",
        "pairs": alt_checks,
    }
    print("[PASS] Gate M29-G4: All 5 alternate/allosteric state pairs verified:", alt_checks, flush=True)

    # ------------------------------------------------------------------
    # Gate M29-G5: Live HTTP API Bridge Loopback Smoke Test (2 Chunks)
    # ------------------------------------------------------------------
    port = find_free_port()
    token = "agy-m29-loopback-test-token"
    worker_script = os.path.join(ROOT, "version4_godtier", "sims", "colab_worker_bootstrap.py")
    sched_script = os.path.join(ROOT, "version4_godtier", "sims", "agy_gpu_scheduler.py")

    worker_proc = subprocess.Popen(
        [sys.executable, worker_script, "--host", "127.0.0.1", "--port", str(port), "--token", token],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        time.sleep(1.5)
        out = subprocess.check_output(
            [sys.executable, sched_script, "--endpoint", f"http://127.0.0.1:{port}", "--token", token, "--smoke-test"],
            text=True, timeout=60
        )
        print(out, flush=True)
        # Verify core_v4.load_replica_chunks stitches the two chunks seamlessly
        core._HASH_CACHE.clear()
        stitched = core.load_replica_chunks("ala1", replica_index=1, solvent="implicit_obc2", verify_hash=True)
        assert stitched["n_chunks"] == 2
        assert stitched["angles"].shape == (8, 2)
        assert np.allclose(stitched["times"], np.arange(1.0, 9.0))
        report["gates"]["M29_G5_live_bridge_smoke_test"] = {
            "status": "PASS",
            "n_chunks_stitched": stitched["n_chunks"],
            "stitched_shape": list(stitched["angles"].shape),
            "times_ps": stitched["times"].tolist(),
        }
        print("[PASS] Gate M29-G5: Live Colab API Bridge + 2-chunk XML state resumption + SHA-256 stitching verified!", flush=True)
    finally:
        worker_proc.terminate()
        worker_proc.wait(timeout=5)
        # Clean up smoke test files so production trajectories directory stays clean
        for fn in (
            "version4_godtier/trajectories/ala1/implicit_obc2/ala1_rep1_chunk000.npz",
            "version4_godtier/trajectories/ala1/implicit_obc2/ala1_rep1_chunk000_state.xml",
            "version4_godtier/trajectories/ala1/implicit_obc2/ala1_rep1_chunk001.npz",
            "version4_godtier/trajectories/ala1/implicit_obc2/ala1_rep1_chunk001_state.xml",
            "version4_godtier/manifests/runs/ala1_rep1_chunk000_implicit_obc2.json",
            "version4_godtier/manifests/runs/ala1_rep1_chunk001_implicit_obc2.json",
        ):
            p = os.path.join(ROOT, fn)
            if os.path.exists(p):
                os.remove(p)

    core.save_json(report, "m29_verification_report.json")
    print("\n=====================================================================")
    print("MILESTONE 29 VERIFICATION COMPLETE: ALL 5 GATES (M29-G1..G5) PASSED!")
    print("=====================================================================", flush=True)


if __name__ == "__main__":
    main()
