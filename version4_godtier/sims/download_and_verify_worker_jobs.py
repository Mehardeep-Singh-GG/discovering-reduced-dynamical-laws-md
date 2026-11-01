#!/usr/bin/env python3
"""
version4_godtier/sims/download_and_verify_worker_jobs.py
=========================================================
Sequential, rock-solid downloader for completed chunks with SHA-256 verification.
Uses 1MB streaming buffer and 600s timeout per chunk to ensure zero timeouts.
"""
import os
import sys
import json
import time
import hashlib
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAJ_DIR = os.path.join(ROOT, "version4_godtier", "trajectories")
MAN_DIR = os.path.join(ROOT, "version4_godtier", "manifests", "runs")

WORKERS = {
    "blackwell": {
        "base": "https://brokers-park-further-auctions.trycloudflare.com",
        "token": "nexus-v4-c6f6fc5541052848",
        "jobs": [
            ("dhfr", "explicit_pme", 1, 0, "dhfr_rep1_c000_fea6fb", "structures/dhfr_1RX2.pdb"),
            ("dhfr", "explicit_pme", 2, 0, "dhfr_rep2_c000_bf4aa4", "structures/dhfr_1RX2.pdb"),
            ("dhfr", "explicit_pme", 3, 0, "dhfr_rep3_c000_a185ee", "structures/dhfr_1RX2.pdb"),
            ("lysozyme", "explicit_pme", 1, 0, "lysozyme_rep1_c000_dea893", "structures/lysozyme_1AKI.pdb"),
            ("lysozyme", "explicit_pme", 2, 0, "lysozyme_rep2_c000_a4361d", "structures/lysozyme_1AKI.pdb"),
            ("lysozyme", "explicit_pme", 3, 0, "lysozyme_rep3_c000_1f0a08", "structures/lysozyme_1AKI.pdb"),
        ]
    },
    "blackwell2": {
        "base": "https://orange-precipitation-placed-recorded.trycloudflare.com",
        "token": "nexus-v4-b1a05a088567508b",
        "jobs": [
            ("bba", "implicit_gbn2", 1, 0, "bba_rep1_c000_44f247", "structures/bba_1FME.pdb"),
            ("bba", "implicit_gbn2", 2, 0, "bba_rep2_c000_408ce7", "structures/bba_1FME.pdb"),
            ("bba", "implicit_gbn2", 3, 0, "bba_rep3_c000_f974f8", "structures/bba_1FME.pdb"),
            ("villin", "implicit_gbn2", 1, 0, "villin_rep1_c000_33b9b1", "structures/villin_2F4K.pdb"),
            ("villin", "implicit_gbn2", 2, 0, "villin_rep2_c000_797dc8", "structures/villin_2F4K.pdb"),
            ("villin", "implicit_gbn2", 3, 0, "villin_rep3_c000_6dd779", "structures/villin_2F4K.pdb"),
        ]
    }
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(base, token, job_id, artifact, dest_path):
    dest_abs = os.path.abspath(dest_path)
    os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
    tmp_path = dest_abs + f".tmp_dl.{os.getpid()}"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"artifact": artifact}
    for attempt in range(5):
        try:
            with requests.get(
                f"{base}/download_chunk/{job_id}",
                params=params,
                headers=headers,
                stream=True,
                timeout=(30.0, 600.0),
            ) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1048576):
                        if chunk:
                            f.write(chunk)
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                os.replace(tmp_path, dest_abs)
                return True
        except Exception as e:
            print(f"  [Retry {attempt+1}/5] {artifact} failed: {e}", flush=True)
            time.sleep(2.0 * (attempt + 1))
        finally:
            if os.path.exists(tmp_path) and not os.path.exists(dest_abs):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    return False


def main():
    print("=== Downloading and Verifying Completed Worker Jobs ===", flush=True)
    total_downloaded = 0
    for wname, wdata in WORKERS.items():
        base = wdata["base"]
        token = wdata["token"]
        print(f"\nProcessing Worker: {wname} ({base})", flush=True)
        for sys_key, solvent, rep, chk, job_id, pdb_rel in wdata["jobs"]:
            base_name = f"{sys_key}_rep{rep}_chunk{chk:03d}"
            out_dir = os.path.join(TRAJ_DIR, sys_key, solvent)
            os.makedirs(out_dir, exist_ok=True)
            local_npz = os.path.join(out_dir, f"{base_name}.npz")
            local_xml = os.path.join(out_dir, f"{base_name}_state.xml")
            local_man = os.path.join(MAN_DIR, f"{base_name}_{solvent}.json")

            if os.path.exists(local_npz) and os.path.exists(local_xml) and os.path.exists(local_man):
                print(f"  [Already Done] {base_name} ({solvent})", flush=True)
                continue

            print(f"  -> Downloading {base_name} ({solvent}, job {job_id})...", flush=True)
            t0 = time.time()
            # 1. Download manifest
            ok_m = download_file(base, token, job_id, "manifest", local_man)
            if not ok_m:
                print(f"  FAILED manifest download for {job_id}")
                continue
            manifest = json.load(open(local_man))

            # 2. Download XML
            ok_x = download_file(base, token, job_id, "xml", local_xml)
            if not ok_x:
                print(f"  FAILED xml download for {job_id}")
                continue

            # 3. Download NPZ
            ok_n = download_file(base, token, job_id, "npz", local_npz)
            if not ok_n:
                print(f"  FAILED npz download for {job_id}")
                continue

            # 4. Verify SHA-256
            npz_sha = sha256_file(local_npz)
            xml_sha = sha256_file(local_xml)
            if npz_sha != manifest["sha256"]:
                print(f"  ERROR: SHA mismatch for NPZ: {npz_sha} != {manifest['sha256']}")
                continue
            if xml_sha != manifest["state_xml_sha256"]:
                print(f"  ERROR: SHA mismatch for XML: {xml_sha} != {manifest['state_xml_sha256']}")
                continue

            # Rewrite relative paths in manifest
            manifest["trajectory_file"] = os.path.relpath(local_npz, ROOT)
            manifest["state_xml_file"] = os.path.relpath(local_xml, ROOT)
            manifest["input_pdb"] = pdb_rel
            with open(local_man, "w") as mf:
                json.dump(manifest, mf, indent=2)

            dt = time.time() - t0
            size_mb = os.path.getsize(local_npz) / (1024 * 1024)
            print(f"  [VERIFIED] {base_name} ({size_mb:.1f} MB in {dt:.1f}s) | SHA={npz_sha[:16]}... | Rate={manifest.get('ns_per_day', 0):.0f} ns/day", flush=True)
            total_downloaded += 1

    print(f"\nFinished: Downloaded and verified {total_downloaded} new chunks.", flush=True)


if __name__ == "__main__":
    main()
