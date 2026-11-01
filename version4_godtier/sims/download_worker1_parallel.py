#!/usr/bin/env python3
"""
download_worker1_parallel.py - Dedicated parallel downloader for Worker 1 (A100): AdK and MBP.
Runs concurrently with Worker 2 and Worker 3 downloaders.
"""
import os, sys, time, requests

ROOT = "/Users/mac/molecular dynamics"
TRAJ_DIR = os.path.join(ROOT, "version4_godtier", "trajectories")
MAN_DIR = os.path.join(ROOT, "version4_godtier", "manifests", "runs")

W1_URL = "https://locate-most-survey-hide.trycloudflare.com"
W1_TOKEN = "nexus-v4-04d55f6603ce2b56"

JOBS = [
    ("adk", "explicit_pme", 1, 0, "adk_rep1_c000_0c3ddb"),
    ("adk", "explicit_pme", 2, 0, "adk_rep2_c000_c40665"),
    ("adk", "explicit_pme", 3, 0, "adk_rep3_c000_8c1273"),
    ("mbp", "explicit_pme", 1, 0, "mbp_rep1_c000_107192"),
    ("mbp", "explicit_pme", 2, 0, "mbp_rep2_c000_919c67"),
    ("mbp", "explicit_pme", 3, 0, "mbp_rep3_c000_0c2b11"),
]

def download_file(url, token, jid, artifact, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  [Already Exists] {dest} ({os.path.getsize(dest)/(1024*1024):.1f} MB)", flush=True)
        return True
    dest_abs = os.path.abspath(dest)
    os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
    tmp_path = dest_abs + f".tmp.{os.getpid()}"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"artifact": artifact}
    for attempt in range(5):
        try:
            with requests.get(f"{url}/download_chunk/{jid}", params=params, headers=headers, stream=True, timeout=(30.0, 600.0)) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1048576):
                        if chunk:
                            f.write(chunk)
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                os.replace(tmp_path, dest_abs)
                print(f"  [Downloaded] {dest_abs} ({os.path.getsize(dest_abs)/(1024*1024):.1f} MB)", flush=True)
                return True
        except Exception as e:
            print(f"  [Retry {attempt+1}] {dest} failed: {e}", flush=True)
            time.sleep(2.0)
    return False

def main():
    print("=== Dedicated Worker 1 (A100) Downloader (AdK & MBP) ===", flush=True)
    for sys_k, solv, rep, chk, jid in JOBS:
        out_dir = os.path.join(TRAJ_DIR, sys_k, solv)
        npz_dest = os.path.join(out_dir, f"{sys_k}_rep{rep}_chunk000.npz")
        xml_dest = os.path.join(out_dir, f"{sys_k}_rep{rep}_chunk000_state.xml")
        man_dest = os.path.join(MAN_DIR, f"{sys_k}_rep{rep}_chunk000_{solv}.json")

        print(f"Processing {sys_k} rep{rep} (job {jid})...", flush=True)
        download_file(W1_URL, W1_TOKEN, jid, "manifest", man_dest)
        download_file(W1_URL, W1_TOKEN, jid, "xml", xml_dest)
        download_file(W1_URL, W1_TOKEN, jid, "npz", npz_dest)

if __name__ == "__main__":
    main()
