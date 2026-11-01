#!/usr/bin/env python3
"""
download_worker3_and_worker1.py - Downloads completed chunks from Worker 1 (A100) and Worker 3 (Blackwell 2).
"""
import os, sys, time, requests

ROOT = "/Users/mac/molecular dynamics"
TRAJ_DIR = os.path.join(ROOT, "version4_godtier", "trajectories")
MAN_DIR = os.path.join(ROOT, "version4_godtier", "manifests", "runs")

WORKERS = [
    {
        "url": "https://orange-precipitation-placed-recorded.trycloudflare.com",
        "token": "nexus-v4-b1a05a088567508b",
        "jobs": [
            ("bba", "implicit_gbn2", 1, 0, "bba_rep1_c000_44f247"),
            ("bba", "implicit_gbn2", 2, 0, "bba_rep2_c000_408ce7"),
            ("bba", "implicit_gbn2", 3, 0, "bba_rep3_c000_f974f8"),
            ("villin", "implicit_gbn2", 1, 0, "villin_rep1_c000_33b9b1"),
            ("villin", "implicit_gbn2", 2, 0, "villin_rep2_c000_797dc8"),
            ("villin", "implicit_gbn2", 3, 0, "villin_rep3_c000_6dd779"),
        ]
    }
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
    print("=== Downloading Worker 1 and Worker 3 Jobs ===", flush=True)
    for w in WORKERS:
        url = w["url"]
        tok = w["token"]
        for sys_k, solv, rep, chk, jid in w["jobs"]:
            out_dir = os.path.join(TRAJ_DIR, sys_k, solv)
            npz_dest = os.path.join(out_dir, f"{sys_k}_rep{rep}_chunk000.npz")
            xml_dest = os.path.join(out_dir, f"{sys_k}_rep{rep}_chunk000_state.xml")
            man_dest = os.path.join(MAN_DIR, f"{sys_k}_rep{rep}_chunk000_{solv}.json")
            
            print(f"Processing {sys_k} rep{rep} (job {jid})...", flush=True)
            download_file(url, tok, jid, "manifest", man_dest)
            download_file(url, tok, jid, "xml", xml_dest)
            download_file(url, tok, jid, "npz", npz_dest)

if __name__ == "__main__":
    main()
