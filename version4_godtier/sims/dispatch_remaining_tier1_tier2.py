#!/usr/bin/env python3
"""
dispatch_remaining_tier1_tier2.py - Dispatches the remaining 3 systems (wwdomain, bpti, barnase)
across Worker 2 and Worker 3, monitors them until completion, and downloads verified artifacts.
"""
import os
import sys
import json
import time
import requests

W2_URL = "https://brokers-park-further-auctions.trycloudflare.com"
W2_TOKEN = "nexus-v4-c6f6fc5541052848"

W3_URL = "https://orange-precipitation-placed-recorded.trycloudflare.com"
W3_TOKEN = "nexus-v4-b1a05a088567508b"

ROOT = "/Users/mac/molecular dynamics"
TRAJ_DIR = os.path.join(ROOT, "version4_godtier", "trajectories")
MAN_DIR = os.path.join(ROOT, "version4_godtier", "manifests", "runs")

SHA_PDB = {
    "wwdomain": ("wwdomain_2F21.pdb", "10b20f4681e2d34a4f9f7627ea6be07aaf4de3997d8486ddf2050e19775e871c"),
    "bpti": ("bpti_5PTI.pdb", "5dc08b34ec0765236e4552c6a2065667fff0d74f23007b146b7b29c39974c260"),
    "barnase": ("barnase_1BNI.pdb", "616d446839e94886c73cdd56ce269e72ad50abbb222df144628d0d9946a040da"),
}

JOBS_SPEC = [
    # Worker 3: wwdomain (3 reps)
    ("wwdomain", 1, W3_URL, W3_TOKEN, 52001),
    ("wwdomain", 2, W3_URL, W3_TOKEN, 52002),
    ("wwdomain", 3, W3_URL, W3_TOKEN, 52003),
    # Worker 3: bpti (3 reps)
    ("bpti", 1, W3_URL, W3_TOKEN, 62001),
    ("bpti", 2, W3_URL, W3_TOKEN, 62002),
    ("bpti", 3, W3_URL, W3_TOKEN, 62003),
    # Worker 2: barnase (3 reps)
    ("barnase", 1, W2_URL, W2_TOKEN, 72001),
    ("barnase", 2, W2_URL, W2_TOKEN, 72002),
    ("barnase", 3, W2_URL, W2_TOKEN, 72003),
]


def start_job(sys_k, rep, url, token, seed):
    headers = {"Authorization": f"Bearer {token}"}
    pdb_fn, pdb_sha = SHA_PDB[sys_k]
    job_id = f"{sys_k}_rep{rep}_c000_v4live"
    payload = {
        "job_id": job_id,
        "system_key": sys_k,
        "pdb_filename": pdb_fn,
        "pdb_sha256": pdb_sha,
        "solvent": "implicit_gbn2",
        "temperature_k": 300.0,
        "friction_ps": 1.0,
        "time_step_fs": 4.0,
        "hmr_amu": 1.5,
        "stride_ps": 1.0,
        "chunk_ns": 10.0,
        "chunk_index": 0,
        "replica_index": rep,
        "seed": seed,
        "platform": "CUDA",
        "skip_equil": False,
    }
    r = requests.post(f"{url}/start_chunk", headers=headers, json=payload, timeout=30)
    print(f"[{sys_k} rep{rep}] Dispatched -> {r.status_code}: {r.json()}", flush=True)
    return job_id


def download_file(url, token, job_id, artifact, dest_path):
    dest_abs = os.path.abspath(dest_path)
    os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
    tmp_path = dest_abs + f".tmp.{os.getpid()}"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"artifact": artifact}
    for attempt in range(5):
        try:
            with requests.get(
                f"{url}/download_chunk/{job_id}",
                params=params,
                headers=headers,
                stream=True,
                timeout=(30.0, 300.0),
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
            print(f"  [Retry {attempt+1}] Download {artifact} failed: {e}", flush=True)
            time.sleep(2.0)
    return False


def main():
    print("=" * 70, flush=True)
    print("DISPATCHING FINAL 3 PROTEINS (WW, BPTI, BARNASE) TO BLACKWELL WORKERS", flush=True)
    print("=" * 70, flush=True)

    active_jobs = {}
    for sys_k, rep, url, token, seed in JOBS_SPEC:
        jid = start_job(sys_k, rep, url, token, seed)
        active_jobs[jid] = (sys_k, rep, url, token)
        time.sleep(1.0)

    print(f"\nDispatched {len(active_jobs)} total production jobs. Monitoring...", flush=True)

    completed = set()
    while len(completed) < len(active_jobs):
        time.sleep(10.0)
        for jid, (sys_k, rep, url, token) in active_jobs.items():
            if jid in completed:
                continue
            headers = {"Authorization": f"Bearer {token}"}
            try:
                r = requests.get(f"{url}/chunk_status/{jid}", headers=headers, timeout=10)
                st = r.json()
                status = st.get("status")
                last_log = st.get("log_tail", [None])[-1] if st.get("log_tail") else "queued"
                if status == "COMPLETED":
                    print(f"\n[DONE] {jid} COMPLETED! Starting download...", flush=True)
                    out_dir = os.path.join(TRAJ_DIR, sys_k, "implicit_gbn2")
                    npz_dest = os.path.join(out_dir, f"{sys_k}_rep{rep}_chunk000.npz")
                    xml_dest = os.path.join(out_dir, f"{sys_k}_rep{rep}_chunk000_state.xml")
                    man_dest = os.path.join(MAN_DIR, f"{sys_k}_rep{rep}_chunk000_implicit_gbn2.json")

                    download_file(url, token, jid, "npz", npz_dest)
                    download_file(url, token, jid, "xml", xml_dest)
                    download_file(url, token, jid, "manifest", man_dest)
                    print(f"[VERIFIED] Saved {npz_dest} ({os.path.getsize(npz_dest)/(1024*1024):.1f} MB)", flush=True)
                    completed.add(jid)
                elif status == "FAILED":
                    print(f"\n[ERROR] {jid} FAILED: {st}", flush=True)
                    completed.add(jid)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] {jid}: {last_log}", flush=True)
            except Exception as e:
                print(f"  [Warning] polling {jid} error: {e}", flush=True)

    print("\nALL 9 JOBS PROCESSED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    main()
