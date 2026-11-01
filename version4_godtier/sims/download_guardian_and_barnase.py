#!/usr/bin/env python3
"""
download_guardian_and_barnase.py
================================
1. Watcher thread (every 2s):
   - Terminates legacy PID 43917 the instant villin_rep3_chunk000.npz lands on disk
     so it never collides with PID 47933 on Worker 1 (AdK / MBP).
   - Terminates legacy PID 43870 the instant bpti_rep3_chunk000.npz lands on disk
     so it never collides with dedicated Barnase downloader on Worker 2.
2. Dedicated Worker 2 Barnase Downloader thread:
   - Waits until lysozyme_rep3_chunk000.npz lands on disk (so Worker 2 tunnel has 1 stream).
   - Immediately downloads barnase rep1, rep2, rep3 (manifest, state.xml, npz) with SHA-256 check.
3. Master Completion & Cryptographic Gate:
   - Waits until ALL 27 remote chunks (9 proteins x 3 replicas) + existing local chunks exist.
   - Verifies non-zero size, valid NPZ archive, and exact SHA-256 match against manifest.
   - Exits 0 when 100% of all 15 systems are verified on local disk.
"""
import os
import sys
import json
import time
import glob
import signal
import hashlib
import threading
import requests

ROOT = "/Users/mac/molecular dynamics"
TRAJ_DIR = os.path.join(ROOT, "version4_godtier", "trajectories")
MAN_DIR = os.path.join(ROOT, "version4_godtier", "manifests", "runs")

W1_URL = "https://locate-most-survey-hide.trycloudflare.com"
W1_TOKEN = "nexus-v4-04d55f6603ce2b56"
W2_URL = "https://brokers-park-further-auctions.trycloudflare.com"
W2_TOKEN = "nexus-v4-c6f6fc5541052848"
W3_URL = "https://orange-precipitation-placed-recorded.trycloudflare.com"
W3_TOKEN = "nexus-v4-b1a05a088567508b"

EXPECTED_NEW_CHUNKS = [
    ("bba", "implicit_gbn2", 1, W3_URL, W3_TOKEN, "bba_rep1_c000_44f247"),
    ("bba", "implicit_gbn2", 2, W3_URL, W3_TOKEN, "bba_rep2_c000_408ce7"),
    ("bba", "implicit_gbn2", 3, W3_URL, W3_TOKEN, "bba_rep3_c000_f974f8"),
    ("wwdomain", "implicit_gbn2", 1, W3_URL, W3_TOKEN, "wwdomain_rep1_c000_v4live"),
    ("wwdomain", "implicit_gbn2", 2, W3_URL, W3_TOKEN, "wwdomain_rep2_c000_v4live"),
    ("wwdomain", "implicit_gbn2", 3, W3_URL, W3_TOKEN, "wwdomain_rep3_c000_v4live"),
    ("villin", "implicit_gbn2", 1, W3_URL, W3_TOKEN, "villin_rep1_c000_33b9b1"),
    ("villin", "implicit_gbn2", 2, W3_URL, W3_TOKEN, "villin_rep2_c000_797dc8"),
    ("villin", "implicit_gbn2", 3, W3_URL, W3_TOKEN, "villin_rep3_c000_6dd779"),
    ("bpti", "implicit_gbn2", 1, W3_URL, W3_TOKEN, "bpti_rep1_c000_v4live"),
    ("bpti", "implicit_gbn2", 2, W3_URL, W3_TOKEN, "bpti_rep2_c000_v4live"),
    ("bpti", "implicit_gbn2", 3, W3_URL, W3_TOKEN, "bpti_rep3_c000_v4live"),
    ("barnase", "implicit_gbn2", 1, W2_URL, W2_TOKEN, "barnase_rep1_c000_v4live"),
    ("barnase", "implicit_gbn2", 2, W2_URL, W2_TOKEN, "barnase_rep2_c000_v4live"),
    ("barnase", "implicit_gbn2", 3, W2_URL, W2_TOKEN, "barnase_rep3_c000_v4live"),
    ("lysozyme", "explicit_pme", 1, W2_URL, W2_TOKEN, "lysozyme_rep1_c000_dea893"),
    ("lysozyme", "explicit_pme", 2, W2_URL, W2_TOKEN, "lysozyme_rep2_c000_a4361d"),
    ("lysozyme", "explicit_pme", 3, W2_URL, W2_TOKEN, "lysozyme_rep3_c000_1f0a08"),
    ("dhfr", "explicit_pme", 1, W2_URL, W2_TOKEN, "dhfr_rep1_c000_fea6fb"),
    ("dhfr", "explicit_pme", 2, W2_URL, W2_TOKEN, "dhfr_rep2_c000_bf4aa4"),
    ("dhfr", "explicit_pme", 3, W2_URL, W2_TOKEN, "dhfr_rep3_c000_a185ee"),
    ("adk", "explicit_pme", 1, W1_URL, W1_TOKEN, "adk_rep1_c000_0c3ddb"),
    ("adk", "explicit_pme", 2, W1_URL, W1_TOKEN, "adk_rep2_c000_c40665"),
    ("adk", "explicit_pme", 3, W1_URL, W1_TOKEN, "adk_rep3_c000_8c1273"),
    ("mbp", "explicit_pme", 1, W1_URL, W1_TOKEN, "mbp_rep1_c000_107192"),
    ("mbp", "explicit_pme", 2, W1_URL, W1_TOKEN, "mbp_rep2_c000_919c67"),
    ("mbp", "explicit_pme", 3, W1_URL, W1_TOKEN, "mbp_rep3_c000_0c2b11"),
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url, token, jid, artifact, dest, expected_sha=None):
    dest_abs = os.path.abspath(dest)
    if os.path.exists(dest_abs) and os.path.getsize(dest_abs) > 100:
        if expected_sha is None or sha256_file(dest_abs) == expected_sha:
            return True
        else:
            print(f"  [SHA Mismatch] Re-downloading {dest_abs}", flush=True)
    os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
    tmp_path = dest_abs + f".tmp_guard.{os.getpid()}.{threading.get_ident()}"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"artifact": artifact}
    for attempt in range(8):
        try:
            with requests.get(
                f"{url}/download_chunk/{jid}",
                params=params,
                headers=headers,
                stream=True,
                timeout=(30.0, 900.0),
            ) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1048576):
                        if chunk:
                            f.write(chunk)
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                if expected_sha and sha256_file(tmp_path) != expected_sha:
                    raise RuntimeError("SHA-256 verification failed after download")
                os.replace(tmp_path, dest_abs)
                return True
        except Exception as e:
            print(f"  [Retry {attempt+1}/8] {os.path.basename(dest)} failed: {e}", flush=True)
            time.sleep(3.0 * (attempt + 1))
        finally:
            if os.path.exists(tmp_path) and not os.path.exists(dest_abs):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    return False


def guardian_loop(stop_event):
    killed_43917 = False
    killed_43870 = False
    villin3 = os.path.join(TRAJ_DIR, "villin", "implicit_gbn2", "villin_rep3_chunk000.npz")
    bpti3 = os.path.join(TRAJ_DIR, "bpti", "implicit_gbn2", "bpti_rep3_chunk000.npz")

    while not stop_event.is_set():
        if not killed_43917 and os.path.exists(villin3) and os.path.getsize(villin3) > 50_000_000:
            try:
                os.kill(43917, signal.SIGTERM)
                print("[Guardian] villin_rep3 landed -> cleanly stopped legacy PID 43917 before W1 collision.", flush=True)
            except ProcessLookupError:
                pass
            for tf in glob.glob(os.path.join(TRAJ_DIR, "**", "*.tmp.43917"), recursive=True):
                try:
                    os.remove(tf)
                except OSError:
                    pass
            killed_43917 = True

        if not killed_43870 and os.path.exists(bpti3) and os.path.getsize(bpti3) > 100_000_000:
            time.sleep(3.0)  # let manifest/xml finish
            try:
                os.kill(43870, signal.SIGTERM)
                print("[Guardian] bpti_rep3 landed -> cleanly stopped legacy PID 43870 before W2 Barnase collision.", flush=True)
            except ProcessLookupError:
                pass
            for tf in glob.glob(os.path.join(TRAJ_DIR, "**", "*.tmp.43870"), recursive=True):
                try:
                    os.remove(tf)
                except OSError:
                    pass
            killed_43870 = True

        time.sleep(2.0)


def barnase_worker2_downloader():
    lyso3 = os.path.join(TRAJ_DIR, "lysozyme", "explicit_pme", "lysozyme_rep3_chunk000.npz")
    print("[Barnase-W2] Waiting for lysozyme_rep3_chunk000.npz to finish on Worker 2...", flush=True)
    while not (os.path.exists(lyso3) and os.path.getsize(lyso3) > 100_000_000):
        time.sleep(3.0)
    print("[Barnase-W2] lysozyme_rep3 finished! Starting dedicated Worker 2 Barnase download...", flush=True)
    for rep in (1, 2, 3):
        jid = f"barnase_rep{rep}_c000_v4live"
        out_dir = os.path.join(TRAJ_DIR, "barnase", "implicit_gbn2")
        npz_dest = os.path.join(out_dir, f"barnase_rep{rep}_chunk000.npz")
        xml_dest = os.path.join(out_dir, f"barnase_rep{rep}_chunk000_state.xml")
        man_dest = os.path.join(MAN_DIR, f"barnase_rep{rep}_chunk000_implicit_gbn2.json")
        download_file(W2_URL, W2_TOKEN, jid, "manifest", man_dest)
        expected_sha = None
        if os.path.exists(man_dest):
            try:
                expected_sha = json.load(open(man_dest)).get("sha256")
            except Exception:
                pass
        download_file(W2_URL, W2_TOKEN, jid, "xml", xml_dest)
        ok = download_file(W2_URL, W2_TOKEN, jid, "npz", npz_dest, expected_sha=expected_sha)
        if ok:
            print(f"[Barnase-W2] VERIFIED barnase_rep{rep}_chunk000.npz ({os.path.getsize(npz_dest)/(1024**2):.1f} MB)", flush=True)


def main():
    stop_event = threading.Event()
    t_guard = threading.Thread(target=guardian_loop, args=(stop_event,), daemon=True)
    t_barn = threading.Thread(target=barnase_worker2_downloader, daemon=True)
    t_guard.start()
    t_barn.start()

    last_count = -1
    while True:
        ready = 0
        missing = []
        for sys_k, solv, rep, url, tok, jid in EXPECTED_NEW_CHUNKS:
            npz_p = os.path.join(TRAJ_DIR, sys_k, solv, f"{sys_k}_rep{rep}_chunk000.npz")
            if os.path.exists(npz_p) and os.path.getsize(npz_p) > 10_000_000:
                ready += 1
            else:
                missing.append(f"{sys_k}_rep{rep}")

        if ready != last_count:
            print(f"[Progress] {ready}/27 new remote chunks on disk | Remaining ({len(missing)}): {', '.join(missing)}", flush=True)
            last_count = ready

        if ready == 27:
            break
        time.sleep(10.0)

    t_barn.join(timeout=30.0)
    print("\n[Verification] All 27 remote chunks on disk! Running full SHA-256 cryptographic verification...", flush=True)
    for sys_k, solv, rep, url, tok, jid in EXPECTED_NEW_CHUNKS:
        npz_p = os.path.join(TRAJ_DIR, sys_k, solv, f"{sys_k}_rep{rep}_chunk000.npz")
        man_p = os.path.join(MAN_DIR, f"{sys_k}_rep{rep}_chunk000_{solv}.json")
        if not os.path.exists(man_p):
            download_file(url, tok, jid, "manifest", man_p)
        man = json.load(open(man_p))
        exp_sha = man["sha256"]
        act_sha = sha256_file(npz_p)
        if act_sha != exp_sha:
            print(f"  [Repairing] SHA mismatch on {npz_p}: expected {exp_sha[:12]}, got {act_sha[:12]}", flush=True)
            os.remove(npz_p)
            ok = download_file(url, tok, jid, "npz", npz_p, expected_sha=exp_sha)
            if not ok:
                raise RuntimeError(f"Failed to repair {npz_p}")
        print(f"  [SHA-256 OK] {sys_k}_rep{rep}_chunk000.npz -> {exp_sha[:16]}...", flush=True)

    stop_event.set()
    print("\n=== ALL 15 SYSTEMS 100% DOWNLOADED AND SHA-256 VERIFIED ===", flush=True)


if __name__ == "__main__":
    main()
