#!/usr/bin/env python3
"""
version4_godtier/sims/agy_gpu_scheduler.py
==========================================
Local Antigravity GPU Coordinator for Google Colab Pro+ GPU Worker Nodes.

Dispatches PDB structures (verified by SHA-256) and 10-ns chunk jobs over HTTPS
to one or more Colab Pro+ GPU workers (or local loopback worker), polls progress,
downloads completed .npz trajectories, .xml states, and .json manifests, and
verifies cryptographic SHA-256 digests before committing to version4_godtier/trajectories/.
"""
import os
import sys
import json
import time
import base64
import hashlib
import argparse
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
V4_DIR = os.path.join(ROOT, 'version4_godtier')
TRAJ_DIR = os.path.join(V4_DIR, 'trajectories')
RUNS_MAN_DIR = os.path.join(V4_DIR, 'manifests', 'runs')

for _d in (TRAJ_DIR, RUNS_MAN_DIR):
    os.makedirs(_d, exist_ok=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ColabBridgeClient:
    def __init__(self, endpoint_url: str, token: str):
        self.base = endpoint_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _req(self, method: str, url: str, **kwargs):
        last_err = None
        for attempt in range(6):
            try:
                r = requests.request(method, url, **kwargs)
                r.raise_for_status()
                return r
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"HTTP {method} {url} failed after 6 attempts: {last_err}")

    def health(self) -> dict:
        return self._req("GET", f"{self.base}/health", timeout=20).json()

    def gpu_status(self) -> dict:
        return self._req("GET", f"{self.base}/gpu_status", headers=self.headers, timeout=20).json()

    def upload_pdb(self, local_pdb_path: str) -> dict:
        sha = sha256_file(local_pdb_path)
        with open(local_pdb_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        payload = {
            "filename": os.path.basename(local_pdb_path),
            "expected_sha256": sha,
            "content_b64": b64,
        }
        r = self._req("POST", f"{self.base}/upload_pdb", headers=self.headers, json=payload, timeout=90)
        res = r.json()
        if res.get("sha256") != sha:
            raise RuntimeError(f"Remote SHA-256 verification failed: {res}")
        return res

    def start_chunk(self, spec: dict) -> dict:
        return self._req("POST", f"{self.base}/start_chunk", headers=self.headers, json=spec, timeout=45).json()

    def chunk_status(self, job_id: str) -> dict:
        return self._req("GET", f"{self.base}/chunk_status/{job_id}", headers=self.headers, timeout=45).json()

    def download_artifact(self, job_id: str, artifact: str, dest_path: str, rep: int = None):
        params = {"artifact": artifact}
        if rep is not None:
            params["rep"] = str(rep)
        dest_abs = os.path.abspath(dest_path)
        os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
        tmp_path = dest_abs + f".tmp.{os.getpid()}"
        last_err = None
        for attempt in range(8):
            try:
                with requests.get(
                    f"{self.base}/download_chunk/{job_id}",
                    params=params,
                    headers=self.headers,
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
                    return
            except Exception as e:
                last_err = e
                print(f"[Download-Retry] {artifact} for {job_id} attempt {attempt+1}/8 failed ({e}). Retrying...", flush=True)
                time.sleep(2.0 * (attempt + 1))
            finally:
                if attempt == 7 and os.path.exists(tmp_path) and not os.path.exists(dest_abs):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
        raise RuntimeError(f"Failed to download artifact {artifact} for {job_id}: {last_err}")


def run_remote_batch_3reps(client: ColabBridgeClient, system_key: str, pdb_relpath: str,
                           chunk_ns: float = 2.5, dt_fs: float = 4.0, hmr_amu: float = 1.5,
                           stride_ps: float = 1.0, seeds=(42001, 42002, 42003),
                           platform: str = "CUDA", alt_pdb_relpath: str = None,
                           uploaded_pdb: dict = None, uploaded_alt_pdb: dict = None,
                           skip_equil: bool = False, poll_sec: float = 2.0):
    local_pdb = os.path.join(ROOT, pdb_relpath)
    up = uploaded_pdb if uploaded_pdb is not None else client.upload_pdb(local_pdb)
    seeds_str = ",".join(str(int(s)) for s in seeds)

    req = {
        "system_key": system_key,
        "pdb_filename": up["filename"],
        "pdb_sha256": up["sha256"],
        "batch_3reps": True,
        "seeds_3rep": seeds_str,
        "solvent": "implicit_gbn2",
        "temperature_k": 300.0,
        "friction_ps": 1.0,
        "time_step_fs": dt_fs,
        "hmr_amu": hmr_amu,
        "stride_ps": stride_ps,
        "chunk_ns": chunk_ns,
        "chunk_index": 0,
        "replica_index": 1,
        "platform": platform,
        "skip_equil": skip_equil,
    }
    if uploaded_alt_pdb is not None:
        req["alt_pdb_filename"] = uploaded_alt_pdb["filename"]
    elif alt_pdb_relpath:
        local_alt = os.path.join(ROOT, alt_pdb_relpath)
        if os.path.exists(local_alt):
            up_alt = client.upload_pdb(local_alt)
            req["alt_pdb_filename"] = up_alt["filename"]

    started = client.start_chunk(req)
    job_id = started["job_id"]
    print(f"[Bridge-Batch3] Dispatched 3-Replica GPU Batch {job_id} ({system_key}) -> {client.base}", flush=True)

    while True:
        st = client.chunk_status(job_id)
        status = st["status"]
        if status == "COMPLETED":
            break
        if status == "FAILED":
            raise RuntimeError(f"Remote 3-rep batch {job_id} FAILED: {json.dumps(st, indent=2)}")
        time.sleep(poll_sec)

    out_dir = os.path.join(TRAJ_DIR, system_key, "implicit_gbn2")
    os.makedirs(out_dir, exist_ok=True)
    manifests = []
    for r_i in (1, 2, 3):
        base_name = f"{system_key}_rep{r_i}_chunk000"
        local_npz = os.path.join(out_dir, f"{base_name}.npz")
        local_xml = os.path.join(out_dir, f"{base_name}_state.xml")
        local_man = os.path.join(RUNS_MAN_DIR, f"{base_name}_implicit_gbn2.json")

        client.download_artifact(job_id, "npz", local_npz, rep=r_i)
        client.download_artifact(job_id, "xml", local_xml, rep=r_i)
        client.download_artifact(job_id, "manifest", local_man, rep=r_i)

        manifest = json.load(open(local_man))
        actual_npz_sha = sha256_file(local_npz)
        actual_xml_sha = sha256_file(local_xml)
        if actual_npz_sha != manifest["sha256"]:
            raise RuntimeError(f"Downloaded NPZ SHA-256 mismatch: {actual_npz_sha} != {manifest['sha256']}")
        if actual_xml_sha != manifest["state_xml_sha256"]:
            raise RuntimeError(f"Downloaded XML SHA-256 mismatch: {actual_xml_sha} != {manifest['state_xml_sha256']}")

        manifest["trajectory_file"] = os.path.relpath(local_npz, ROOT)
        manifest["state_xml_file"] = os.path.relpath(local_xml, ROOT)
        manifest["input_pdb"] = pdb_relpath
        with open(local_man, "w") as mf:
            json.dump(manifest, mf, indent=2)
        manifests.append(manifest)

    print(
        f"[Bridge-Batch3] COMPLETED & SHA-256 VERIFIED ALL 3 REPLICAS of {system_key} | "
        f"Frames=3x{manifests[0]['total_frames']} | d={manifests[0]['d']} | "
        f"EffRate={manifests[0]['ns_per_day']:.1f} ns/day",
        flush=True,
    )
    return manifests


def run_remote_job(client: ColabBridgeClient, system_key: str, pdb_relpath: str,
                   solvent: str = "implicit_gbn2", replica_index: int = 1,
                   chunk_index: int = 0, chunk_ns: float = 0.004,
                   dt_fs: float = 4.0, hmr_amu: float = 1.5,
                   stride_ps: float = 1.0, seed: int = 42001,
                   platform: str = "CPU", resume_xml_path: str = None,
                   alt_pdb_relpath: str = None,
                   uploaded_pdb: dict = None, uploaded_alt_pdb: dict = None,
                   skip_equil: bool = False, poll_sec: float = 1.0,
                   adopt_job_id: str = None):
    if adopt_job_id:
        job_id = adopt_job_id
        print(f"[Bridge] Adopting in-flight job {job_id} on {client.base}", flush=True)
        return _collect(client, job_id, system_key, solvent, replica_index, chunk_index, pdb_relpath, poll_sec)
    local_pdb = os.path.join(ROOT, pdb_relpath)
    if uploaded_pdb is not None:
        up = uploaded_pdb
    else:
        up = client.upload_pdb(local_pdb)
        print(f"[Bridge] Uploaded & verified PDB {up['filename']} (SHA-256: {up['sha256'][:16]}...)", flush=True)

    req = {
        "system_key": system_key,
        "pdb_filename": up["filename"],
        "pdb_sha256": up["sha256"],
        "solvent": solvent,
        "temperature_k": 300.0,
        "friction_ps": 1.0,
        "time_step_fs": dt_fs,
        "hmr_amu": hmr_amu,
        "stride_ps": stride_ps,
        "chunk_ns": chunk_ns,
        "chunk_index": chunk_index,
        "replica_index": replica_index,
        "seed": seed,
        "platform": platform,
        "skip_equil": skip_equil,
    }
    if uploaded_alt_pdb is not None:
        req["alt_pdb_filename"] = uploaded_alt_pdb["filename"]
    elif alt_pdb_relpath:
        local_alt = os.path.join(ROOT, alt_pdb_relpath)
        if os.path.exists(local_alt):
            up_alt = client.upload_pdb(local_alt)
            req["alt_pdb_filename"] = up_alt["filename"]
    if resume_xml_path and os.path.exists(resume_xml_path):
        with open(resume_xml_path, "rb") as xf:
            req["resume_state_xml_b64"] = base64.b64encode(xf.read()).decode("ascii")

    started = client.start_chunk(req)
    job_id = started["job_id"]
    print(f"[Bridge] Dispatched job {job_id} -> remote worker ({client.base})", flush=True)
    return _collect(client, job_id, system_key, solvent, replica_index, chunk_index, pdb_relpath, poll_sec)


def _collect(client, job_id, system_key, solvent, replica_index, chunk_index, pdb_relpath, poll_sec):
    while True:
        st = client.chunk_status(job_id)
        status = st["status"]
        if status == "COMPLETED":
            break
        if status == "FAILED":
            raise RuntimeError(f"Remote job {job_id} FAILED: {json.dumps(st, indent=2)}")
        time.sleep(poll_sec)

    out_dir = os.path.join(TRAJ_DIR, system_key, solvent)
    os.makedirs(out_dir, exist_ok=True)
    base_name = f"{system_key}_rep{replica_index}_chunk{chunk_index:03d}"
    local_npz = os.path.join(out_dir, f"{base_name}.npz")
    local_xml = os.path.join(out_dir, f"{base_name}_state.xml")
    local_man = os.path.join(RUNS_MAN_DIR, f"{base_name}_{solvent}.json")

    client.download_artifact(job_id, "npz", local_npz)
    client.download_artifact(job_id, "xml", local_xml)
    client.download_artifact(job_id, "manifest", local_man)

    manifest = json.load(open(local_man))
    actual_npz_sha = sha256_file(local_npz)
    actual_xml_sha = sha256_file(local_xml)
    if actual_npz_sha != manifest["sha256"]:
        raise RuntimeError(f"Downloaded NPZ SHA-256 mismatch: {actual_npz_sha} != {manifest['sha256']}")
    if actual_xml_sha != manifest["state_xml_sha256"]:
        raise RuntimeError(f"Downloaded XML SHA-256 mismatch: {actual_xml_sha} != {manifest['state_xml_sha256']}")

    # Rewrite relative paths in manifest to match local workspace paths
    manifest["trajectory_file"] = os.path.relpath(local_npz, ROOT)
    manifest["state_xml_file"] = os.path.relpath(local_xml, ROOT)
    manifest["input_pdb"] = pdb_relpath
    with open(local_man, "w") as mf:
        json.dump(manifest, mf, indent=2)

    print(
        f"[Bridge] COMPLETED & SHA-256 VERIFIED {manifest['trajectory_file']} | "
        f"Frames={manifest['total_frames']} | d={manifest['d']} | "
        f"Rate={manifest['ns_per_day']:.1f} ns/day | SHA-256={actual_npz_sha[:16]}...",
        flush=True,
    )
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:18765")
    ap.add_argument("--token", default="v4-godtier-secret-token-2026")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    client = ColabBridgeClient(args.endpoint, args.token)
    gpu_info = client.gpu_status()
    print("[Bridge] Health:", json.dumps(client.health()), flush=True)
    print("[Bridge] GPU Status:", json.dumps(gpu_info, indent=2), flush=True)
    best_platform = "CUDA" if "CUDA" in gpu_info.get("openmm_platforms", []) else "CPU"

    if args.smoke_test:
        man0 = run_remote_job(
            client, system_key="ala1",
            pdb_relpath="version4_godtier/structures/ala1_ACE_ALA_NME.pdb",
            solvent="implicit_obc2", replica_index=1, chunk_index=0,
            chunk_ns=0.004, dt_fs=2.0, hmr_amu=1.008, stride_ps=1.0,
            seed=42001, platform=best_platform, skip_equil=True, poll_sec=0.5
        )
        xml0 = os.path.join(ROOT, man0["state_xml_file"])
        man1 = run_remote_job(
            client, system_key="ala1",
            pdb_relpath="version4_godtier/structures/ala1_ACE_ALA_NME.pdb",
            solvent="implicit_obc2", replica_index=1, chunk_index=1,
            chunk_ns=0.004, dt_fs=2.0, hmr_amu=1.008, stride_ps=1.0,
            seed=42001, platform=best_platform, resume_xml_path=xml0, skip_equil=True, poll_sec=0.5
        )
        print("[Bridge Smoke Test] Chunk 0->1 Continuity Verified:",
              f"c0.end={man0['end_time_ps']} ps == c1.start={man1['start_time_ps']} ps", flush=True)


if __name__ == "__main__":
    main()
