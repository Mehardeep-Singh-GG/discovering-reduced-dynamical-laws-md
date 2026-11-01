#!/usr/bin/env python3
"""
version4_godtier/sims/colab_worker_bootstrap.py
===============================================
Self-contained Zero-Dependency HTTP/REST GPU Worker Daemon for Google Colab Pro+ (A100 / L4 / T4)
and local loopback verification.

Implements the exact REST protocol specified in M29:
  GET  /health
  GET  /gpu_status
  POST /upload_pdb
  POST /start_chunk
  GET  /chunk_status/<job_id>
  GET  /download_chunk/<job_id>?artifact=npz|xml|manifest
"""
import os
import re
import sys
import json
import time
import uuid
import base64
import shutil
import hashlib
import urllib.parse
import argparse
import threading
import subprocess
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional

WORKDIR = os.path.abspath("./colab_v4_workspace")
STRUCT_DIR = os.path.join(WORKDIR, "structures")
JOBS_DIR = os.path.join(WORKDIR, "jobs")
for _d in (WORKDIR, STRUCT_DIR, JOBS_DIR):
    os.makedirs(_d, exist_ok=True)

API_TOKEN = os.environ.get("AGY_BRIDGE_TOKEN", "v4-godtier-secret-token-2026")
JOBS_LOCK = threading.Lock()
JOBS: Dict[str, Dict[str, Any]] = {}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_gpu_status() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "openmm_platforms": [],
        "nvidia_smi": None,
        "disk_free_gb": round(shutil.disk_usage(WORKDIR).free / (1024**3), 2),
    }
    try:
        import openmm
        info["openmm_version"] = openmm.__version__
        info["openmm_platforms"] = [
            openmm.Platform.getPlatform(i).getName()
            for i in range(openmm.Platform.getNumPlatforms())
        ]
    except Exception as e:
        info["openmm_error"] = str(e)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5, stderr=subprocess.DEVNULL
        ).strip()
        info["nvidia_smi"] = out
    except Exception:
        info["nvidia_smi"] = "No NVIDIA GPU (Local OpenCL/CPU mode)"
    return info


def _run_chunk_subprocess(job_id: str, req: Dict[str, Any]):
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    pdb_path = os.path.join(STRUCT_DIR, req["pdb_filename"])
    sys_key = req["system_key"]
    rep_idx = int(req.get("replica_index", 1))
    chk_idx = int(req.get("chunk_index", 0))
    batch_3reps = bool(req.get("batch_3reps", False))

    out_npz = os.path.join(job_dir, f"{sys_key}_rep{rep_idx}_chunk{chk_idx:03d}.npz")
    out_xml = os.path.join(job_dir, f"{sys_key}_rep{rep_idx}_chunk{chk_idx:03d}_state.xml")
    out_man = os.path.join(job_dir, f"{sys_key}_rep{rep_idx}_chunk{chk_idx:03d}_manifest.json")
    log_path = os.path.join(job_dir, "worker.log")

    resume_xml_path = None
    if req.get("resume_state_xml_b64"):
        resume_xml_path = os.path.join(job_dir, "input_resume_state.xml")
        with open(resume_xml_path, "wb") as f:
            f.write(base64.b64decode(req["resume_state_xml_b64"]))

    engine_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "md_engine_v4_gpu.py")
    if batch_3reps:
        seeds_str = req.get("seeds_3rep", "42001,42002,42003")
        cmd = [
            sys.executable, engine_script,
            "--pdb", pdb_path,
            "--batch-3reps",
            "--out-dir-traj", job_dir,
            "--out-dir-man", job_dir,
            "--seeds-3rep", str(seeds_str),
            "--system-key", sys_key,
            "--solvent", "implicit_gbn2",
            "--temperature-k", str(req.get("temperature_k", 300.0)),
            "--friction-ps", str(req.get("friction_ps", 1.0)),
            "--dt-fs", str(req.get("time_step_fs", 4.0)),
            "--hmr-amu", str(req.get("hmr_amu", 1.5)),
            "--stride-ps", str(req.get("stride_ps", 1.0)),
            "--ns", str(req.get("chunk_ns", 6.0)),
            "--chunk-index", str(chk_idx),
            "--platform", str(req.get("platform", "CUDA")),
        ]
    else:
        cmd = [
            sys.executable, engine_script,
            "--pdb", pdb_path,
            "--out-npz", out_npz,
            "--out-xml", out_xml,
            "--out-manifest", out_man,
            "--system-key", sys_key,
            "--solvent", str(req.get("solvent", "implicit_gbn2")),
            "--temperature-k", str(req.get("temperature_k", 300.0)),
            "--friction-ps", str(req.get("friction_ps", 1.0)),
            "--dt-fs", str(req.get("time_step_fs", 4.0)),
            "--hmr-amu", str(req.get("hmr_amu", 1.5)),
            "--stride-ps", str(req.get("stride_ps", 1.0)),
            "--ns", str(req.get("chunk_ns", 10.0)),
            "--chunk-index", str(chk_idx),
            "--replica-index", str(rep_idx),
            "--seed", str(req.get("seed", 42001)),
            "--platform", str(req.get("platform", "CUDA")),
            "--cache-dir", os.path.join(WORKDIR, "cache", f"{sys_key}_{req.get('solvent','implicit_gbn2')}_rep{rep_idx}"),
        ]
    if resume_xml_path:
        cmd.extend(["--resume-xml", resume_xml_path])
    if req.get("alt_pdb_filename"):
        cmd.extend(["--alt-pdb", os.path.join(STRUCT_DIR, req["alt_pdb_filename"])])
    if req.get("skip_equil"):
        cmd.append("--skip-equil")

    with JOBS_LOCK:
        JOBS[job_id]["status"] = "RUNNING"
        JOBS[job_id]["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    try:
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
            with JOBS_LOCK:
                JOBS[job_id]["pid"] = proc.pid
            rc = proc.wait()
        if batch_3reps and rc == 0:
            rep_artifacts = {}
            for r_i in (1, 2, 3):
                bname = f"{sys_key}_rep{r_i}_chunk{chk_idx:03d}"
                r_npz = os.path.join(job_dir, f"{bname}.npz")
                r_xml = os.path.join(job_dir, f"{bname}_state.xml")
                r_man = os.path.join(job_dir, f"{bname}_implicit_gbn2.json")
                if os.path.exists(r_npz) and os.path.exists(r_man):
                    rep_artifacts[str(r_i)] = {
                        "npz_path": r_npz,
                        "xml_path": r_xml,
                        "manifest": json.load(open(r_man)),
                    }
            if len(rep_artifacts) == 3:
                with JOBS_LOCK:
                    JOBS[job_id]["status"] = "COMPLETED"
                    JOBS[job_id]["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    JOBS[job_id]["rep_artifacts"] = rep_artifacts
            else:
                with JOBS_LOCK:
                    JOBS[job_id]["status"] = "FAILED"
                    JOBS[job_id]["return_code"] = rc
        elif rc == 0 and os.path.exists(out_npz) and os.path.exists(out_man):
            with open(out_man, "r") as mf:
                manifest_data = json.load(mf)
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "COMPLETED"
                JOBS[job_id]["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                JOBS[job_id]["npz_path"] = out_npz
                JOBS[job_id]["xml_path"] = out_xml
                JOBS[job_id]["manifest"] = manifest_data
        else:
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "FAILED"
                JOBS[job_id]["return_code"] = rc
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "FAILED"
            JOBS[job_id]["error"] = str(e)


class BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return  # quiet standard logging

    def _check_auth(self) -> bool:
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth.split(" ", 1)[1].strip() != API_TOKEN:
            self._send_json(401, {"error": "Unauthorized"})
            return False
        return True

    def _send_json(self, code: int, payload: Dict[str, Any]):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, filepath: str):
        with open(filepath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(filepath)}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/health":
            return self._send_json(200, {"status": "ok", "service": "colab_worker_bootstrap_v4"})

        if not self._check_auth():
            return

        if path == "/gpu_status":
            return self._send_json(200, probe_gpu_status())

        if path.startswith("/chunk_status/"):
            job_id = path.split("/chunk_status/", 1)[1].strip("/")
            with JOBS_LOCK:
                if job_id not in JOBS:
                    return self._send_json(404, {"error": f"Unknown job_id {job_id}"})
                info = dict(JOBS[job_id])
            log_path = os.path.join(JOBS_DIR, job_id, "worker.log")
            if os.path.exists(log_path):
                with open(log_path, "r") as lf:
                    info["log_tail"] = [l.rstrip() for l in lf.readlines()[-10:]]
            return self._send_json(200, info)

        if path.startswith("/download_chunk/"):
            job_id = path.split("/download_chunk/", 1)[1].strip("/")
            artifact = qs.get("artifact", ["npz"])[0]
            rep_q = qs.get("rep", [None])[0]
            with JOBS_LOCK:
                if job_id not in JOBS:
                    return self._send_json(404, {"error": f"Unknown job_id {job_id}"})
                job = dict(JOBS[job_id])
            if job.get("status") != "COMPLETED":
                return self._send_json(409, {"error": f"Job {job_id} in state {job.get('status')}"})
            if rep_q and "rep_artifacts" in job:
                ra = job["rep_artifacts"][str(rep_q)]
                if artifact == "npz":
                    return self._send_file(ra["npz_path"])
                elif artifact == "xml":
                    return self._send_file(ra["xml_path"])
                elif artifact == "manifest":
                    return self._send_json(200, ra["manifest"])
            if artifact == "npz":
                return self._send_file(job["npz_path"])
            elif artifact == "xml":
                return self._send_file(job["xml_path"])
            elif artifact == "manifest":
                return self._send_json(200, job["manifest"])
            else:
                return self._send_json(400, {"error": f"Unknown artifact {artifact}"})

        return self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if not self._check_auth():
            return
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

        if path == "/upload_pdb":
            filename = os.path.basename(body["filename"])
            expected_sha = body["expected_sha256"]
            content = base64.b64decode(body["content_b64"])
            dest = os.path.join(STRUCT_DIR, filename)
            with open(dest, "wb") as f:
                f.write(content)
            actual_sha = sha256_file(dest)
            if actual_sha != expected_sha:
                os.remove(dest)
                return self._send_json(400, {"error": f"SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"})
            return self._send_json(200, {"status": "verified", "filename": filename, "sha256": actual_sha})

        if path == "/start_chunk":
            pdb_path = os.path.join(STRUCT_DIR, body["pdb_filename"])
            if not os.path.exists(pdb_path):
                return self._send_json(404, {"error": f"PDB {body['pdb_filename']} not uploaded"})
            if sha256_file(pdb_path) != body["pdb_sha256"]:
                return self._send_json(400, {"error": "Stored PDB SHA-256 mismatch"})

            sys_key = body["system_key"]
            rep_idx = int(body.get("replica_index", 1))
            chk_idx = int(body.get("chunk_index", 0))
            job_id = body.get("job_id") or f"{sys_key}_rep{rep_idx}_c{chk_idx:03d}_{uuid.uuid4().hex[:6]}"
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "job_id": job_id,
                    "request": body,
                    "status": "QUEUED",
                    "queued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            t = threading.Thread(target=_run_chunk_subprocess, args=(job_id, body), daemon=True)
            t.start()
            return self._send_json(200, {"job_id": job_id, "status": "QUEUED"})

        return self._send_json(404, {"error": "Not found"})


def launch_cloudflared_tunnel(port: int) -> Optional[str]:
    cf_bin = shutil.which("cloudflared")
    if not cf_bin and sys.platform.startswith("linux"):
        cf_dest = "/tmp/cloudflared"
        if not os.path.exists(cf_dest):
            print("[Tunnel] Downloading cloudflared-linux-amd64...", flush=True)
            subprocess.run(
                [
                    "curl", "-fsSL",
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                    "-o", cf_dest,
                ],
                check=True,
            )
            os.chmod(cf_dest, 0o755)
        cf_bin = cf_dest
    if not cf_bin:
        return None
    proc = subprocess.Popen(
        [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    pattern = re.compile(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)")
    found_url = []

    def _drain():
        for line in proc.stdout:
            if not found_url:
                m = pattern.search(line)
                if m:
                    found_url.append(m.group(1))

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    t0 = time.time()
    while time.time() - t0 < 30:
        if found_url:
            return found_url[0]
        time.sleep(0.2)
    return None


def main():
    global API_TOKEN, WORKDIR, STRUCT_DIR, JOBS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--token", default=API_TOKEN)
    ap.add_argument("--workdir", default=WORKDIR)
    ap.add_argument("--tunnel", action="store_true", help="Launch cloudflared tunnel if installed")
    args = ap.parse_args()
    API_TOKEN = args.token
    WORKDIR = os.path.abspath(args.workdir)
    STRUCT_DIR = os.path.join(WORKDIR, "structures")
    JOBS_DIR = os.path.join(WORKDIR, "jobs")
    for _d in (WORKDIR, STRUCT_DIR, JOBS_DIR):
        os.makedirs(_d, exist_ok=True)

    if sys.platform.startswith("linux") and shutil.which("nvidia-cuda-mps-control"):
        try:
            subprocess.run(["nvidia-cuda-mps-control", "-d"], check=False, timeout=5)
            print("[MPS] Enabled NVIDIA Multi-Process Service (MPS) hardware multiplexer.", flush=True)
        except Exception:
            pass

    server = ThreadingHTTPServer((args.host, args.port), BridgeHandler)
    public_url = f"http://{args.host}:{args.port}"
    if args.tunnel:
        cf_url = launch_cloudflared_tunnel(args.port)
        if cf_url:
            public_url = cf_url

    gpu_info = probe_gpu_status()
    print("=" * 76, flush=True)
    print(f"ANTIGRAVITY_COLAB_ENDPOINT = {public_url}", flush=True)
    print(f"ANTIGRAVITY_COLAB_TOKEN    = {API_TOKEN}", flush=True)
    print(f"OPENMM_PLATFORMS           = {gpu_info.get('openmm_platforms')}", flush=True)
    print(f"NVIDIA_GPU                 = {gpu_info.get('nvidia_smi')}", flush=True)
    print("=" * 76, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
