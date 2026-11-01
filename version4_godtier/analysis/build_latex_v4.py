#!/usr/bin/env python3
r"""
build_latex_v4.py
=================
Automated LaTeX compilation and strict visual/log QA gate for Milestone 42:
1. Compiles MANUSCRIPT_GENERAL_V4.tex -> MANUSCRIPT_GENERAL_V4.pdf (and copies to manuscripts/)
2. Compiles SI_APPENDIX_V4.tex -> SI_APPENDIX_V4.pdf (and copies to manuscripts/)
3. Compiles RESPONSE_TO_EDITOR_LIU.tex -> manuscripts/RESPONSE_TO_EDITOR_LIU.pdf (if present)
4. Enforces strict log gates:
   - Zero LaTeX errors / undefined control sequences / undefined references
   - Reports any Overfull \hbox > 2.0pt and fails if --strict-hbox is passed
5. Renders every compiled PDF page to 200 DPI PNGs in version4_godtier/figures/qa_pages/
   using PyMuPDF (`pymupdf`) for page-by-page visual QA inspection.
"""
import os
import re
import sys
import shutil
import argparse
import subprocess
import pymupdf

ROOT = "/Users/mac/molecular dynamics"
QA_DIR = os.path.join(ROOT, "version4_godtier", "figures", "qa_pages")
TECTONIC = "/opt/homebrew/bin/tectonic"


def find_compiler():
    if os.path.exists(TECTONIC):
        return ("tectonic", TECTONIC)
    for p in ("/Library/TeX/texbin/pdflatex", "/opt/homebrew/bin/pdflatex", "/usr/local/bin/pdflatex"):
        if os.path.exists(p):
            return ("pdflatex", p)
    res = shutil.which("tectonic")
    if res:
        return ("tectonic", res)
    res = shutil.which("pdflatex")
    if res:
        return ("pdflatex", res)
    raise RuntimeError("No LaTeX compiler (tectonic or pdflatex) found on system!")


def compile_tex(tex_path: str, strict_hbox: bool = True) -> dict:
    tex_abs = os.path.abspath(tex_path)
    work_dir = os.path.dirname(tex_abs)
    stem = os.path.splitext(os.path.basename(tex_abs))[0]
    kind, bin_path = find_compiler()

    print(f"\n[LaTeX Build] Compiling {os.path.relpath(tex_abs, ROOT)} via {kind} ({bin_path})...", flush=True)
    if kind == "tectonic":
        cmd = [bin_path, "-X", "compile", tex_abs, "--keep-logs", "--outdir", work_dir]
        proc = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    else:
        cmd = [bin_path, "-interaction=nonstopmode", "-halt-on-error", os.path.basename(tex_abs)]
        subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
        proc = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)

    log_path = os.path.join(work_dir, f"{stem}.log")
    log_text = ""
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            log_text = f.read()
    combined_out = (proc.stdout or "") + "\n" + (proc.stderr or "") + "\n" + log_text

    if proc.returncode != 0:
        print(combined_out[-4000:], flush=True)
        raise RuntimeError(f"Compilation failed for {tex_abs} (exit {proc.returncode})")

    # Check undefined references / citations / control sequences
    undef_refs = re.findall(r"LaTeX Warning: Reference `([^']+)' on page \d+ undefined", log_text)
    undef_cites = re.findall(r"LaTeX Warning: Citation `([^']+)' on page \d+ undefined", log_text)
    undef_ctrl = re.findall(r"Undefined control sequence", log_text)

    overfulls = []
    for m in re.finditer(r"Overfull \\hbox \(([0-9.]+)pt too wide\) in paragraph at lines (\d+)--(\d+)", log_text):
        pts = float(m.group(1))
        if pts > 2.0:
            overfulls.append((pts, int(m.group(2)), int(m.group(3))))

    pdf_path = os.path.join(work_dir, f"{stem}.pdf")
    if not os.path.exists(pdf_path):
        raise RuntimeError(f"Expected PDF output missing: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    n_pages = len(doc)
    doc.close()

    print(f"  -> Generated {os.path.relpath(pdf_path, ROOT)} ({n_pages} pages, {os.path.getsize(pdf_path)/1024:.1f} KB)")
    print(f"  -> Undefined refs: {len(undef_refs)} | Undefined cites: {len(undef_cites)} | Overfull >2pt: {len(overfulls)}")
    for pts, l1, l2 in overfulls:
        print(f"     [Overfull] {pts:.2f}pt too wide at lines {l1}--{l2}")

    if undef_refs or undef_cites or undef_ctrl:
        raise RuntimeError(f"Undefined references/citations/macros in {tex_abs}: refs={undef_refs}, cites={undef_cites}")
    if strict_hbox and overfulls:
        raise RuntimeError(f"Found {len(overfulls)} Overfull \\hbox > 2.0pt in {tex_abs}: {overfulls}")

    return {"pdf_path": pdf_path, "n_pages": n_pages, "overfulls": overfulls}


def render_qa_pages(pdf_path: str, prefix: str, dpi: int = 200) -> list:
    os.makedirs(QA_DIR, exist_ok=True)
    # Remove old PNGs for this prefix
    for fn in os.listdir(QA_DIR):
        if fn.startswith(prefix) and fn.endswith(".png"):
            os.remove(os.path.join(QA_DIR, fn))

    doc = pymupdf.open(pdf_path)
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    out_paths = []
    for i, page in enumerate(doc, 1):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_png = os.path.join(QA_DIR, f"{prefix}-{i:02d}.png")
        pix.save(out_png)
        out_paths.append(out_png)
    doc.close()
    print(f"  -> Rendered {len(out_paths)} QA pages ({dpi} DPI) to {os.path.relpath(QA_DIR, ROOT)}/{prefix}-*.png")
    return out_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-strict-hbox", action="store_true")
    args = parser.parse_args()
    strict_hbox = not args.no_strict_hbox

    targets = [
        (os.path.join(ROOT, "MANUSCRIPT_GENERAL_V4.tex"), "ms_v4_page", True),
        (os.path.join(ROOT, "SI_APPENDIX_V4.tex"), "si_v4_page", True),
        (os.path.join(ROOT, "manuscripts", "RESPONSE_TO_EDITOR_LIU.tex"), "resp_v4_page", False),
        (os.path.join(ROOT, "manuscripts", "RESPONSE_TO_PEER_REVIEW.tex"), "peer_v4_page", False),
        (os.path.join(ROOT, "manuscripts", "COVER_LETTER_JCTC_V4_RESUBMISSION.tex"), "cov_v4_page", False),
    ]

    for tex_p, prefix, required in targets:
        if not os.path.exists(tex_p):
            if required:
                raise FileNotFoundError(f"Required TeX file missing: {tex_p}")
            continue
        res = compile_tex(tex_p, strict_hbox=strict_hbox)
        render_qa_pages(res["pdf_path"], prefix)
        # Mirror root PDFs into manuscripts/ for convenience
        if os.path.dirname(tex_p) == ROOT:
            dst = os.path.join(ROOT, "manuscripts", os.path.basename(res["pdf_path"]))
            shutil.copy2(res["pdf_path"], dst)
            shutil.copy2(tex_p, os.path.join(ROOT, "manuscripts", os.path.basename(tex_p)))

    print("\n=== ALL V4 LATEX DOCUMENTS COMPILED & QA-VERIFIED CLEANLY ===")


if __name__ == "__main__":
    main()
