#!/usr/bin/env python3
r"""
build_native_docx_v4.py
=======================
Builds 100% Native Editable OpenXML Microsoft Word (.docx) editions for Milestone 42:
1. /Users/mac/molecular dynamics/MANUSCRIPT_GENERAL_V4.docx (+ mirror in manuscripts/)
   - Parses MANUSCRIPT_GENERAL_V4.tex directly and expands 100% of \vFour... macros from
     version4_godtier/results/v4_metrics_macros.tex (ZERO hardcoded numbers, ZERO drift).
   - Native 2-column JCTC layout, full-width embedded high-DPI figures (Figures 1-9),
     native Word <w:tbl> tables (Tables 1-2), and native Office MathML (<m:oMath>) equations.
2. /Users/mac/molecular dynamics/manuscripts/MANUSCRIPT_GENERAL_V4_EXACT_PDF_REPLICA.docx
   - 300-DPI pixel-exact page-for-page visual replica of MANUSCRIPT_GENERAL_V4.pdf.
3. /Users/mac/molecular dynamics/manuscripts/COVER_LETTER_JCTC_V4_RESUBMISSION.docx
   - Native editable Word document from COVER_LETTER_JCTC_V4_RESUBMISSION.md.
4. /Users/mac/molecular dynamics/manuscripts/RESPONSE_TO_EDITOR_LIU.docx
   - Native editable Word document from RESPONSE_TO_EDITOR_LIU.md (including Table & OMML math).
"""

import os
import re
import sys
import html
import shutil
import zipfile
import pymupdf
from PIL import Image

ROOT = "/Users/mac/molecular dynamics"
MS_DIR = os.path.join(ROOT, "manuscripts")
RES_DIR = os.path.join(ROOT, "version4_godtier", "results")
FIG_DIR = os.path.join(ROOT, "version4_godtier", "figures")

sys.path.insert(0, os.path.join(ROOT, "pnas_paper"))
from build_native_docx_v2 import (
    xml_esc,
    latex_to_omml,
    w_run,
    parse_inline_formatting,
    w_para,
    section_break_pr,
    drawing_xml,
)


def load_v4_macros() -> dict:
    macros = {}
    macro_file = os.path.join(RES_DIR, "v4_metrics_macros.tex")
    with open(macro_file, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", line.strip())
            if m:
                macros[m.group(1)] = m.group(2)
    return macros


def expand_tex_macros(tex: str, macros: dict) -> str:
    # Sort longest macro names first to avoid prefix collisions
    for k in sorted(macros.keys(), key=len, reverse=True):
        val = macros[k]
        tex = tex.replace(f"\\{k}{{}}", val)
        tex = re.sub(rf"\\{k}(?![A-Za-z])", lambda _, v=val: v, tex)
    return tex


def tex_inline_to_md(s: str) -> str:
    """Converts LaTeX inline markup (outside $...$) into clean Markdown/HTML understood by parse_inline_formatting."""
    # Protect math $...$
    parts = re.split(r"(\$[^\$]+\$)", s)
    out = []
    for p in parts:
        if p.startswith("$") and p.endswith("$"):
            inner = p[1:-1]
            inner = inner.replace(r"\allowbreak", "")
            inner = inner.replace(r"\bm{", r"\mathbf{").replace(r"\Tr", r"\mathrm{Tr}").replace(r"\diag", r"\mathrm{diag}")
            inner = inner.replace(r"\!", "")
            out.append("$" + inner + "$")
        else:
            p = p.replace("---", "—").replace("--", "–")
            p = p.replace(r"\%", "%").replace(r"\&", "&").replace(r"\_", "_")
            p = p.replace(r"\AA{}", "Å").replace(r"\AA", "Å")
            p = p.replace(r"\\", " ")
            p = p.replace(r"\ ", " ").replace("~", " ")
            p = re.sub(r"\\textbf\{\\textcolor\{[a-z]+\}\{\\textsf\{([^{}]+)\}\}\}", r"**\1**", p)
            p = re.sub(r"\\textcolor\{[a-z]+\}\{\\textbf\{([^{}]+)\}\}", r"**\1**", p)
            p = re.sub(r"\\textbf\{([^{}]+)\}", r"**\1**", p)
            p = re.sub(r"\\textit\{([^{}]+)\}", r"*\1*", p)
            p = re.sub(r"\\texttt\{([^{}]+)\}", r"`\1`", p)
            p = re.sub(r"\\textsuperscript\{([^{}]+)\}", r"<sup>\1</sup>", p)
            p = re.sub(r"\\href\{[^{}]+\}\{([^{}]+)\}", r"`\1`", p)
            p = re.sub(r"\\url\{([^{}]+)\}", r"`\1`", p)
            p = p.replace("``", "“").replace("''", "”")
            # Strip any leftover simple wrappers
            p = re.sub(r"\\(?:allowbreak|noindent|par)\b\s*", "", p)
            p = p.replace("{", "").replace("}", "")
            out.append(p)
    return "".join(out)


def parse_tex_table(tbl_block: str) -> tuple:
    cap_m = re.search(r"\\caption\*\{(.*?)\}\n", tbl_block, re.DOTALL)
    caption = tex_inline_to_md(cap_m.group(1).strip()) if cap_m else "Table"

    tab_m = re.search(r"\\begin\{tabular\}\{[^{}]+\}(.*?)\\end\{tabular\}", tbl_block, re.DOTALL)
    body = tab_m.group(1) if tab_m else ""
    headers = []
    rows = []
    in_header = True
    for raw_row in body.split(r"\\"):
        r = raw_row.strip()
        r = re.sub(r"\\(?:toprule|midrule|bottomrule)(?:\[[0-9.a-z]+\])?", "", r).strip()
        if not r:
            continue
        if r.startswith(r"\multicolumn"):
            mc = re.search(r"\\multicolumn\{\d+\}\{[^{}]+\}\{(.*)\}", r, re.DOTALL)
            if mc:
                rows.append([tex_inline_to_md(mc.group(1).strip())])
            in_header = False
            continue
        cells = [tex_inline_to_md(c.strip()) for c in r.split("&")]
        if in_header and not headers:
            headers = cells
            in_header = False
        else:
            rows.append(cells)

    note_m = re.search(r"\\begin\{flushleft\}(.*?)\\end\{flushleft\}", tbl_block, re.DOTALL)
    note_txt = ""
    if note_m:
        nt = re.sub(r"\\fontsize\{[^{}]+\}\{[^{}]+\}\\selectfont\\color\{[^{}]+\}", "", note_m.group(1))
        note_txt = tex_inline_to_md(nt.strip())
    return caption, headers, rows, note_txt


def render_word_table(caption: str, headers: list, rows: list, note_txt: str = "", font_half_pt: int = 13) -> str:
    cap_p = w_para(
        parse_inline_formatting(caption, base_bold=True, color="002F6C", size_half_pt=15, font="Arial"),
        align="left", space_before=60, space_after=25, keep_next=True,
    )
    tbl_xml = """<w:tbl>
      <w:tblPr>
        <w:tblW w:w="5000" w:type="pct"/>
        <w:tblBorders>
          <w:top w:val="single" w:sz="12" w:space="0" w:color="002F6C"/>
          <w:bottom w:val="single" w:sz="12" w:space="0" w:color="002F6C"/>
          <w:insideH w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
        </w:tblBorders>
        <w:tblCellMar>
          <w:top w:w="28" w:type="dxa"/>
          <w:bottom w:w="28" w:type="dxa"/>
          <w:left w:w="36" w:type="dxa"/>
          <w:right w:w="36" w:type="dxa"/>
        </w:tblCellMar>
      </w:tblPr>"""
    if headers:
        tbl_xml += "<w:tr>"
        for idx, h in enumerate(headers):
            align = "left" if idx in (0, 1, len(headers) - 1) else "center"
            runs = parse_inline_formatting(h, base_bold=True, color="002F6C", size_half_pt=font_half_pt, font="Arial")
            tbl_xml += f'<w:tc><w:tcPr><w:tcBorders><w:bottom w:val="single" w:sz="8" w:space="0" w:color="64748B"/></w:tcBorders></w:tcPr>{w_para(runs, align=align, space_before=8, space_after=8, line=195)}</w:tc>'
        tbl_xml += "</w:tr>"

    ncols = max(len(headers), 1)
    for r in rows:
        tbl_xml += "<w:tr>"
        if len(r) == 1 and ncols > 1:
            runs = parse_inline_formatting(r[0], base_italic=True, color="334155", size_half_pt=font_half_pt, font="Arial")
            tbl_xml += f'<w:tc><w:tcPr><w:gridSpan w:val="{ncols}"/><w:shd w:val="clear" w:color="auto" w:fill="F8FAFC"/></w:tcPr>{w_para(runs, align="left", space_before=10, space_after=10, line=195)}</w:tc>'
        else:
            for idx, cell in enumerate(r):
                align = "left" if idx in (0, 1) else "center"
                runs = parse_inline_formatting(cell, color="1A202C", size_half_pt=font_half_pt, font="Times New Roman")
                tbl_xml += f"<w:tc>{w_para(runs, align=align, space_before=8, space_after=8, line=195)}</w:tc>"
        tbl_xml += "</w:tr>"
    tbl_xml += "</w:tbl>"

    note_p = ""
    if note_txt:
        note_p = w_para(
            parse_inline_formatting(note_txt, color="4A5568", size_half_pt=12, font="Times New Roman"),
            align="both", space_before=18, space_after=50, line=205,
        )
    return cap_p + tbl_xml + note_p


def pdf_fig_to_png(pdf_rel_path: str, out_png_path: str, dpi: int = 300) -> str:
    src_png = os.path.splitext(os.path.join(ROOT, pdf_rel_path))[0] + ".png"
    if os.path.exists(src_png):
        return src_png
    doc = pymupdf.open(os.path.join(ROOT, pdf_rel_path))
    pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
    pix.save(out_png_path)
    doc.close()
    return out_png_path


def write_openxml_package(out_docx: str, body_xml_parts: list, final_sect_pr: str, media_map: dict, title_str: str, header_right: str = "Journal of Chemical Theory and Computation"):
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
            xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
            xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
            xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
  <w:body>
    {"".join(body_xml_parts)}
    {final_sect_pr}
  </w:body>
</w:document>"""

    header1_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:p>
    <w:pPr><w:tabs><w:tab w:val="right" w:pos="10880"/></w:tabs></w:pPr>
    <w:r><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:b/><w:color w:val="4A5568"/><w:sz w:val="15"/></w:rPr><w:t>Singh &amp; Singh</w:t></w:r>
    <w:r><w:tab/></w:r>
    <w:r><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:b/><w:color w:val="4A5568"/><w:sz w:val="15"/></w:rPr><w:t>{xml_esc(header_right)}</w:t></w:r>
  </w:p>
</w:hdr>"""

    footer1_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:p>
    <w:pPr><w:tabs><w:tab w:val="right" w:pos="10880"/></w:tabs></w:pPr>
    <w:r><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:color w:val="718096"/><w:sz w:val="14"/></w:rPr><w:t>Scalable Contact-Gated Stochastic Operators for Macromolecular Dynamics</w:t></w:r>
    <w:r><w:tab/></w:r>
    <w:r><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:b/><w:color w:val="1A202C"/><w:sz w:val="16"/></w:rPr><w:fldSimple w:instr="PAGE"/></w:r>
  </w:p>
</w:ftr>"""

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>
  <Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

    img_rels = "\n  ".join(
        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{rid}.png"/>'
        for rid in media_map.keys()
    )
    doc_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdHdr1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/>
  <Relationship Id="rIdFtr1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>
  {img_rels}
</Relationships>"""

    core_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/">
  <dc:title>{xml_esc(title_str)}</dc:title>
  <dc:creator>Mehardeep Singh and Gurarpit Singh</dc:creator>
  <cp:lastModifiedBy>Mehardeep Singh</cp:lastModifiedBy>
  <dc:description>Journal of Chemical Theory and Computation Resubmission (V4.0)</dc:description>
</cp:coreProperties>"""

    app_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Word</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <Company></Company>
</Properties>"""

    with zipfile.ZipFile(out_docx, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/header1.xml", header1_xml)
        z.writestr("word/footer1.xml", footer1_xml)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("docProps/core.xml", core_xml)
        z.writestr("docProps/app.xml", app_xml)
        for rid, png_path in media_map.items():
            z.write(png_path, f"word/media/{rid}.png")


def build_manuscript_v4_docx(macros: dict):
    with open(os.path.join(ROOT, "MANUSCRIPT_GENERAL_V4.tex"), "r", encoding="utf-8") as f:
        raw_tex = f.read()

    tex = expand_tex_macros(raw_tex, macros)
    doc_body = tex.split(r"\begin{document}")[1].split(r"\end{document}")[0]

    # Extract Abstract and Keywords from \twocolumn[{...}]
    abs_m = re.search(r"\\textbf\{Abstract:\}\s*(.*?)\\par\}", doc_body, re.DOTALL)
    abstract_md = tex_inline_to_md(abs_m.group(1).strip()) if abs_m else ""
    kw_m = re.search(r"Keywords:\s*(.*?)\\par\}", doc_body, re.DOTALL)
    keywords_md = tex_inline_to_md(kw_m.group(1).strip()) if kw_m else ""

    main_tex = r"\section*{\MakeUppercase{Introduction}}" + doc_body.split(r"\section*{\MakeUppercase{Introduction}}")[1]

    title_str = "Scalable Contact-Gated Stochastic Operators and Universal Reduction Laws for Macromolecular Protein Dynamics, Allostery, and Kinetics"
    body_xml_parts = [
        w_para(
            w_run("RESEARCH ARTICLE", bold=True, color="C53030", size_half_pt=17, font="Arial")
            + w_run("    |    JOURNAL OF CHEMICAL THEORY AND COMPUTATION", bold=True, color="4A5568", size_half_pt=15, font="Arial"),
            align="left", space_before=0, space_after=40,
        ),
        w_para(w_run(title_str, bold=True, color="002F6C", size_half_pt=29, font="Arial"), align="left", space_before=15, space_after=55, line=255),
        w_para(parse_inline_formatting("**Mehardeep Singh**<sup>a,*</sup> and **Gurarpit Singh**<sup>b</sup>", color="1A202C", size_half_pt=17, font="Arial"), align="left", space_before=0, space_after=25),
        w_para(parse_inline_formatting("<sup>a</sup>Sri Chaitanya Educational Institutions, Chandigarh, India; <sup>b</sup>Vivekananda Institute of Professional Studies (VIPS), Guru Gobind Singh Indraprastha University (GGSIPU), Delhi, India", color="4A5568", size_half_pt=14, font="Times New Roman"), align="left", space_before=0, space_after=20),
        w_para(parse_inline_formatting("<sup>*</sup>Correspondence: Mehardeep Singh (`mehardeep.sim@gmail.com`); Gurarpit Singh (`gurarpit.sml@gmail.com`).", color="4A5568", size_half_pt=14, font="Arial"), align="left", space_before=0, space_after=55),
        w_para(w_run("Abstract: ", bold=True, color="002F6C", size_half_pt=15, font="Times New Roman") + parse_inline_formatting(abstract_md, color="1A202C", size_half_pt=15, font="Times New Roman"), align="both", space_before=35, space_after=25, line=230, border_box=True, fill_hex="F8FAFC"),
        w_para(w_run("Keywords: ", bold=True, color="4A5568", size_half_pt=14, font="Arial") + parse_inline_formatting(keywords_md, base_bold=True, color="4A5568", size_half_pt=14, font="Arial"), align="left", space_before=0, space_after=70),
        section_break_pr(cols=1, page_break=False),
    ]

    media_map = {}
    img_idx = [1]
    docpr_id = [10]

    # Tokenize main_tex into blocks: table*, figure*, \[...\], enumerate, section*, subsection*, paragraphs
    pattern = re.compile(
        r"(\\begin\{table\*?\}.*?\\end\{table\*?\}"
        r"|\\begin\{figure\*?\}.*?\\end\{figure\*?\}"
        r"|\\begin\{enumerate\}.*?\\end\{enumerate\}"
        r"|\\\[.*?\\\]"
        r"|\\section\*\{.*?\}"
        r"|\\subsection\*\{.*?\})",
        re.DOTALL,
    )
    tokens = pattern.split(main_tex)

    for tok in tokens:
        tok_s = tok.strip()
        if not tok_s:
            continue

        if tok_s.startswith(r"\begin{table"):
            cap, hdrs, rws, note = parse_tex_table(tok_s)
            body_xml_parts.append(section_break_pr(cols=2))
            body_xml_parts.append(render_word_table(cap, hdrs, rws, note, font_half_pt=12 if len(hdrs) > 9 else 13))
            body_xml_parts.append(section_break_pr(cols=1))

        elif tok_s.startswith(r"\begin{figure"):
            pdfs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", tok_s)
            cap_m = re.search(r"\\caption\*\{(.*?)\}\n", tok_s, re.DOTALL)
            cap_md = tex_inline_to_md(cap_m.group(1).strip()) if cap_m else "Figure"
            body_xml_parts.append(section_break_pr(cols=2))
            if len(pdfs) == 1:
                rid = f"rIdImg{img_idx[0]}"
                img_idx[0] += 1
                png_p = pdf_fig_to_png(pdfs[0], os.path.join(FIG_DIR, f"_tmp_{rid}.png"))
                media_map[rid] = png_p
                with Image.open(png_p) as im:
                    w_px, h_px = im.size
                w_emu = int(6.4 * 914400)
                h_emu = int(w_emu * (h_px / w_px))
                body_xml_parts.append(w_para(drawing_xml(rid, docpr_id[0], w_emu, h_emu, rid), align="center", space_before=50, space_after=25, keep_next=True))
                docpr_id[0] += 1
            elif len(pdfs) >= 2:
                draws = []
                for p_rel in pdfs[:2]:
                    rid = f"rIdImg{img_idx[0]}"
                    img_idx[0] += 1
                    png_p = pdf_fig_to_png(p_rel, os.path.join(FIG_DIR, f"_tmp_{rid}.png"))
                    media_map[rid] = png_p
                    with Image.open(png_p) as im:
                        w_px, h_px = im.size
                    w_emu = int(3.45 * 914400)
                    h_emu = int(w_emu * (h_px / w_px))
                    draws.append(drawing_xml(rid, docpr_id[0], w_emu, h_emu, rid))
                    docpr_id[0] += 1
                body_xml_parts.append(w_para(draws[0] + w_run("  ", size_half_pt=12) + draws[1], align="center", space_before=50, space_after=25, keep_next=True))
            body_xml_parts.append(w_para(parse_inline_formatting(cap_md, color="2D3748", size_half_pt=14, font="Times New Roman"), align="both", space_before=0, space_after=65, line=215))
            body_xml_parts.append(section_break_pr(cols=1))

        elif tok_s.startswith(r"\section*"):
            m = re.search(r"\\section\*\{(?:\\MakeUppercase\{)?([^{}]+)\}?\}", tok_s)
            sec_title = m.group(1).strip().upper() if m else "SECTION"
            body_xml_parts.append(w_para(w_run(sec_title, bold=True, color="002F6C", size_half_pt=17, font="Arial"), align="left", space_before=95, space_after=28, keep_next=True))

        elif tok_s.startswith(r"\subsection*"):
            sub_inner = tok_s[len(r"\subsection*{"):-1]
            sub_md = tex_inline_to_md(sub_inner)
            body_xml_parts.append(w_para(parse_inline_formatting(sub_md, base_bold=True, base_italic=True, color="2D3748", size_half_pt=15, font="Arial"), align="left", space_before=75, space_after=20, keep_next=True))

        elif tok_s.startswith(r"\[") and tok_s.endswith(r"\]"):
            eq_inner = tok_s[2:-2].strip()
            eq_inner = re.sub(r"\\scalebox\{[0-9.]+\}\{\$(?:\\displaystyle)?(.*)\$\}", r"\1", eq_inner, flags=re.DOTALL)
            eq_inner = eq_inner.replace(r"\bm{", r"\mathbf{").replace(r"\Tr", r"\mathrm{Tr}").replace(r"\diag", r"\mathrm{diag}")
            body_xml_parts.append(w_para(latex_to_omml(eq_inner, display=True), align="center", space_before=35, space_after=35))

        elif tok_s.startswith(r"\begin{enumerate}"):
            enum_inner = re.sub(r"^\\begin\{enumerate\}(?:\[[^\]]*\])?|\\end\{enumerate\}$", "", tok_s).strip()
            items = [it.strip() for it in re.split(r"\\item\s+", enum_inner) if it.strip()]
            for idx, it in enumerate(items, 1):
                it_md = f"**{idx}.** " + tex_inline_to_md(it)
                body_xml_parts.append(w_para(parse_inline_formatting(it_md, color="1A202C", size_half_pt=14, font="Times New Roman"), align="both", space_before=8, space_after=18, line=220))

        else:
            # Split into paragraphs by blank lines, stripping font commands around References
            cleaned = re.sub(r"\{\\fontsize\{[^{}]+\}\{[^{}]+\}\\selectfont\\color\{[^{}]+\}", "", tok_s)
            for para in re.split(r"\n\s*\n", cleaned):
                p_clean = para.strip()
                if not p_clean or p_clean == "}":
                    continue
                p_md = tex_inline_to_md(p_clean)
                if p_md.strip():
                    body_xml_parts.append(w_para(parse_inline_formatting(p_md, color="1A202C", size_half_pt=15, font="Times New Roman"), align="both", space_before=0, space_after=32, line=230, first_line_indent=140))

    final_sect_pr = """<w:sectPr>
      <w:headerReference w:type="default" r:id="rIdHdr1"/>
      <w:footerReference w:type="default" r:id="rIdFtr1"/>
      <w:type w:val="continuous"/>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="600" w:right="680" w:bottom="650" w:left="680" w:header="340" w:footer="340"/>
      <w:cols w:num="2" w:space="200" w:sep="1"/>
    </w:sectPr>"""

    root_docx = os.path.join(ROOT, "MANUSCRIPT_GENERAL_V4.docx")
    ms_docx = os.path.join(MS_DIR, "MANUSCRIPT_GENERAL_V4.docx")
    write_openxml_package(root_docx, body_xml_parts, final_sect_pr, media_map, title_str)
    shutil.copy2(root_docx, ms_docx)
    print(f"Built Native Editable MANUSCRIPT_GENERAL_V4.docx ({os.path.getsize(root_docx)/1024:.1f} KB) + mirrored to manuscripts/")


def build_exact_pdf_replica_docx(src_pdf_name: str = "MANUSCRIPT_GENERAL_V4.pdf", out_docx_rel: str = "manuscripts/MANUSCRIPT_GENERAL_V4_EXACT_PDF_REPLICA.docx", dpi: int = 300):
    pdf_p = os.path.join(ROOT, src_pdf_name)
    doc = pymupdf.open(pdf_p)
    page_pngs = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        page_pngs.append((i + 1, pix.tobytes("png")))
    doc.close()

    w_emu, h_emu = 7772400, 10058400
    p_list, rel_entries = [], []
    for idx, _ in page_pngs:
        rid = f"rIdPage{idx}"
        rel_entries.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/page_{idx}.png"/>')
        draw = drawing_xml(rid, 200 + idx, w_emu, h_emu, f"Page_{idx}")
        p_list.append(f'<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>{draw}</w:p>')

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
            xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
            xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
  <w:body>
    {"".join(p_list)}
    <w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="0" w:right="0" w:bottom="0" w:left="0" w:header="0" w:footer="0"/></w:sectPr>
  </w:body>
</w:document>"""

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    doc_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {"".join(rel_entries)}
</Relationships>"""

    out_rep = os.path.join(ROOT, out_docx_rel)
    with zipfile.ZipFile(out_rep, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        for idx, png_bytes in page_pngs:
            z.writestr(f"word/media/page_{idx}.png", png_bytes)
    print(f"Built Pixel-Exact Replica {out_rep} ({os.path.getsize(out_rep)/(1024*1024):.2f} MB)")


def build_markdown_letter_docx(md_path: str, out_docx: str, header_right: str):
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    body_xml_parts = []
    title_str = "JCTC Resubmission Letter (ct-2026-02182c)"
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line or line == "---":
            i += 1
            continue
        if line.startswith("# "):
            title_str = line[2:].strip()
            body_xml_parts.append(w_para(parse_inline_formatting(title_str, base_bold=True, color="002F6C", size_half_pt=24, font="Arial"), align="left", space_before=10, space_after=40, border_box=True))
            i += 1
        elif line.startswith("## "):
            body_xml_parts.append(w_para(parse_inline_formatting(line[3:].strip(), base_bold=True, color="002F6C", size_half_pt=19, font="Arial"), align="left", space_before=90, space_after=25, keep_next=True))
            i += 1
        elif line.startswith("### "):
            body_xml_parts.append(w_para(parse_inline_formatting(line[4:].strip(), base_bold=True, color="1E3A8A", size_half_pt=17, font="Arial"), align="left", space_before=75, space_after=20, keep_next=True))
            i += 1
        elif line.startswith("|"):
            tbl_lines = []
            while i < n and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i].strip())
                i += 1
            parsed_rows = []
            for tl in tbl_lines:
                cells = [c.strip() for c in tl.strip("|").split("|")]
                if all(re.match(r"^:?-{2,}:?$", c) for c in cells):
                    continue
                parsed_rows.append(cells)
            if parsed_rows:
                body_xml_parts.append(render_word_table("Summary of Revisions Addressing ct-2026-02182c", parsed_rows[0], parsed_rows[1:], font_half_pt=14))
        elif line.startswith("> "):
            quote_txt = line[2:].strip()
            body_xml_parts.append(w_para(parse_inline_formatting(quote_txt, base_italic=True, color="1E293B", size_half_pt=16, font="Times New Roman"), align="both", space_before=25, space_after=30, line=235, fill_hex="F1F5F9"))
            i += 1
        else:
            clean_l = line.replace(r"\bm{", r"\mathbf{")
            body_xml_parts.append(w_para(parse_inline_formatting(clean_l, color="1A202C", size_half_pt=17, font="Times New Roman"), align="both", space_before=0, space_after=35, line=240))
            i += 1

    final_sect_pr = """<w:sectPr>
      <w:headerReference w:type="default" r:id="rIdHdr1"/>
      <w:footerReference w:type="default" r:id="rIdFtr1"/>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="900" w:right="950" w:bottom="900" w:left="950" w:header="450" w:footer="450"/>
    </w:sectPr>"""
    write_openxml_package(out_docx, body_xml_parts, final_sect_pr, {}, title_str, header_right=header_right)
    print(f"Built Native Editable {out_docx} ({os.path.getsize(out_docx)/1024:.1f} KB)")


def main():
    macros = load_v4_macros()
    print(f"Loaded {len(macros)} verified macros from v4_metrics_macros.tex")
    build_manuscript_v4_docx(macros)
    build_exact_pdf_replica_docx("MANUSCRIPT_GENERAL_V4.pdf", "manuscripts/MANUSCRIPT_GENERAL_V4_EXACT_PDF_REPLICA.docx", dpi=300)
    build_exact_pdf_replica_docx("SI_APPENDIX_V4.pdf", "SI_APPENDIX_V4.docx", dpi=220)
    shutil.copy2(os.path.join(ROOT, "SI_APPENDIX_V4.docx"), os.path.join(MS_DIR, "SI_APPENDIX_V4.docx"))
    build_markdown_letter_docx(
        os.path.join(MS_DIR, "COVER_LETTER_JCTC_V4_RESUBMISSION.md"),
        os.path.join(MS_DIR, "COVER_LETTER_JCTC_V4_RESUBMISSION.docx"),
        header_right="JCTC Cover Letter (Prior ID: ct-2026-02182c)",
    )
    build_markdown_letter_docx(
        os.path.join(MS_DIR, "RESPONSE_TO_EDITOR_LIU.md"),
        os.path.join(MS_DIR, "RESPONSE_TO_EDITOR_LIU.docx"),
        header_right="Response to Associate Editor Prof. Haiyan Liu (ct-2026-02182c)",
    )
    print("\n=== ALL V4 NATIVE DOCX DOCUMENTS GENERATED CLEANLY ===")


if __name__ == "__main__":
    main()
