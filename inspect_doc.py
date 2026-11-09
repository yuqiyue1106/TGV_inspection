"""Dump caption fields, tables and headings of the report for numbering checks."""
from pathlib import Path

import docx
from docx.oxml.ns import qn

DOC = Path("E:/TGV inspection detection/TGV通孔检测技术预研报告.docx")
d = docx.Document(str(DOC))
body = d.element.body

out = []
p_idx = 0
t_idx = 0
for child in body.iterchildren():
    if child.tag == qn("w:p"):
        p = docx.text.paragraph.Paragraph(child, d)
        text = p.text.strip()
        instr = []
        cached = []
        state = 0
        for r in child.findall(qn("w:r")):
            for fc in r.findall(qn("w:fldChar")):
                if fc.get(qn("w:fldCharType")) == "begin":
                    state = 1
                elif fc.get(qn("w:fldCharType")) == "separate":
                    state = 2
                elif fc.get(qn("w:fldCharType")) == "end":
                    state = 0
            for it in r.findall(qn("w:instrText")):
                if state == 1:
                    instr.append(it.text or "")
            for t in r.findall(qn("w:t")):
                if state == 2:
                    cached.append(t.text or "")
        if p.style.name.startswith("Heading") or text[:1] in "表图" and (" " in text or "　" in text):
            mark = f" [SEQ:{''.join(instr).strip()} => {''.join(cached)!r}]" if instr else ""
            out.append(f"P{p_idx:4d} {p.style.name:12s} {text[:70]}{mark}")
        p_idx += 1
    elif child.tag == qn("w:tbl"):
        t = docx.table.Table(child, d)
        hdr = " | ".join(c.text.strip() for c in t.rows[0].cells)
        out.append(f"T{t_idx:3d} rows={len(t.rows)} cols={len(t.columns)} :: {hdr[:100]}")
        t_idx += 1

out.append(f"TOTAL paragraphs={len(d.paragraphs)} tables={len(d.tables)}")
Path("E:/TGV inspection detection/Exp_Data/TGV_yolo_Dataset/outputs/_inspect.txt").write_text(
    "\n".join(out), encoding="utf-8")
print("ok", len(out))
