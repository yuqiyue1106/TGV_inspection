"""List in-text references to tables/figures and chapters."""
import re
from pathlib import Path

import docx

DOC = Path("E:/TGV inspection detection/TGV通孔检测技术预研报告.docx")
d = docx.Document(str(DOC))
out = []
pat = re.compile(r"(表\s*\d+|图\s*\d+|第\s*\d+\s*[、，,]?\s*\d*\s*章)")
for i, p in enumerate(d.paragraphs):
    if p.style.name == "Caption":
        continue
    hits = pat.findall(p.text)
    if hits:
        out.append(f"P{i:4d} {p.style.name:14s} {sorted(set(hits))} :: {p.text.strip()[:160]}")
Path("E:/TGV inspection detection/Exp_Data/TGV_yolo_Dataset/outputs/_refs.txt").write_text(
    "\n".join(out), encoding="utf-8")
print("refs", len(out))
