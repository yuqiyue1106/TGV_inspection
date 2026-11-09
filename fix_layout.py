"""Fix post-insert layout: stale section ref, revision row overflow, risk table split."""
from pathlib import Path

import docx
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

DOC = Path("E:/TGV inspection detection/TGV通孔检测技术预研报告.docx")
d = docx.Document(str(DOC))

# 1) stale section reference 9.1 -> 10.1 (cost model section moved)
fixed = 0
for p in d.paragraphs:
    if "均依赖 9.1 节参数取值" in p.text:
        for r in p.runs:
            if "9.1" in r.text:
                r.text = r.text.replace("9.1", "10.1")
                fixed += 1
                break
        else:
            raise SystemExit("run split, handle manually")
print("ref fixed", fixed)

# 2) revision table: drop the last empty row so it stops spilling to page 3
rev = None
for t in d.tables:
    if t.rows[0].cells[0].text.strip() == "日期":
        rev = t
last = rev.rows[-1]
assert all(not c.text.strip() for c in last.cells)
rev._tbl.remove(last._tr)
print("rev rows", len(rev.rows))

# 3) risk table: no row split across pages, repeat header
risk = None
for t in d.tables:
    if t.rows[0].cells[0].text.strip() == "风险项":
        risk = t
head_tr = risk.rows[0]._tr
head_pr = head_tr.find(qn("w:trPr"))
if head_pr is None:
    head_pr = OxmlElement("w:trPr")
    head_tr.insert(0, head_pr)
if head_pr.find(qn("w:tblHeader")) is None:
    hdr = OxmlElement("w:tblHeader")
    hpos = 0
    for k, ch in enumerate(list(head_pr)):
        if ch.tag in (qn("w:trHeight"), qn("w:jc")):
            hpos = k + 1
    head_pr.insert(hpos, hdr)
for row in risk.rows[1:]:
    tr = row._tr
    pr = tr.find(qn("w:trPr"))
    if pr is None:
        pr = OxmlElement("w:trPr")
        tr.insert(0, pr)
    if pr.find(qn("w:cantSplit")) is None:
        pr.insert(0, OxmlElement("w:cantSplit"))
print("risk rows", len(risk.rows))

d.save(str(DOC))
print("saved")
