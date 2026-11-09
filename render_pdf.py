"""Render pages of the exported PDF and report chapter page ranges."""
from pathlib import Path

import fitz

PDF = Path("E:/TGV inspection detection/Exp_Data/TGV_yolo_Dataset/outputs/report_check.pdf")
OUT = Path("E:/TGV inspection detection/Exp_Data/TGV_yolo_Dataset/outputs/pages")
OUT.mkdir(exist_ok=True)
for old in OUT.glob("*.png"):
    old.unlink()

doc = fitz.open(PDF)
lines = []
hits = {}
for i, page in enumerate(doc, start=1):
    text = page.get_text()
    for key in ("7. 板面灰尘", "7.1 专题定位", "7.3 数据与标注", "7.5 验证过程与结果",
                "7.6 可行性判定", "7.7 结论与后续", "8.风险评估", "9.资源投入",
                "10.成本分析", "11.预研结论", "12.附件"):
        if key in text:
            hits[key] = i
    lines.append(f"p{i}: chars={len(text)}")
print("pages", doc.page_count)
for k, v in hits.items():
    print(k, "->", v)
for i, page in enumerate(doc, start=1):
    if 1 <= i <= doc.page_count:
        pix = page.get_pixmap(dpi=110)
        pix.save(OUT / f"p{i:02d}.png")
print("rendered", doc.page_count)
