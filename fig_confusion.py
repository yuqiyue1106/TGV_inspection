import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
s = json.loads((ROOT / "code/runs/eval_test_gpu_v8s_1024/confusion.json").read_text(encoding="utf-8"))
m = s["matrix_gt_rows_pred_cols_dust_scratch_miss"]

NAVY, AMBER, GREY = "#14243B", "#E08A00", "#AEB6C2"
rows = [(0, "灰尘", m[0]), (1, "划痕", m[1])]

W, H = 5.85, 2.2
fig, ax = plt.subplots(figsize=(W, H), dpi=300)
ys = [1.0, 0.0]
for y, (cls, name, cells) in zip(ys, rows):
    total = sum(cells)
    left = 0.0
    for idx, val in enumerate(cells):
        color = NAVY if idx == cls else (AMBER if idx < 2 else GREY)
        ax.barh(y, val, left=left, height=0.44, color=color, linewidth=0)
        if val / total > 0.06:
            ax.text(left + val / 2, y, f"{val}", ha="center", va="center",
                    color="white", fontsize=8.5, fontweight="bold")
        left += val
    ax.text(-8, y, name, ha="right", va="center", fontsize=9, color="#2B3440")
    ax.text(total + 8, y, f"真值 {total}·召回 {cells[cls] / total:.3f}", ha="left", va="center",
            fontsize=8, color="#5A6472")

ax.set_ylim(-0.55, 1.55)
ax.set_xlim(0, max(sum(r[2]) for r in rows) * 1.34)
ax.set_yticks([])
ax.set_xlabel("标注框数量（留出集 176 张，conf 0.25，匹配 IoU 0.5）", fontsize=8, color="#5A6472", labelpad=6)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color("#C7CDD6")
ax.tick_params(axis="x", labelsize=7.5, colors="#5A6472", length=2)
ax.grid(axis="x", color="#E7EAEF", linewidth=0.6)
ax.set_axisbelow(True)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (NAVY, AMBER, GREY)]
ax.legend(handles, ["正确类别检出", "误判为另一类", "未检出"], loc="lower center",
          bbox_to_anchor=(0.55, 1.0), ncol=3, frameon=False, fontsize=8,
          handlelength=0.9, handleheight=0.9, columnspacing=1.2)
fig.subplots_adjust(left=0.14, right=0.985, top=0.84, bottom=0.2)
fig.savefig(ROOT / "outputs/figs/chart_confusion.png", dpi=300, facecolor="white")
print("ok")
