#!/usr/bin/env python3
"""Build training curves and a readable report from results.csv + eval_test summary."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent
RUN = CODE / "runs" / "official_v8s_1024"
EVAL = CODE / "runs" / "eval_test_official" / "summary.json"
OUT = RUN / "report"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=RUN)
    p.add_argument("--eval", type=Path, default=EVAL)
    p.add_argument("--out", type=Path, default=None, help="defaults to <run>/report")
    return p.parse_args()


def load_results(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items() if v not in ("", None)})
    return rows


def wall_seconds(times):
    """Cumulative time restarts after a resume, so sum the segments."""
    total, seg_start, prev = 0.0, 0.0, None
    for t in times:
        if prev is not None and t < prev:
            total += prev - seg_start
            seg_start = 0.0
        prev = t
    return total + (prev or 0.0) - seg_start


def plot_metrics(rows, out):
    epochs = [r["epoch"] for r in rows]
    series = [
        ("mAP50-95", "metrics/mAP50-95(B)"),
        ("mAP50", "metrics/mAP50(B)"),
        ("Precision", "metrics/precision(B)"),
        ("Recall", "metrics/recall(B)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    best_epoch = max(rows, key=lambda r: r["metrics/mAP50-95(B)"])["epoch"]
    for ax, (title, key) in zip(axes.ravel(), series):
        values = [r[key] for r in rows]
        ax.plot(epochs, values, lw=1.6, color="#1f77b4")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.axvline(best_epoch, color="#d62728", ls="--", lw=1, label=f"best epoch {int(best_epoch)}")
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    fig.suptitle("TGV detection: validation metrics")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_losses(rows, out, total_epochs, close_mosaic):
    epochs = [r["epoch"] for r in rows]
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    for part, style in (("train", "-"), ("val", "--")):
        for name in ("box_loss", "cls_loss", "dfl_loss"):
            key = f"{part}/{name}"
            ax1.plot(epochs, [r[key] for r in rows], style, lw=1.4, label=f"{part} {name}")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.3)
    ax1.legend(ncol=3, fontsize=8)
    ax1.set_title("Train vs val losses")
    ax2.plot(epochs, [r["lr/pg0"] for r in rows], lw=1.8, color="#2ca02c")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("learning rate (pg0)")
    ax2.grid(alpha=0.3)
    if close_mosaic:
        start = total_epochs - close_mosaic + 1
        for ax in (ax1, ax2):
            ax.axvspan(start, total_epochs, color="#ff7f0e", alpha=0.12)
        ax1.annotate("mosaic off", xy=(start, ax1.get_ylim()[1]), xytext=(start, ax1.get_ylim()[1]),
                     fontsize=8, color="#ff7f0e", va="top")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_test(eval_data, out):
    breakdown = eval_data["breakdown"]
    overall = [r for r in breakdown if r["scope"] == "all"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    labels = [r["class"] for r in overall]
    xs = range(len(labels))
    axes[0].bar([x - 0.2 for x in xs], [r["precision"] for r in overall], 0.4, label="precision")
    axes[0].bar([x + 0.2 for x in xs], [r["recall"] for r in overall], 0.4, label="recall")
    axes[0].set_xticks(list(xs))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"Test P/R @ conf={eval_data['conf']}")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    for ax, scope, title in ((axes[1], "aperture", "Recall by aperture group"),
                             (axes[2], "label_origin", "Recall by label origin")):
        subsets = sorted({r["subset"] for r in breakdown if r["scope"] == scope})
        classes = sorted({r["class"] for r in breakdown if r["scope"] == scope})
        width = 0.8 / max(len(classes), 1)
        for i, cls in enumerate(classes):
            values = [next(r["recall"] for r in breakdown
                           if r["scope"] == scope and r["subset"] == s and r["class"] == cls) for s in subsets]
            ax.bar([x + i * width for x in range(len(subsets))], values, width, label=cls)
        ax.set_xticks(range(len(subsets)))
        ax.set_xticklabels(subsets)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def markdown_table(rows, columns):
    head = "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n"
    body = "".join("| " + " | ".join(str(r[c]) for c in columns) + " |\n" for r in rows)
    return head + body


def main():
    global RUN, EVAL, OUT
    args = parse_args()
    RUN, EVAL = args.run, args.eval
    OUT = args.out or RUN / "report"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_results(RUN / "results.csv")
    if not rows:
        raise SystemExit("no epochs recorded")

    args_file = RUN / "args.yaml"
    total_epochs, close_mosaic = 100, 10
    if args_file.exists():
        import yaml
        cfg = yaml.safe_load(args_file.read_text())
        total_epochs = int(cfg.get("epochs", total_epochs))
        close_mosaic = int(cfg.get("close_mosaic", close_mosaic))

    best = max(rows, key=lambda r: r["metrics/mAP50-95(B)"])
    hours = wall_seconds([r["time"] for r in rows]) / 3600
    plot_metrics(rows, OUT / "curves_metrics.png")
    plot_losses(rows, OUT / "curves_losses.png", total_epochs, close_mosaic)

    eval_data = json.loads(EVAL.read_text()) if EVAL.exists() else None
    if eval_data:
        plot_test(eval_data, OUT / "test_breakdown.png")

    lines = ["# TGV 缺陷检测训练报告", ""]
    lines += [
        "## 1. 结论",
        "",
        f"- 完成 epoch：{int(rows[-1]['epoch'])}/{total_epochs}（早停或跑满）",
        f"- 最佳 epoch：{int(best['epoch'])}",
        f"- 验证集 mAP50：**{best['metrics/mAP50(B)']:.4f}**，mAP50-95：**{best['metrics/mAP50-95(B)']:.4f}**",
        f"- 该 epoch 的 P/R：{best['metrics/precision(B)']:.4f} / {best['metrics/recall(B)']:.4f}",
        f"- 累计训练时长：{hours:.2f} h（含中断续训）",
        "",
    ]
    if eval_data:
        official = eval_data["official"]
        per_class = eval_data.get("official_per_class", {})
        lines += [
            "## 2. 留出集（test，未参与训练）",
            "",
            f"- 图像数：{eval_data['images']}（人工标注 {eval_data['hand_images']} 张 / 机器预标 {eval_data['machine_images']} 张）",
            f"- 官方指标：mAP50 = **{official['mAP50']:.4f}**，mAP50-95 = **{official['mAP50-95']:.4f}**",
            "",
        ]
        if per_class:
            pc_rows = [{"类别": k, **{kk: f"{vv:.4f}" for kk, vv in v.items()}} for k, v in per_class.items()]
            lines.append(markdown_table(pc_rows, ["类别", "precision", "recall", "mAP50", "mAP50-95"]))
            lines.append("")
        overall = [r for r in eval_data["breakdown"] if r["scope"] == "all"]
        lines += [
            f"### 工作点（conf={eval_data['conf']}，匹配 IoU={eval_data['match_iou']}）",
            "",
            markdown_table(
                [{"类别": r["class"], "GT": r["gt"], "TP": r["tp"], "FP": r["fp"], "FN": r["fn"],
                  "P": f"{r['precision']:.3f}", "R": f"{r['recall']:.3f}",
                  "框/图": f"{r['shapes_per_image']:.2f}"} for r in overall],
                ["类别", "GT", "TP", "FP", "FN", "P", "R", "框/图"]),
            "",
            "### 按孔径分组",
            "",
            markdown_table(
                [{"组": r["subset"], "类别": r["class"], "图像": r["images"], "GT": r["gt"],
                  "P": f"{r['precision']:.3f}", "R": f"{r['recall']:.3f}", "框/图": f"{r['shapes_per_image']:.2f}"}
                 for r in eval_data["breakdown"] if r["scope"] == "aperture"],
                ["组", "类别", "图像", "GT", "P", "R", "框/图"]),
            "",
            "### 按标签来源（人工 vs 机器预标）",
            "",
            markdown_table(
                [{"来源": r["subset"], "类别": r["class"], "图像": r["images"], "GT": r["gt"],
                  "P": f"{r['precision']:.3f}", "R": f"{r['recall']:.3f}"}
                 for r in eval_data["breakdown"] if r["scope"] == "label_origin"],
                ["来源", "类别", "图像", "GT", "P", "R"]),
            "",
            "### 置信度扫描",
            "",
            markdown_table(
                [{"conf": r["threshold"], "类别": r["class"], "P": f"{r['precision']:.3f}",
                  "R": f"{r['recall']:.3f}", "FP/图": f"{r['fp_per_image']:.2f}",
                  "框/图": f"{r['boxes_per_image']:.2f}"} for r in eval_data["sweep"]],
                ["conf", "类别", "P", "R", "FP/图", "框/图"]),
            "",
            "## 3. 曲线",
            "",
            "![metrics](curves_metrics.png)",
            "",
            "![losses](curves_losses.png)",
            "",
            "![test](test_breakdown.png)",
            "",
        ]
    lines += [
        "## 4. 局限与解读注意",
        "",
        "- 1760 张标签中约 1260 张由模型预标注生成，测试指标同时包含「与预标模型一致度」的成分，真实精度应以人工标注子集的数字为准。",
        "- 测试集中人工标注样本数量较少，分组指标（尤其 TGV_60 人工部分）置信区间很宽。",
        "- 256 张空标签图未逐张复核，若其中有漏标，会体现为虚高的 recall。",
        "",
        "## 5. 产物",
        "",
        f"- 指标明细：`{RUN.name}/results.csv`",
        f"- 评测明细：`runs/eval_test_official/breakdown.csv`、`sweep.csv`、`summary.json`",
        f"- 曲线：`{OUT.name}/curves_metrics.png`、`curves_losses.png`、`test_breakdown.png`",
    ]
    report = OUT / "report.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"best epoch {int(best['epoch'])}: mAP50={best['metrics/mAP50(B)']:.4f} "
          f"mAP50-95={best['metrics/mAP50-95(B)']:.4f}")
    if eval_data:
        print(f"test: mAP50={eval_data['official']['mAP50']:.4f} "
              f"mAP50-95={eval_data['official']['mAP50-95']:.4f}")
    print(f"report -> {report}")


if __name__ == "__main__":
    main()
