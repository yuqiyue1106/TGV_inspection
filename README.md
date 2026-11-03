# TGV 缺陷检测 — YOLOv8 训练 / 推理代码

同一份代码在两台机器上跑：

- **Windows 11 + NVIDIA GPU**（CUDA，主力训练机）
- **MacBook Air M4**（Apple Silicon，`mps` 或纯 `cpu`，复核 / 预标注 / 小规模微调）

平台差异全部收敛到 `device.py` 和 `requirements-*.txt` 两处，脚本本身不带任何
`if windows:` 之类的分支。

---

## 1. 该文件在两台机器上的对应关系

| | Windows 11 + NVIDIA | MacBook Air M4（`--device mps`） | MacBook Air M4（`--device cpu`） |
|---|---|---|---|
| 依赖清单 | `requirements-cuda.txt` | `requirements-mac.txt` | `requirements-mac.txt` |
| `--device` | `auto` / `0` | `auto` / `mps` | `auto` / `cpu` |
| batch（未显式指定时） | 4 | 2 | 2 |
| workers | 0 | ≤ 4 | ≤ 8 |
| cache | ram | disk | disk |
| AMP 混合精度 | 开 | 关（MPS 不支持） | 关 |
| `PYTORCH_ENABLE_MPS_FALLBACK` | 不需要 | `device.py` 自动设为 1 | 同左 |
| `imgsz` | **1024，三档一致** | 1024 | 1024 |

`imgsz` 不参与自动降级：`runs/` 里所有 checkpoint 都是 1024 训出来的，偷偷改小会让两台
机器的指标无法对比。内存不够请降 `batch`，或显式传 `--imgsz 640` 并接受结果不可比。

---

## 2. Windows 11 + NVIDIA：首次搭建

```powershell
cd "e:\TGV inspection detection\Exp_Data\TGV_yolo_Dataset\code"

uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install --index-strategy unsafe-best-match -r requirements-cuda.txt

# 必须看到 True 和显卡名，否则装成了 PyPI 的 CPU 版 torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python train.py --device 0 --smoke     # 3 epoch 冒烟测试
```

`cu130` 对应 CUDA 13，已放弃 Maxwell / Pascal / Volta（算力 < 7.5）。若你的卡是
GTX 10 系或 V100，把 `requirements-cuda.txt` 里的 `cu130` 全部改成 `cu126`。

---

## 3. MacBook Air M4：首次搭建

```bash
git clone <repo> code && cd code

python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt

# 只想要 CPU 就把 mps 换成 cpu，依赖完全一样
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"

# 拷来的数据放到 code 的上一级目录，然后重建 split（见第 4 节）
python prepare_pc.py

python train.py --device mps --smoke
```

注意两点：

- Mac 上 **不要传 `--device 0`**，数字索引的语义就是 CUDA 设备号，会直接失败。用
  `auto` / `mps` / `cpu`。
- `device.py` 已经在 import torch 之前设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`，所以不需要
  再手动 export 环境变量。这也是所有脚本都必须把 `import device` 写在
  `import ultralytics` / `import torch` 之前的原因。

---

## 4. 数据与权重**不进 Git**

`.gitignore` 已排除 `dataset/`、`images/`、`labels/`、`weights/`、`runs/`、`*.pt`、
`*.jpg`。仓库里只有代码和依赖清单。

| 内容 | 怎么到另一台机器 |
|---|---|
| `images/` + `labels/`（1760 图 / 1762 标注） | 移动硬盘或网盘，放到 `code/` 的**上一级** |
| `dataset/`（train/val/test 切分） | 不拷。到目标机器跑 `python prepare_pc.py` 重建 |
| `weights/yolov8s.pt` 等预训练权重 | GitHub Release 附件；缺了也行，`train.py` 会让 ultralytics 自动下载 |
| `runs/<run>/weights/best.pt` | GitHub Release 附件（22 MB 级别，不值得上 Git LFS） |
| `runs/` 的曲线图、`results.csv` | 不共享。每台机器自己的训练时序没有可比性 |

`dataset/data.yaml` 里是**绝对路径**，这是 ultralytics 的解析规则决定的：相对 `path`
会被它拿到 `settings["datasets_dir"]`（默认 `~/datasets`）底下拼，而不是相对
`data.yaml` 所在目录。所以换机器后 `prepare_pc.py` 必须跑，它会重写这个字段并清掉
过期的 `labels/*.cache`。

---

## 5. 常用命令

```bash
# 训练（batch/workers/cache 自动按后端取预设，显式传参则覆盖）
python train.py --device auto --name tgv_v8s_1024 --epochs 200
python supervise.py --device auto --run official_v8s_1024     # 守护 + 自动接评估

# 推理 / 预标注（全分辨率走 tiling）
python predict.py --device auto --model runs/official_v8s_1024/weights/best.pt \
    --source ../images/TGV_60 --tiled --labels-out ../labels --skip-existing

# 评估 + 置信度阈值扫描
python evaluate.py --device auto --model runs/official_v8s_1024/weights/best.pt
python eval_test.py --device auto --model runs/official_v8s_1024/weights/best.pt

# 训练进度 / 指标导出 / TensorBoard
python progress.py
python export_metrics.py
python tb_bridge.py --all
python export.py --format onnx          # Mac 上不要 --format engine，也不要 --half
```

---

## 6. 跨机器同步的硬性约束

1. **`ultralytics` 必须两台机器同一个版本**（当前 pin 在 `8.4.184`）。
   `eval_test.py:relax_nms_time_limit()` 会往 `utils.nms.non_max_suppression` 注入
   `max_time_img=` 关键字，这个参数名随版本变，版本不一致时 `eval_test.py` 会抛
   `TypeError` 且栈顶看不出原因。
2. **Python 版本保持一致**（3.12），否则 `numpy==` / `polars-runtime-32==` 这类
   精确 pin 可能在 arm64 上找不到轮子。真遇到了就单独放开那一行的版本号，不要动 torch。
3. **不要用 iCloud / OneDrive 同步整个目录**。`.venv` 和 `runs/` 虽然被 Git 忽略，但
   网盘会把 Windows 的 `.venv` 原样推到 Mac 上，得到一坨不能用的二进制。
4. **不要手工拷 `runs/` 到另一台机器**，`args.yaml` 里的 device、cache 路径都是本机相关
   的，`supervise.py` 续跑时会照着错误的路径找 checkpoint。

---

## 7. 第一次把这套改动推上去

```powershell
cd "e:\TGV inspection detection\Exp_Data\TGV_yolo_Dataset\code"
git config core.autocrlf false        # 换行符交给 .gitattributes，不交给 git config
git add --renormalize .               # 让 .gitattributes 对已有文件生效
git status --short                    # 必须能看到 requirements-cuda.txt / requirements-mac.txt / .gitattributes
git check-ignore -v requirements-mac.txt   # 期望无输出；有输出说明 *.txt 又在吃文件
git commit -m "Add per-machine requirement pins and unify device selection"
git push -u origin main
```
