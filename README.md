# TGV 缺陷检测 — YOLOv8 训练 / 推理代码

同一份代码在两台机器上跑：

- **Windows 11 + NVIDIA GPU**（CUDA，主力训练机）
- **MacBook Air M4**（Apple Silicon，`mps` 或纯 `cpu`，复核 / 预标注 / 小规模微调）

平台差异全部收敛到 `device.py` 和 `requirements-*.txt` 两处，脚本本身不带任何
`if windows:` 之类的分支。

**第一次上手请直接看 §0。** 装环境见 §2（Windows）或 §3（Mac），拷数据和重建切分见 §4。

---

## 0. 快速上手：训练 → 验证 → 报告（新手路线）

照着本节从上到下抄命令即可，不需要先读懂其余章节。全部命令在 **cmd**（不是 PowerShell）
里执行；每开一个新窗口都要先做 §0.1 的两行。

### 0.1 进入代码目录并激活环境

```bat
cd /d "E:\TGV inspection detection\Exp_Data\TGV_yolo_Dataset\code"
.venv\Scripts\activate
```

激活成功的标志：命令行提示符前面出现 `(.venv)`。若这一步就失败，说明环境还没装好，
先回到 §2。

### 0.2 第一步：冒烟测试（约 3 分钟，先确认整条链路是通的）

```bat
python train.py --name gpu_smoke --smoke --batch 4 --workers 8 --cache false
```

- `--smoke` 会把 epoch 强制压到 3，并**自动在名字后追加 `_smoke`**，所以产物在
  **`runs\gpu_smoke_smoke\`**（不是 `runs\gpu_smoke`）。
- 判定成功：屏幕最后打印 `best checkpoint: ...\runs\gpu_smoke_smoke\weights\best.pt`，
  且 `runs\gpu_smoke_smoke\results.csv` 有 3 行。
- 这一步失败就别开正式训练：常见原因是没装成 CUDA 版 torch（回 §2 用
  `python -c "import torch; print(torch.cuda.is_available())"` 检查）或 `dataset\data.yaml`
  路径过期（回 §4 跑 `python prepare_pc.py`）。

### 0.3 第二步：正式训练（100 epoch）

```bat
python train.py --name gpu_v8s_1024 --epochs 100 --batch 4 --imgsz 1024 --workers 8 --cache false --patience 30 --tensorboard
```

用新名字 `gpu_v8s_1024` 是为了**不覆盖** Mac 上已有的 `official_v8s_1024`，两台机器的
结果才能并排比较。

本机（Windows + NVIDIA）实测参数取舍：

| 设置 | 实测结果 | 建议 |
|---|---|---|
| `--batch 4` | 5.2 it/s，115 s/epoch，显存峰值 4094/6144 MiB，100 epoch 约 3.2 h（Mac 同配置约 12.3 h） | **默认就用它** |
| `--batch 8` | 只比 batch 4 快约 8%，显存峰值 5739/6144 MiB | 除非先关掉微信/浏览器等桌面程序，否则容易 OOM，不建议 |
| `--cache ram` | 数据集约 20 GB，64 GB 内存放得下，但瓶颈是 70 W 的 GPU，速度没有提升 | 可保持 `false` |
| `--patience 30` | mAP 连续 30 epoch 不涨就早停 | 避免跑满 100 epoch 浪费时间 |
| `--tensorboard` | 事件文件写到 run 目录 | 需要看曲线时另开窗口 `tensorboard --logdir runs` |

产物：`runs\gpu_v8s_1024\weights\best.pt`、`last.pt`、`results.csv`、曲线图和
TensorBoard 事件。

上面这条命令用的是默认的 V8s。要换成 YOLO11 / YOLO26，见 §5.1（换模型就是换 `--model`）。

训练中断了要接着跑、和训练跑完了想再多训几十个 epoch，是两件不同的事，命令和效果都不同，
见 §0.4。

### 0.4 续训与加练：`--resume` 还是从 `last.pt` 新开一段

先按情况选：

| 你的情况 | 用哪种 | 结果 |
|---|---|---|
| 进程崩了 / 被关了，原计划仍是 100 epoch | **A. `--resume` 断点续训** | 从断点 epoch 接着跑到 100 |
| 100 epoch 已正常跑完，想再补 50 epoch | **B. 从 `last.pt` 新开一段** | 在已有权重上重新走 50 epoch 的调度 |
| 不想动原 run 的任何记录 | **B. 从 `last.pt` 新开一段** | 写到新目录，原 run 不受影响 |

#### 方法 A：`--resume` 断点续训（只接断点，不能加 epoch）

```bat
python train.py --model runs\gpu_v8s_1024\weights\last.pt --name gpu_v8s_1024 --resume
```

- 恢复的不只是权重，还有 `train_args`（`epochs`、`lr0`、优化器、`patience`、`name`、
  `save_dir`、数据路径）、优化器状态、EMA、`best_fitness` 和当前 epoch。
- **epoch 总数不能改。** `epochs` 不在允许覆盖的白名单里（白名单只有 `imgsz`、`batch`、
  `device`、`workers`、`cache`、`patience`、`close_mosaic`、`save_period`、`time`、`freeze`、
  `val`、`plots` 等），传 `--epochs 150` 会被忽略，日志打印
  `Resume ignores ['epochs'], using checkpoint values`。`--name`、`--lr0`、`--optimizer`、
  增强参数同样改不了。
- 该 run 已经跑满 100 时，续训会在 `assert 0 < start_epoch < self.epochs` 处直接失败。
- 产物写回**原 run 目录**，`results.csv` 是**追加**（`resume=True` 时 ultralytics 不删旧文件），
  曲线接着原来的往后画。

#### 方法 B：从 `last.pt` 新开一段（延长训练 / 加练）

```bat
python train.py --model runs\gpu_v8s_1024\weights\last.pt --name gpu_v8s_1024_ft50 --epochs 50 --batch 4 --imgsz 1024 --workers 8 --cache false --patience 30 --tensorboard
```

- **不带 `--resume`**，`--epochs N` 就是这一段实际要跑的 epoch 数；想加 50 就写 50。
- 只加载权重，**不加载**优化器状态和学习率进度：epoch 从 1 重新计数，重新 warmup，
  按新的 50 epoch 重新走余弦衰减。本质是「接着已有权重做微调」，不是严格连续的第 101–150 轮。
- 超参数全部可改。想更稳可加 `--optimizer AdamW --lr0 0.001` 走低学习率；注意
  `optimizer=auto` 时 `lr0` 不生效（见 `train.py:44-46`）。
- **必须换新名字。** `runs\<name>` 已存在时 `train.py` 走 `exist_ok=True`，ultralytics 会
  **删掉该目录下旧的 `results.csv`** 并覆盖 `weights/` 与曲线；用 `gpu_v8s_1024_ft50` 这类
  新名字，原 run 才不会被破坏。
- 起训权重用 `last.pt`（最后一轮）或 `best.pt`（验证集最优）都行；加练更推荐 `best.pt`，
  避免从可能已经过拟合的末轮接着练。

#### 两者的区别

| 维度 | A. `--resume` 断点续训 | B. 从 `last.pt` 新开一段 |
|---|---|---|
| 适用场景 | 进程中断，原计划 epoch 不变 | 原计划已跑完，要增加 epoch |
| 总 epoch | 固定为 checkpoint 里的原值（如 100） | 由 `--epochs` 决定（如 50） |
| epoch 计数 | 从断点 epoch+1 接着数 | 从 1 重新数 |
| 优化器 / 学习率 | 恢复优化器状态，lr 接着原曲线 | 重新初始化，重新 warmup、重走衰减 |
| 能改的超参 | 仅白名单内的几项 | 几乎全部 |
| 写入目录 | 原 run（`save_dir` 取自 checkpoint） | `--name` 指定的新目录 |
| `results.csv` | 追加，曲线连续 | 新文件，从 0 开始 |
| 原 run 数据 | 被继续写入 | 完全不动 |
| run 已跑满时 | 直接报错 | 正常工作 |
| 最终选模型 | 仍看原 run 的 `best.pt` | 两段的 `best.pt` 要分别评估后比较 |

#### 注意

- **不要和正在跑的训练并发。** 两个进程写同一个 run 会互相覆盖，batch 4 也已占约 2/3 显存。
- 加练那一段的 `best.pt` 只代表「新 50 epoch 内最好」，不等于整体最好。选型时把两段的
  `best.pt` 分别用 §0.5 的 `evaluate.py` / `eval_test.py` 跑一遍再比。
- `report.py` 按 `results.csv` 的 `time` 列算累计时长，断点续训时 `time` 会归零，脚本已按
  「分段求和」处理；新开一段是独立目录，各自出报告即可。

### 0.5 第三步：验证（val 阈值扫描 + test 留出集）

先在 val 集（参与过训练的验证集）上扫置信度阈值，拿到每一类推荐的 `conf`：

```bat
python evaluate.py --model runs\gpu_v8s_1024\weights\best.pt --device auto --out runs\eval_gpu_v8s_1024
```

产物 `runs\eval_gpu_v8s_1024\threshold_report.csv` 与 `summary.json`，后者里的
`recommended` 给出「recall ≥ 0.95 前提下 precision 最高」的阈值，产线用哪个 `conf` 看这里。

再在留出的 test 集（没参与训练的测试集）上按口径出明细（人工标注 / 机器预标分开，另按孔径分组）：

```bat
python eval_test.py --model runs\gpu_v8s_1024\weights\best.pt --out runs\eval_test_gpu_v8s_1024
```

产物 `runs\eval_test_gpu_v8s_1024\summary.json`（含官方 mAP50 / mAP50-95）、
`breakdown.csv`、`sweep.csv`。

### 0.6 第四步：汇总成 Markdown 报告

```bat
python report.py --run runs\gpu_v8s_1024 --eval runs\eval_test_gpu_v8s_1024\summary.json
```

产物在 `runs\gpu_v8s_1024\report\`：

| 文件 | 内容 |
|---|---|
| `report.md` | 结论、test 集官方指标与逐类 P/R、按孔径/标签来源分组、置信度扫描、局限说明 |
| `curves_metrics.png` | mAP50-95 / mAP50 / P / R 随 epoch 变化，标出最佳 epoch |
| `curves_losses.png` | train/val 三类损失与学习率，标出 mosaic 关闭区间 |
| `test_breakdown.png` | test 集 P/R、按孔径与标签来源的 recall |

两个前提：`--run` 目录里必须有非空的 `results.csv`（至少训练完 1 个 epoch）；
`--eval` 指向的 `summary.json` 不存在时报告仍会生成，但会缺 §2 和 test 图。

和 Mac 基线对比时，把 `--run` / `--eval` 换成基线路径即可：

```bat
python report.py --run runs\official_v8s_1024 --eval runs\eval_test_official\summary.json
```

### 0.7 第五步（可选）：跑图 / 预标注

```bat
python predict.py --model runs\gpu_v8s_1024\weights\best.pt --source ..\images --tiled
```

全分辨率图必须带 `--tiled`（默认 1024 切片、20% 重叠）。带标注的预览图和每图 `.txt`
写到 `runs\predict\`；要给 LabelImg 预标注，加 `--labels-out ..\labels --skip-existing`。

### 0.8 验收清单

| 检查项 | 期望结果 |
|---|---|
| 冒烟测试 | `runs\gpu_smoke_smoke\results.csv` 有 3 行，`weights\best.pt` 存在 |
| 正式训练 | 跑满 100 epoch 或被 `patience` 早停，`runs\gpu_v8s_1024\weights\best.pt` 存在 |
| val 扫描 | `runs\eval_gpu_v8s_1024\summary.json` 的 `recommended` 每类都有 `threshold` |
| test 评估 | `runs\eval_test_gpu_v8s_1024\summary.json` 有 `official` 与 `breakdown` |
| 总结报告 | `runs\gpu_v8s_1024\report\report.md` 五节齐全，三张 PNG 都能打开 |

单机走完全流程约 3.5 小时（0.1 h 冒烟 + 3.2 h 训练 + 约 0.2 h 两次评估）。
训练和评估**不要同时跑**：batch 4 已占用约 2/3 显存，叠一个进程基本必然 OOM。

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

# 用 uv 建环境：与 §2 Windows 同一套工具链，也绕开系统 python3.12 可能不在 PATH 的问题
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements-mac.txt

# 只想要 CPU 就把 mps 换成 cpu，依赖完全一样
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"

# 拷来的数据放到 code 的上一级目录，然后重建 split（见第 4 节）
python prepare_pc.py

python train.py --device mps --smoke
```

注意三点：

- Mac 上 **不要传 `--device 0`**，数字索引的语义就是 CUDA 设备号，会直接失败。用
  `auto` / `mps` / `cpu`。
- `uv venv` 默认**不往环境里装 pip**：激活后 `which pip` 会是 not found，`which pip3`
  会命中 `/usr/bin/pip3`（系统 Python 3.9）。装依赖请一律用 `uv pip install ...`，
  **不要退回 `pip`/`pip3`**，否则会用到系统 3.9，触发 `absl-py==2.5.1` 这类精确 pin
  的“找不到版本”报错。
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

### 5.1 换模型版本（YOLOv8 / YOLO11 / YOLO26）

**没有版本开关，换模型就是换 `--model` 指向的权重文件。** 代码里没有任何
`if v8 / if v11` 分支，`train.py` 也不知道自己训的是哪一代。

原理：`.pt` 文件内部自带一份 `model.yaml`，记录着网络结构。`YOLO(path)` 先照这份 yaml
把骨架搭出来，再把权重灌进去。所以**架构由权重文件决定，不由代码决定**：

| `--model` 传什么 | 会发生什么 |
|---|---|
| `runs\<run>\weights\best.pt`（本项目训出来的） | 按 checkpoint 内嵌结构重建 → 加载你的权重 |
| `weights/yolov8s.pt`（官方预训练） | 官方 V8s，结构同样内嵌在文件里 |
| `weights/yolo11s.pt` | 官方 YOLO11s |
| `yolov8s.yaml` / `yolo11s.yaml` | 只有结构、随机初始化，从头训 |

本机 `weights\` 现有 `yolov8n.pt`、`yolov8s.pt`、`yolo11s.pt`、`yolo26n.pt`。缺的模型
（如 `yolo26s.pt`）在第一次 `YOLO(...)` 时由 ultralytics 自动下载。该目录被 Git 忽略，
换机器后为空也能跑，只是会多一次下载。

#### 各版本的起始命令

```bat
:: V8s —— 默认就是它，不传 --model 即走 weights\yolov8s.pt
python train.py --name gpu_v8s_1024 --epochs 100 --batch 4 --imgsz 1024 --workers 8 --cache false --patience 30 --tensorboard

:: 换版本时照抄这一行，只改 --model / --name / --imgsz / --batch
python train.py --model weights\yolov8s.pt --name gpu_v8s_1024 --epochs 100 --batch 4 --imgsz 1024 --workers 8 --cache false --patience 30 --tensorboard

:: YOLO11s —— 本机 1280 实测：100 epoch / 3.81 h / 显存峰值约 3.6 GB
python train.py --model weights\yolo11s.pt --name gpu_y11s_1280 --epochs 100 --batch 2 --imgsz 1280 --workers 8 --cache false --patience 25 --tensorboard

:: YOLO26s —— 8.4.184 支持 n/s/m/l/x，首次运行自动下载
python train.py --model yolo26s.pt --name gpu_y26s_1280 --epochs 100 --batch 2 --imgsz 1280 --workers 8 --cache false --patience 25 --tensorboard
```

换版本时必须一起改的有三件事：

1. **`--name` 必须换新名字。** `runs\<name>` 已存在时 `train.py` 走 `exist_ok=True`，
   ultralytics 会删掉旧的 `results.csv` 并覆盖 `weights\` 与曲线——跨版本复用同一个名字
   等于把原 run 的记录毁掉。
2. **`--imgsz` 训练与推理必须同档。** 实测代价：1024 训的 V8s 拿 1280 推理，mAP50-95 从
   0.4357 掉到 0.4116（−0.024）；1280 训的 YOLO11s 拿 1024 推理，从 0.4307 掉到 0.4157
   （−0.015），且先丢的是细划痕。`evaluate.py` / `eval_test.py` / `predict.py` / `export.py`
   的 `--imgsz` 都是独立参数，**不会**自动跟随 checkpoint。
3. **`--batch` 按分辨率给。** 1024 用 4（6 GB 卡峰值 4094 MiB），1280 用 2（峰值约 3.6 GB）。

评估、推理、导出**不需要任何版本参数**，只给 `.pt` 路径（分辨率仍要自己带上）：

```bat
python evaluate.py --model runs\gpu_v8s_1024\weights\best.pt --imgsz 1024
python eval_test.py --model runs\gpu_v8s_1024\weights\best.pt --imgsz 1024
python predict.py  --model runs\gpu_v8s_1024\weights\best.pt --source ..\images\TGV_60 --tiled
python export.py   --model runs\gpu_v8s_1024\weights\best.pt --format onnx
```

#### 不要做的事

- **不要改 checkpoint 的文件名或 yaml 来“换版本”。** 架构和权重的键名是绑死的，
  V8s 的权重灌进 YOLO11 的骨架会直接报 state_dict 不匹配。换版本只能拿对应的官方
  `.pt` / `.yaml` 重新训。
- **不要只换 `--model` 而沿用旧的 `--name`**（原因见上面第 1 条）。
- **不要拿两个版本的绝对值直接下结论。** 本仓库唯一的跨版本对照（V8s@1024 vs
  YOLO11s@1280）是 mAP50-95 0.4357 对 0.4307，差 0.005，在 176 张测试图上属于噪声；
  真正的差异只有假框数量（同分辨率下 YOLO11s 少 12–28%，取决于类别与分辨率），代价是
  训练 1.8×、推理算力 1.56×。
  换版本前先按 §0.2 冒烟，再按 §0.5 评估，别只看训练日志里的 mAP。

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
