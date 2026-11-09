"""Insert chapter 7 (surface-defect AI validation) into the pre-research report."""
import copy
import re
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

DOC = Path("E:/TGV inspection detection/TGV通孔检测技术预研报告.docx")
ROOT = Path("E:/TGV inspection detection/Exp_Data/TGV_yolo_Dataset")

d = docx.Document(str(DOC))
P = d.paragraphs


def find_par(text, style=None):
    for p in d.paragraphs:
        if p.text.strip() == text and (style is None or p.style.name == style):
            return p
    raise KeyError(text)


def find_heading(prefix):
    for p in d.paragraphs:
        if p.style.name.startswith("Heading") and p.text.strip().startswith(prefix):
            return p
    raise KeyError(prefix)


# ---------------------------------------------------------------- helpers ----
def strip_bookmarks(p_el):
    for tag in ("w:bookmarkStart", "w:bookmarkEnd"):
        for el in p_el.findall(qn(tag)):
            p_el.remove(el)
    for el in p_el.findall(".//" + qn("w:lastRenderedPageBreak")):
        el.getparent().remove(el)


def clear_content(p_el):
    for child in list(p_el):
        if child.tag != qn("w:pPr"):
            p_el.remove(child)


def base_rpr(tpl_p):
    for r in tpl_p.findall(qn("w:r")):
        rpr = r.find(qn("w:rPr"))
        if rpr is not None:
            return copy.deepcopy(rpr)
    rpr = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), "Times New Roman")
    fonts.set(qn("w:hAnsi"), "Times New Roman")
    rpr.append(fonts)
    return rpr


def add_runs(p_el, segments, rpr_tpl):
    for text, bold in segments:
        r = OxmlElement("w:r")
        rpr = copy.deepcopy(rpr_tpl)
        for tag in ("w:b", "w:bCs"):
            for el in rpr.findall(qn(tag)):
                rpr.remove(el)
        if bold:
            b = OxmlElement("w:b")
            bcs = OxmlElement("w:bCs")
            idx = 1 if len(rpr) else 0
            rpr.insert(idx, b)
            rpr.insert(idx + 1, bcs)
        r.append(rpr)
        t = OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = text
        r.append(t)
        p_el.append(r)


def rich(text):
    parts = re.split(r"(\*\*.+?\*\*)", text)
    segs = []
    for s in parts:
        if not s:
            continue
        if s.startswith("**") and s.endswith("**"):
            segs.append((s[2:-2], True))
        else:
            segs.append((s, False))
    return segs


def new_par(tpl, text, style_keep=True):
    p_el = copy.deepcopy(tpl._p)
    strip_bookmarks(p_el)
    rpr = base_rpr(p_el)
    clear_content(p_el)
    add_runs(p_el, rich(text), rpr)
    return p_el


def set_par_text(p, text):
    p_el = p._p
    rpr = base_rpr(p_el)
    clear_content(p_el)
    add_runs(p_el, [(text, False)], rpr)


# ------------------------------------------------------------ templates -----
h1_tpl = find_par("1. 项目概述", "Heading 1")
h2_tpl = find_par("6.1 验证目的与依据", "Heading 2")
body_tpl = None
for p in d.paragraphs:
    if p.style.name == "Normal" and p.text.startswith("本方案的原理与韩国"):
        body_tpl = p
        break
list_tpl = None
for p in d.paragraphs:
    if p.style.name == "List Paragraph" and p.text.startswith("玻璃基板与载板制造厂"):
        list_tpl = p
        break
cap_tpl = find_par("表 6 试验样品清单与可支撑的验证内容", "Caption")
fig_tpl = find_par("图 8 数字全息断层扫描算法处理流程", "Caption")

# ----------------------------------------------------- renumber chapters ----
def rebuild_heading(p, new_text):
    set_par_text(p, new_text)


renames = [
    ("11.附件（调研资料、仿真数据、流程图、测试数据等）", "12.附件（调研资料、仿真数据、流程图、测试数据等）"),
    ("10.预研结论与落地建议", "11.预研结论与落地建议"),
    ("9.成本分析与量产经济评估", "10.成本分析与量产经济评估"),
    ("8.资源投入", "9.资源投入"),
    ("7.风险评估", "8.风险评估"),
    ("10.1 主要结论", "11.1 主要结论"),
    ("10.2 落地建议", "11.2 落地建议"),
    ("9.1 成本估算模型", "10.1 成本估算模型"),
    ("9.2 量产经济评估", "10.2 量产经济评估"),
    ("8.1 投入口径说明", "9.1 投入口径说明"),
    ("8.2 人力投入", "9.2 人力投入"),
    ("8.3 设备、机时与外部资源", "9.3 设备、机时与外部资源"),
]
for old, new in renames:
    rebuild_heading(find_par(old), new)

# ------------------------------------------------------ cross references ----
def replace_in_par(p, old, new):
    if old not in p.text:
        return False
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return True
    set_par_text(p, p.text.replace(old, new))
    return True


pairs = [
    ("第 8、9 章参数表", "第 9、10 章参数表"),
    ("第 9 章参数表", "第 10 章参数表"),
    ("第 7 章风险登记", "第 8 章风险登记"),
    ("第 7 章风险表", "第 8 章风险表"),
]
for p in d.paragraphs:
    for old, new in pairs:
        replace_in_par(p, old, new)

# ------------------------------------- 5.1 note about module scope ----------
p51 = None
for p in d.paragraphs:
    if p.text.startswith("四个模块的分工定位为："):
        p51 = p
        break
last_run = p51.runs[-1]
last_run.text = last_run.text + "需说明，上述四个模块均以通孔内径为对象，不含板面灰尘与划痕的检出；板面缺陷 AI 检测作为独立专题见第 7 章，其结论不改变本章的模块划分。"

# ------------------------------------------- tables: risk / manpower -------
def table_by_header(text):
    for t in d.tables:
        try:
            if t.rows[0].cells[0].text.strip() == text:
                return t
        except IndexError:
            continue
    raise KeyError(text)


def clone_row(table, texts, bold=False):
    tr = copy.deepcopy(table.rows[-1]._tr)
    for tc, txt in zip(tr.findall(qn("w:tc")), texts):
        for p_el in tc.findall(qn("w:p")):
            tc.remove(p_el)
        p_el = OxmlElement("w:p")
        ppr = OxmlElement("w:pPr")
        jc = OxmlElement("w:jc")
        jc.set(qn("w:val"), "center")
        ppr.append(jc)
        rpr = OxmlElement("w:rPr")
        fonts = OxmlElement("w:rFonts")
        fonts.set(qn("w:ascii"), "Times New Roman")
        fonts.set(qn("w:hAnsi"), "Times New Roman")
        rpr.append(fonts)
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), "18")
        rpr.append(sz)
        szcs = OxmlElement("w:szCs")
        szcs.set(qn("w:val"), "18")
        rpr.append(szcs)
        if bold:
            rpr.append(OxmlElement("w:b"))
        ppr.append(rpr)
        p_el.append(ppr)
        r = OxmlElement("w:r")
        r.append(copy.deepcopy(rpr))
        t = OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = txt
        r.append(t)
        p_el.append(r)
        tc.append(p_el)
    table._tbl.append(tr)
    return tr


risk = table_by_header("风险项")
clone_row(risk, ["标注质量与预标污染",
                 "留出集指标含与预标模型的一致度，召回被系统性高估；空标签漏标同样抬高召回",
                 "高",
                 "以人工标注子集（63 张）为对外口径；复核 1260 张预标与 255 张空标签后重训复评（7.3、7.5.3 节）"])
clone_row(risk, ["细划痕召回不足",
                 "TGV_10 档划痕漏检 128/266，四种配置召回封顶 0.579，无法承诺零漏检",
                 "高",
                 "补标并统一类间判定细则；产线按提示级使用并人工复检；切块推理增益待联调实测（7.5.4、7.6 节）"])

man = table_by_header("角色")
clone_row(man, ["缺陷检测算法",
                "板面灰尘与划痕的目标检测模型训练、切块推理与 ONNX 部署（第 7 章）",
                "待定", "待定"])

rev = table_by_header("日期")
new_tr = clone_row(rev, ["2026/11/9", "0.2",
                         "新增第 7 章板面灰尘与划痕 AI 缺陷检测的工程验证与可行性；原第 7—11 章顺延为第 8—12 章；风险表、人力表与结论同步补充",
                         "于綦悦"])
# insert revision row right after the first data row
first_data = rev.rows[1]._tr
first_data.addnext(new_tr)

# --------------------------------------------- conclusions / suggestions ----
concl = []
idx11 = None
for i, p in enumerate(d.paragraphs):
    if p.text.strip().startswith("11.1 主要结论"):
        idx11 = i
        break
last = d.paragraphs[idx11 + 1]
for p in d.paragraphs[idx11 + 1:]:
    if p.style.name.startswith("Heading"):
        break
    if p.style.name == "Normal":
        last = p
new_p = new_par(last, "板面缺陷 AI：以 YOLOv8s 在 1760 张板面图像上完成工程验证，灰尘召回 0.761（人工子集 0.597）达到自动判定水平，划痕召回 0.561（人工子集 0.472）定为提示级；指标上限由标注质量而非模型容量决定（第 7 章）。")
last._p.addnext(new_p)
d = docx.Document(str(DOC)) if False else d  # keep object; no reopen

# suggestions bullet
idx12 = None
for i, p in enumerate(d.paragraphs):
    if p.text.strip().startswith("11.2 落地建议"):
        idx12 = i
        break
last_list = None
for p in d.paragraphs[idx12 + 1:]:
    if p.style.name.startswith("Heading"):
        break
    if p.style.name == "List Paragraph":
        last_list = p
new_b = new_par(last_list, "标注优先：板面缺陷的投入应先用于复核 255 张空标签与 TGV_10 细划痕、统一类间标注细则，再重训复评；复标完成前，划痕只作人工复检提示，不承诺零漏检（7.7 节）。")
last_list._p.addnext(new_b)

# --------------------------------------------------------- table builder ----
def make_table(caption, header, rows, widths_cm):
    ncol = len(header)
    tbl = d.add_table(rows=len(rows) + 1, cols=ncol)
    tbl_el = tbl._tbl
    tblPr = tbl_el.tblPr
    for el in list(tblPr):
        tblPr.remove(el)
    tblW = OxmlElement("w:tblW")
    tblW.set(qn("w:w"), str(int(sum(widths_cm) * 567)))
    tblW.set(qn("w:type"), "dxa")
    tblPr.append(tblW)
    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "center")
    tblPr.append(jc)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), "000000")
        borders.append(e)
    tblPr.append(borders)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)

    grid = tbl_el.find(qn("w:tblGrid"))
    for gc in list(grid):
        grid.remove(gc)
    for w in widths_cm:
        gc = OxmlElement("w:gridCol")
        gc.set(qn("w:w"), str(int(w * 567)))
        grid.append(gc)

    def fill(cell, text, bold=False, header_row=False):
        tcPr = cell._tc.get_or_add_tcPr()
        for el in list(tcPr):
            tcPr.remove(el)
        tcW = OxmlElement("w:tcW")
        tcW.set(qn("w:w"), str(int(widths_cm[cell._tc.getparent().index(cell._tc)] * 567)))
        tcW.set(qn("w:type"), "dxa")
        tcPr.append(tcW)
        if header_row:
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), "C6D9F1")
            tcPr.append(shd)
        va = OxmlElement("w:vAlign")
        va.set(qn("w:val"), "center")
        tcPr.append(va)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        run.font.size = docx.shared.Pt(9)
        if bold:
            run.bold = True

    for j, h in enumerate(header):
        fill(tbl.rows[0].cells[j], h, bold=True, header_row=True)
    trPr = tbl.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:trHeight")
    th.set(qn("w:val"), "567")
    trPr.append(th)
    trPr.append(OxmlElement("w:tblHeader"))
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            fill(tbl.rows[i].cells[j], v)
        cs = OxmlElement("w:cantSplit")
        tbl.rows[i]._tr.get_or_add_trPr().append(cs)

    cap_el = copy.deepcopy(cap_tpl._p)
    strip_bookmarks(cap_el)
    children = list(cap_el)
    end_idx = None
    for k, ch in enumerate(children):
        if ch.tag == qn("w:r") and ch.find(".//" + qn("w:fldChar")) is not None:
            fc = ch.find(".//" + qn("w:fldChar"))
            if fc.get(qn("w:fldCharType")) == "end":
                end_idx = k
    for ch in children[end_idx + 1:]:
        cap_el.remove(ch)
    add_runs(cap_el, [(" " + caption, False)], base_rpr(cap_el))
    return tbl_el, cap_el


def make_figure(path, caption, width_cm=14.5):
    p_el = copy.deepcopy(body_tpl._p)
    strip_bookmarks(p_el)
    clear_content(p_el)
    ppr = p_el.find(qn("w:pPr"))
    for el in ppr.findall(qn("w:ind")):
        ppr.remove(el)
    jc = ppr.find(qn("w:jc"))
    if jc is None:
        jc = OxmlElement("w:jc")
        ppr.append(jc)
    jc.set(qn("w:val"), "center")
    p = docx.text.paragraph.Paragraph(p_el, body_tpl._parent)
    p.add_run().add_picture(str(path), width=Cm(width_cm))

    cap_el = copy.deepcopy(fig_tpl._p)
    strip_bookmarks(cap_el)
    children = list(cap_el)
    end_idx = None
    for k, ch in enumerate(children):
        if ch.tag == qn("w:r") and ch.find(".//" + qn("w:fldChar")) is not None:
            fc = ch.find(".//" + qn("w:fldChar"))
            if fc.get(qn("w:fldCharType")) == "end":
                end_idx = k
    for ch in children[end_idx + 1:]:
        cap_el.remove(ch)
    add_runs(cap_el, [(" " + caption, False)], base_rpr(cap_el))
    return p_el, cap_el


# ------------------------------------------------------ chapter 7 blocks -----
CH = []
def h1(t): CH.append(("h1", t))
def h2(t): CH.append(("h2", t))
def par(t): CH.append(("p", t))
def lst(t): CH.append(("l", t))
def fig(name, cap): CH.append(("fig", (name, cap)))
def tbl(cap, header, rows, widths): CH.append(("tbl", (cap, header, rows, widths)))

h1("7. 板面灰尘与划痕 AI 缺陷检测的工程验证与可行性")
par("本章为独立于通孔内径主线的专题。第 2.3 节已把表面与浅层缺陷 AOI 列为 TGV 检测种类之一，第 3.6 节把“AI 缺陷分类”列为国产替代打包能力的一环；但第 4 至 6 章的四个阶段均以通孔内径为对象，未覆盖板面缺陷的检出能力。本章以本项目现有 S1—S4 四块标称 10/50/60/80 µm 玻板（表 6）采集的板面图像为基础，验证以目标检测模型检出灰尘与划痕的可行性，给出数据口径、技术方案、验证结果与分级使用结论。需先明确边界：本章不改变第 4 章的阶段指标、第 6 章的验收口径与第 5 章的四模块划分；其验证对象是板面而非孔壁，照明、目标尺度与判据均与 4.3 节阶段二的暗场裂纹与崩边验证不同，两者不可互相替代。")

h2("7.1 专题定位与边界")
par("从竞品看，缺陷分类已是在线检测机台的标准配置：Zeus HS3000 公开的 AI 分类准确率 ≥99%，Venus 6200 采用 CV+DL 双引擎，第 3.6 节亦把“AI 缺陷分类 + MES 数据闭环”列为国产替代的打包能力之一。本项目前六个章节解决的是孔在哪里、多大、圆不圆，尚未回答板面是否洁净；板面灰尘与划痕若不能在蚀刻后工位被检出，会随种子层与电镀工序进入下道，形成外观不良与潜在的电性风险，因此本章将其作为并行专题单列。")
par("本章回答三个问题：一是在现有 2448×2048 灰度成像与 1024 输入下，两类缺陷能否被稳定检出；二是检出能力处于什么水平，哪些档位可自动判定、哪些只能提示；三是能力上限受模型、数据还是标注约束。")
par("边界与不做项：仅灰尘与划痕两类，输出矩形框、类别与置信度，不做像素级分割、不做缺陷等级评定、不做面积与深度的量值输出；样本来自 S1—S4 四块玻板的板面，不含金属化后、CMP 后与 RDL 后工序；节拍与产线集成不在本章验证范围内。")

h2("7.2 任务定义与判定口径")
par("任务定义为两类目标检测：类别 0 为灰尘，类别 1 为划痕；模型对每张图像输出若干检测框，每个框包含类别与置信度。评估采用与第 6 章一致的“留出集 + 固定工作点”口径：留出集 176 张与训练、验证互斥；置信度阈值 0.25，检测框与标注框的匹配 IoU 为 0.5；同时报告官方 mAP（IoU 0.50—0.95 平均）与该工作点的分类别精确率与召回率。")
par("为区分数据来源的影响，指标按标注来源分为人工标注子集与机器预标子集两组，按孔径分为 TGV_10/50/60/80 四组，分层报告。本章不以总体平均值作为结论依据，理由见 7.3 节。")
lst("**灰尘自动判定级：**留出集召回 ≥0.75，且人工标注子集召回 ≥0.55。达此线时灰尘可作为自动判定项，误报由后处理与人工抽检兜底。")
lst("**划痕提示级：**人工标注子集召回 ≥0.45。达此线时划痕作为人工复检的提示通道，不承诺零漏检。")
lst("**不可用级：**低于上述任一条，对应类别不进入产线判定流程。")
par("上述阈值为建议值：报告未取得客户验收线，阈值按现有实测水平与产线可接受的复检负荷设定，须在联调阶段按实际节拍与复检能力重新标定，确认前不构成合同条款。")

h2("7.3 数据与标注")
par("数据取自表 6 的 S1—S4 四块玻板，板面通孔孔径标称值分别为 10、50、60、80 µm；采集图像为 2448×2048 灰度图，共 1760 张，覆盖整板不同视场。数据按 8:1:1 切分为训练 1408 张、验证 176 张、留出 176 张，四档孔径在三个子集中保持同比例（TGV_10 为 528/66/66，TGV_50 与 TGV_80 为 240/30/30，TGV_60 为 400/50/50），避免切分引入孔径偏置。")
par("标注为两类矩形框，框数统计见表 7。留出集共 995 个标注框，其中灰尘 665、划痕 330，平均每图 5.65 个框。")
tbl("数据集与标注框统计",
    ["子集", "图像数", "灰尘框", "划痕框", "框/图", "空标签", "标注来源"],
    [["训练", "1408", "5549", "2427", "5.66", "198", "预标与人工混合"],
     ["验证", "176", "732", "307", "5.90", "34", "预标与人工混合"],
     ["留出", "176", "665", "330", "5.65", "23", "其中 113 张为预标"],
     ["合计", "1760", "6946", "3064", "5.69", "255", "预标 1260 张"]],
    [2.6, 1.9, 2.0, 2.0, 1.9, 1.8, 2.45])
par("标注来源存在结构性缺口，须在解读指标前说明：1760 张中 1260 张（71.6%）的标签由模型预标注生成，按孔径分别为 TGV_10 210 张、TGV_50 300 张、TGV_60 450 张、TGV_80 300 张，其余 500 张为纯人工标注；另有 255 张为空标签（训练 198、验证 34、留出 23），尚未逐张复核，其中可能包含未标注的细小缺陷。")
par("上述缺口带来两条口径约束。其一，留出集中 113 张属机器预标，其指标含有与预标模型一致度的成分，会系统性抬高召回；因此本章以人工标注子集（63 张）作为对外结论口径，机器预标子集仅作同源性参考。其二，空标签样本若确有漏标，会使召回被高估，该影响同样由人工子集口径抵消。此外，现行标注未规定最小标注尺寸与类间判定细则，细长灰尘与短划痕存在类间重叠，这是 7.5.7 节错类检出的部分来源。")

h2("7.4 技术方案")
h2("7.4.1 模型与训练配置")
par("采用 YOLOv8s 单阶段目标检测模型，以公开预训练权重迁移初始化，输入 1024×1024，批大小 4，训练 100 轮，早停耐心 30 轮，固定随机种子并开启确定性训练，混合精度加速，单卡 6 GB 显存可容纳。选择 s 级模型的依据不是精度上限，而是样本规模：全量仅 1760 张，更大模型在验证集上无增益（7.5.6 节的四配置对照给出同类证据）。累计训练耗时 2.09 h（含中断续训）。")
par("训练采用自动化优化器配置，最后 10 轮关闭马赛克增强，使模型在后段收敛于真实尺度分布。数据增强未做额外的几何或色彩扩展，理由是灰度板面图像的对比度分布稳定，过强增强会改变缺陷的灰度特征。")
h2("7.4.2 推理与部署")
par("单帧推理以整图 1024 为基线。产线部署按原图 2448×2048 切分为 1024×1024 的块，块间保留 20% 重叠，各块结果映射回原图坐标后统一做非极大值抑制，以避免缺陷跨块截断造成的漏检；模型导出为 ONNX，由产线上位机调用。需强调一条工程约束：推理输入分辨率必须与训练分辨率一致，1280 训练的权重按 1024 推理会同时损失小目标与定位精度，反之亦然。")
par("本章的验证结果均为整图 1024 基线口径，未对切块推理的增益做定量验证；切块会把小目标放大到有效分辨率，预计对 TGV_10 细划痕有正向作用，但会带来块边界的重复检测与误报，需在联调阶段实测后定档。")
h2("7.4.3 选型对照设计")
par("为区分模型容量不足与数据上限，设置四组对照：YOLOv8s 与 YOLO11s 两种主干，各取 1024 与 1280 两种输入分辨率，在同一数据切分、同一评估口径下比较。判读规则为：若四组的划痕召回落在同一窄区间，则召回上限归因于数据与标注；若随模型或分辨率显著提升，则归因于模型容量。")

h2("7.5 验证过程与结果")
h2("7.5.1 训练收敛")
par("训练按 100 轮跑满，最佳轮次为第 88 轮。该轮验证集 mAP50 为 0.7157、mAP50-95 为 0.4917，精确率与召回率为 0.6390 与 0.7091。验证集 mAP 在 60 轮后进入平台，末段波动小于 0.01，说明在该数据规模下继续训练不再带来实质增益（图 9）。")
fig(ROOT / "outputs/figs/chart_train.png", "训练过程验证集 mAP 随轮次的变化（最佳第 88 轮）")
h2("7.5.2 留出集总体结果")
par("留出集 176 张的官方指标为 mAP50 0.6815、mAP50-95 0.4357；工作点（conf 0.25、匹配 IoU 0.5）下，灰尘精确率 0.626、召回率 0.761，划痕精确率 0.591、召回率 0.561（表 8）。两类精确率接近，差距集中在召回，即当前的主要矛盾是划痕漏检，而不是误报。")
tbl("留出集检测结果（工作点 conf = 0.25，匹配 IoU = 0.5）",
    ["分组", "图像", "灰尘 P", "灰尘 R", "划痕 P", "划痕 R", "说明"],
    [["总体（留出集）", "176", "0.626", "0.761", "0.591", "0.561", "mAP50 0.682；mAP50-95 0.436"],
     ["人工标注子集", "63", "0.587", "0.597", "0.567", "0.472", "对外结论口径"],
     ["机器预标子集", "113", "0.651", "0.904", "0.630", "0.773", "与预标同源，仅作参考"],
     ["TGV_10", "66", "0.630", "0.626", "0.627", "0.519", "划痕 GT 266，漏检 128"],
     ["TGV_50", "30", "0.562", "0.947", "0.636", "0.636", "划痕 GT 11，样本不足"],
     ["TGV_60", "50", "0.655", "0.881", "0.481", "0.760", "划痕 GT 50"],
     ["TGV_80", "30", "0.442", "0.950", "0.667", "0.667", "划痕 GT 3，不作独立结论"]],
    [3.0, 1.4, 1.7, 1.7, 1.7, 1.7, 3.45])
h2("7.5.3 分标注来源")
par("人工标注子集 63 张：灰尘 0.587/0.597，划痕 0.567/0.472；机器预标子集 113 张：灰尘 0.651/0.904，划痕 0.630/0.773。两组差距集中在召回，与预标样本由同一模型生成的预期一致；这也说明若直接引用总体指标，会把与预标模型的一致度误读为检测能力。")
h2("7.5.4 分孔径")
par("TGV_10 档灰尘召回 0.626、划痕召回 0.519（划痕标注框 266 个，漏检 128 个）；TGV_50、TGV_60、TGV_80 档的灰尘召回分别为 0.947、0.881、0.950。划痕在 TGV_50/60/80 的标注框仅 11/50/3 个，样本量不足以支撑独立结论。数据表明召回损失高度集中在 10 µm 档的细划痕上（图 10）：该档孔径最小、缺陷尺度最小，在 1024 输入下细划痕仅数像素宽，与背景纹理的对比度也最低。")
fig(ROOT / "outputs/figs/chart_aperture.png", "留出集分孔径召回（灰尘与划痕）")
h2("7.5.5 阈值与工作点")
par("置信度阈值决定漏检与误报的交换关系。阈值由 0.05 提高到 0.50 时，划痕召回由 0.639 降至 0.403，精确率由 0.410 升至 0.689，误报由 1.73 框/图降至 0.34 框/图；灰尘的对应变化为召回 0.863 降至 0.644、精确率 0.441 升至 0.737（图 11）。本章取 0.25 为工作点：划痕召回 0.561、误报 0.73 框/图，是漏检可控且复检负荷可接受的折中。若产线以漏检为红线，可下调至 0.15；若以误报为红线，可上调至 0.30，最终档位须由复检人力与节拍共同确定。")
fig(ROOT / "outputs/figs/chart_sweep.png", "留出集划痕精确率与召回率随置信度阈值的变化（工作点 0.25）")
h2("7.5.6 四配置对照")
par("四种配置的留出集结果为：mAP50 0.658—0.689，mAP50-95 0.412—0.436，划痕召回 0.533—0.579（表 9）。提升输入分辨率到 1280 未显著提高划痕召回：YOLOv8s 由 0.561 升至 0.579，YOLO11s 由 0.533 升至 0.570，增幅均在 0.04 以内；更换主干与增大分辨率的组合也未突破 0.58。据此可以判定：当前召回上限由数据与标注决定，继续在模型侧投入的边际收益很低。")
tbl("四种模型配置的留出集对照",
    ["配置", "输入分辨率", "mAP50", "mAP50-95", "划痕召回"],
    [["YOLOv8s", "1024", "0.682", "0.436", "0.561"],
     ["YOLOv8s", "1280", "0.658", "0.412", "0.579"],
     ["YOLO11s", "1024", "0.689", "0.416", "0.533"],
     ["YOLO11s", "1280", "0.688", "0.431", "0.570"]],
    [3.6, 3.4, 2.5, 2.5, 2.65])
h2("7.5.7 混淆结构")
par("留出集 995 个标注框的归属为：正确类别检出 691 个、错判为另一类 113 个（灰尘判为划痕 45、划痕判为灰尘 68）、未检出 191 个（灰尘 114、划痕 77），整体召回 0.694（图 12）。与此同时，按类别统计的误报框为灰尘 302、划痕 128，合计 430 个，其中一部分正落在另一类真值框上，构成上述错类计数。结构上有两点判断：其一，错类占比 11.4%，低于漏检占比 19.2%，说明模型对有无缺陷的判别优于对类别的判别；其二，漏检的成因以低对比、细长与成团结构为主，与 7.5.4 节的孔径分布一致。")
fig(ROOT / "outputs/figs/chart_confusion.png", "留出集混淆统计（按标注框计，含错类与漏检）")
par("误报的分布需结合产线处置：1.7 框/图的灰尘误报与 0.7 框/图的划痕误报，意味着单张 2448×2048 图像约有 2.4 个待复检框；按整板视场数换算后的复检负荷，须在联调阶段以实际视场数核算，本章不给出面板级结论。")

h2("7.6 可行性判定")
lst("**灰尘：可行，具备自动判定条件。**留出集召回 0.761，TGV_50/60/80 档达 0.881—0.950；人工标注子集 0.597，仍高于 0.55 的判定线。以 0.25 工作点配合 1.7 框/图的误报，可由后处理与人工抽检兜底。")
lst("**划痕：有条件可行，定位为提示级。**人工标注子集召回 0.472，高于 0.45 的判定线；但 TGV_10 档仅 0.519，四种配置封顶 0.579，不足以承诺零漏检，只能作为人工复检的提示通道。")
lst("**瓶颈归因：**其一为标注质量，71.6% 为预标、255 张空标签未复核、类间细则缺失；其二为目标尺度，10 µm 档细划痕在 1024 输入下仅数像素宽；其三为成像对比度，灰尘与划痕在现有明场下均属低对比目标。三者中标注质量是唯一可在短期内改善的环节。")
par("与竞品的可比性必须限定：Zeus HS3000 的“AI 分类 ≥99%”是分类准确率口径，样本构成、类别体系与验收条件均未公开；本章给出的是检出召回与精确率，两者不可直接比较，本章不将其作为对标结论。")

h2("7.7 结论与后续")
lst("**结论一：**以 YOLOv8s 与现有 1760 张数据，板面灰尘检测达到自动判定水平，划痕检测达到提示级水平，板面缺陷 AI 检测在工程上可行。")
lst("**结论二：**指标上限由数据与标注决定，不由模型容量决定；四种配置的划痕召回集中在 0.53—0.58，换模型或提高分辨率的收益有限。")
lst("**结论三：**本章结论与通孔内径主线相互独立，不改变第 4 章的阶段指标、第 6 章的验收口径与第 5 章的四模块划分；板面检测与阶段二暗场通道共用同一成像底座的可能性存在，但需另行验证。")
par("后续动作按优先级排列：其一，复核 255 张空标签与 TGV_10 档细划痕标注，补全漏标与错标；其二，补充类间判定细则与最小标注尺寸，统一灰尘与划痕的边界；其三，按统一规范重训并在同一留出集上复评，确认召回提升来自标注而非随机波动；其四，与 4.3 节阶段二的暗场通道在同批样件上联测，确认划痕在两套通道上的一致性；其五，产线按灰尘自动判定、划痕人工复检分级使用，待复标完成后重估自动化比例。")

# ------------------------------------------------------------ assemble -----
anchor = find_heading("8.风险评估")
for kind, payload in CH:
    if kind == "h1":
        el = new_par(h1_tpl, payload)
    elif kind == "h2":
        el = new_par(h2_tpl, payload)
    elif kind == "p":
        el = new_par(body_tpl, payload)
    elif kind == "l":
        el = new_par(list_tpl, payload)
    elif kind == "fig":
        img, cap = payload
        img_el, cap_el = make_figure(img, cap)
        anchor._p.addprevious(img_el)
        anchor._p.addprevious(cap_el)
        continue
    elif kind == "tbl":
        cap, header, rows, widths = payload
        tbl_el, cap_el = make_table(cap, header, rows, widths)
        anchor._p.addprevious(cap_el)
        anchor._p.addprevious(tbl_el)
        continue
    anchor._p.addprevious(el)

d.save(str(DOC))
print("saved", DOC.stat().st_size)
