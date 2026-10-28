"""Build the Chinese Assignment 1 report from actual measurements and logs."""
from html import escape
import json
import re
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT.parent / "23300240009-佟铭洋"
DEST.mkdir(exist_ok=True)
DATA = json.loads((ROOT / "local_results/summary.json").read_text())
pdfmetrics.registerFont(TTFont("Song", "/System/Library/Fonts/Supplemental/Songti.ttc", subfontIndex=0))
BODY_FONT = "FangSong"
BODY_FONT_PATH = Path("/Users/tmy_sam/Library/Fonts/仿宋_GB18030.ttf")
pdfmetrics.registerFont(TTFont(BODY_FONT, str(BODY_FONT_PATH)))
MATH_FONT = "TNR"
MATH_FONT_PATH = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
pdfmetrics.registerFont(TTFont(MATH_FONT, str(MATH_FONT_PATH)))
BODY_FALLBACK_CHARS = {"²", "ŷ", "₂"}
INK = colors.HexColor("#1C3045")
BLUE = colors.HexColor("#215E84")
PALE = colors.HexColor("#EDF4F8")
GRAY = colors.HexColor("#5A6671")
STYLE = ParagraphStyle("body", fontName=BODY_FONT, fontSize=10.5, leading=16,
                       textColor=INK, wordWrap="CJK", spaceAfter=8)
SMALL = ParagraphStyle("small", parent=STYLE, fontSize=9, leading=13)
CELL = ParagraphStyle("cell", parent=STYLE, fontSize=9.5, leading=14)
CV = canvas.Canvas(str(DEST / "report.pdf"), pagesize=A4)
CV.setTitle("Assignment 1 - 23300240009 - 佟铭洋")
CV.setAuthor("佟铭洋")
CV.setSubject("Assignment 1")
PAGE = 0
W = H = Y = 0
M = 42


def font_fallback(text):
    """Use TNR inside <math>...</math>; preserve FangSong for body prose."""
    parts = re.split(r"(<math>.*?</math>)", str(text), flags=re.DOTALL)
    rendered = []
    for part in parts:
        if part.startswith("<math>") and part.endswith("</math>"):
            formula = escape(part[6:-7]).replace("₂", "<sub>2</sub>")
            formula = re.sub(r"\b([pwNxz])_([cLRj])\b", r"\1<sub>\2</sub>", formula)
            rendered.append(f'<font name="{MATH_FONT}">{formula}</font>')
        else:
            rendered.append("".join(f'<font name="{MATH_FONT}">'
                                    + ("<sub>2</sub>" if char == "₂" else char) + "</font>"
                                    if char in BODY_FALLBACK_CHARS else escape(char)
                                    for char in part))
    return "".join(rendered)


def paragraph(text, style=STYLE, gap=8):
    global Y
    p = Paragraph(font_fallback(text), style)
    _, height = p.wrap(W - 2 * M, H)
    if Y - height < 42:
        raise ValueError(f"Page {PAGE} overflow: {text[:60]}")
    p.drawOn(CV, M, Y - height)
    Y -= height + gap


def heading(text):
    paragraph(text, ParagraphStyle("heading", parent=STYLE, fontName="Song", fontSize=12.5,
                                   leading=19, textColor=BLUE), gap=7)


def new_page(title, wide=False, subtitle=""):
    global PAGE, W, H, Y
    if PAGE:
        CV.showPage()
    PAGE += 1
    W, H = landscape(A4) if wide else A4
    CV.setPageSize((W, H))
    CV.setFillColor(GRAY)
    CV.setFont("Song", 9)
    CV.drawString(M, H - 26, " Assignment 1")
    CV.drawRightString(W - M, H - 26, "23300240009 / 佟铭洋")
    CV.setStrokeColor(colors.HexColor("#CBD8E2"))
    CV.line(M, H - 35, W - M, H - 35)
    CV.setFont("Song", 19)
    CV.setFillColor(INK)
    CV.drawString(M, H - 65, title)
    CV.setFont("Song", 8.5)
    CV.setFillColor(GRAY)
    CV.drawString(M, 22, "2026-10")
    CV.drawRightString(W - M, 22, str(PAGE))
    Y = H - 84
    if subtitle:
        paragraph(subtitle, SMALL, gap=12)


def table(rows, widths):
    global Y
    cells = [[Paragraph(font_fallback(x), CELL) for x in row] for row in rows]
    t = Table(cells, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FA")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#B5C8D7")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#CBD8E2")),
    ]))
    _, height = t.wrap(W - 2 * M, H)
    if Y - height < 42:
        raise ValueError(f"Table overflow on page {PAGE}")
    t.drawOn(CV, M, Y - height)
    Y -= height + 12


def log(text, size=9, leading=13):
    global Y
    lines = text.strip().splitlines()
    height = len(lines) * leading + 16
    if Y - height < 42:
        raise ValueError(f"Log overflow on page {PAGE}")
    CV.setFillColor(colors.HexColor("#F4F7F9"))
    CV.roundRect(M, Y - height, W - 2 * M, height, 5, stroke=0, fill=1)
    CV.setFillColor(INK)
    CV.setFont(BODY_FONT, size)
    for i, line in enumerate(lines):
        if pdfmetrics.stringWidth(line, BODY_FONT, size) > W - 2 * M - 20:
            raise ValueError(f"Log line too wide: {line}")
        CV.drawString(M + 10, Y - 13 - i * leading, line)
    Y -= height + 12


def image_at(path, x, top, width):
    with Image.open(path) as im:
        height = width * im.height / im.width
    CV.drawImage(str(path), x, top - height, width=width, height=height)
    return height


new_page("Part 1 分类评测指标", subtitle="1. 指标定义及使用场景")
table([
    ["指标", "含义", "使用场景"],
    ["Accuracy", "预测正确的样本占全部样本的比例：正确数除以 N。", "分类整体表现；在类别较均衡、各种错误代价接近时比较适合。类别失衡时不太好（比如预测雨天，如果下雨天数本来就很少）。"],
    ["MSE", "预测值与真实值的差先平方，再对样本取平均：<math>Σ(y - ŷ)² / N</math>。较大的误差受到更大惩罚。", "回归或连续数值预测。不宜用作一般分类指标。"],
    ["Precision", "预测为正类的样本中真正为正类的比例（真阳性在所有阳性样本中所占的比例）：<math>TP / (TP + FP)</math>。", "希望减少误报的任务。"],
    ["Recall", "真实正类样本中被识别出来的比例：<math>TP / (TP + FN)</math>。", "希望减少漏报的任务。"],
    ["F1", "Precision 与 Recall 的调和平均：<math>2PR / (P + R)</math>，等价于 <math>2TP / (2TP + FP + FN)</math>。", "需要同时考虑误报和漏报，尤其关注正类表现时。"],
], [65, 223, W - 2 * M - 288])
# heading("指标解释")
# paragraph("Accuracy 越高表示整体预测越准确；MSE 越低表示数值误差越小。Precision 和 Recall 分别关注预测正类与真实正类两个不同的集合，因此不能互相替代。F1 的调和平均会受到较低一项的明显限制。")
# paragraph("数学上，Precision、Recall 或 F1 的分母为零时可能没有定义。代码按题目提供的 <math>EPS = 1e-12</math> 加到分母上，此类情况返回 0；正常样例中的微小数值偏差不影响指定容差下的测试。", SMALL)

new_page("Part 1 计算过程与测试", subtitle="")
table([
    ["混淆矩阵", "预测为 1", "预测为 0"],
    ["真实为 1", "<math>TP = 2</math>", "<math>FN = 4</math>"],
    ["真实为 0", "<math>FP = 1</math>", "<math>TN = 2</math>"],
], [95, 185, W - 2 * M - 280])
paragraph("四项计数之和为 <math>2 + 1 + 4 + 2 = 9</math>。预测正确的是 TP 与 TN，共 4 个样本。")
table([
    ["指标", "代入及计算", "结果"],
    ["Accuracy", "<math>(TP + TN) / 9 = (2 + 2) / 9</math>", "<math>4/9 ≈ 0.4444</math>"],
    ["Precision", "<math>TP / (TP + FP) = 2 / (2 + 1)</math>", "<math>2/3 ≈ 0.6667</math>"],
    ["Recall", "<math>TP / (TP + FN) = 2 / (2 + 4)</math>", "<math>1/3 ≈ 0.3333</math>"],
    ["F1", "<math>2TP / (2TP + FP + FN) = 4 / (4 + 1 + 4)</math>", "<math>4/9 ≈ 0.4444</math>"],
], [83, 295, W - 2 * M - 378])
# paragraph("以 <math>P = 2/3</math>、<math>R = 1/3</math> 代入调和平均，<math>F1 = 2 × (2/3) × (1/3) / (2/3 + 1/3) = 4/9</math>，与计数公式一致。")
heading("代码实现与基础自测")
# paragraph("accuracy_score 比较标签是否相等后取平均；mean_squared_error 将输入转为浮点数，计算平方差的均值。二分类函数调用 _binary_counts 得到 TP、FP、FN，再按公式计算。", SMALL)
log((ROOT / "local_results/logs/classification.txt").read_text(), size=9, leading=13)

new_page("Part 2 决策树", subtitle="")
heading("四种分裂准则")
paragraph("设节点类别比例为 <math>p_c</math>，左右子节点权重为 <math>w_L = N_L/N</math>、<math>w_R = N_R/N</math>。所有类别都参与统计，某个类别未出现在子节点时不产生该类别的求和项。")
table([
    ["准则", "节点不纯度与分裂得分"],
    ["Information Gain", "<math>H(y) = -Σ p_c log₂(p_c)</math>；得分 = <math>H(y) - w_L H(L) - w_R H(R)</math>。"],
    ["Gain Ratio", "得分为信息增益除以 <math>SplitInfo</math>；<math>SplitInfo = -w_L log₂(w_L) - w_R log₂(w_R)</math>。"],
    ["Gini", "<math>G(y) = 1 - Σ p_c²</math>；得分 = <math>G(y) - w_L G(L) - w_R G(R)</math>。"],
    ["Classification Error", "<math>E(y) = 1 - max(p_c)</math>；得分 = <math>E(y) - w_L E(L) - w_R E(R)</math>。"],
], [132, W - 2 * M - 132])
# paragraph("得分越大，表示此次分裂降低的不纯度越多。若父节点为空或某个子节点为空，返回 0，避免空分裂以及信息增益率分母为零。树深度采用框架的计数方式，根节点深度为 1。", SMALL)
heading("实验结果与结构比较")
table([["准则", "训练正确数", "测试正确数 / 准确率", "深度", "叶节点"]] + [
    [t["criterion"], f'{t["train_correct"]}/142',
     f'<math>{t["test_correct"]}/36 = {t["test_correct"]/36:.4f}</math>', t["depth"], t["leaves"]]
    for t in DATA["trees"]
], [118, 90, 170, 58, W - 2 * M - 436])
paragraph("信息增益率在此次划分中表现最好（34/36），但树也最深；Gini 的叶节点最多（11），测试正确数最少（31/36）。因此，复杂度增加并不保证更好的泛化。")
paragraph("分类误差只关注最大类别比例，对其他类别分布变化不够敏感。框架要求分裂收益严格大于 0，所以一些仍包含混合标签的节点会停止分裂；本次训练正确数为 140/142，其他三种为 142/142。信息增益和增益率的根节点均为 flavanoids，但阈值不同；Gini 和分类误差的根节点为 proline。")
# paragraph("测试集只有 36 条，一条样本就对应约 2.78 个百分点；本次结果不足以证明某个准则在所有数据上更优。后续四页以同一模型重绘矢量树图，便于阅读；左分支表示特征小于阈值，右分支表示大于或等于阈值。", SMALL)

sys.path.insert(0, str(ROOT / "decision_tree"))
from decision_tree import DecisionTreeClassifier
train = pd.read_csv(ROOT / "decision_tree/dataset/wine_train.csv")
features = list(train.columns[:-1])
X, y = train[features].to_numpy(), train.label.to_numpy()
for result in DATA["trees"]:
    crit = result["criterion"]
    tree = DecisionTreeClassifier(criterion=crit, random_state=0)
    tree.fit(X, y)
    new_page(f"Part 2 · 树结构 / {crit}", wide=True,
             subtitle=f'测试准确率 <math>{result["test_correct"]}/36 = {result["test_correct"]/36:.4f}</math>；深度 {tree.tree_depth}；叶节点 {tree.tree_leaf_num}。')
    positions = {}
    leaves = [0]
    def place(node, depth):
        if node.value is not None:
            xpos = leaves[0]
            leaves[0] += 1
        else:
            xpos = (place(node.left, depth + 1) + place(node.right, depth + 1)) / 2
        positions[id(node)] = (xpos, depth)
        return xpos
    place(tree.root, 0)
    spacing = (W - 2 * M - 72) / max(leaves[0] - 1, 1)
    top, bottom = Y - 30, 108
    maxdepth = tree.tree_depth - 1
    bw = min(82, spacing * 0.92)
    def point(node):
        xpos, depth = positions[id(node)]
        return M + 36 + xpos * spacing, top - depth * (top - bottom) / max(maxdepth, 1)
    def edges(node):
        if node.value is not None:
            return
        px, py = point(node)
        for child, branch in ((node.left, "<"), (node.right, ">=")):
            cx, cy = point(child)
            CV.setStrokeColor(colors.HexColor("#8A9EAE"))
            CV.setLineWidth(0.8)
            CV.line(px, py - 21, cx, cy + (12 if child.value is not None else 21))
            CV.setFont(MATH_FONT, 8)
            CV.setFillColor(GRAY)
            CV.drawCentredString((px + cx) / 2, (py + cy) / 2, branch)
            edges(child)
    edges(tree.root)
    def boxes(node):
        xx, yy = point(node)
        CV.setStrokeColor(BLUE)
        CV.setLineWidth(0.7)
        if node.value is not None:
            CV.setFillColor([colors.HexColor("#FAE4CF"), colors.HexColor("#E0EFCB"),
                             colors.HexColor("#F3D6E1")][node.value])
            CV.roundRect(xx - 24, yy - 12, 48, 24, 4, fill=1, stroke=1)
            CV.setFillColor(INK)
            CV.setFont("Song", 9)
            CV.drawCentredString(xx, yy - 3, f"class {node.value}")
            return
        CV.setFillColor(PALE)
        CV.roundRect(xx - bw / 2, yy - 21, bw, 42, 4, fill=1, stroke=1)
        CV.setFillColor(INK)
        for i, line in enumerate((features[node.feat_idx], f"thr = {node.threshold:.4g}",
                                  f"gain = {node.split_score:.4f}")):
            node_font = "Song" if i == 0 else MATH_FONT
            fontsize = min(8.5, (bw - 6) / max(pdfmetrics.stringWidth(line, node_font, 1), 1))
            CV.setFont(node_font, fontsize)
            CV.drawCentredString(xx, yy + 10 - 11 * i, line)
        boxes(node.left)
        boxes(node.right)
    boxes(tree.root)
    CV.setFont("Song", 9)
    CV.setFillColor(GRAY)
    CV.drawString(M, 66, "thr：分裂阈值；gain：当前准则的分裂得分；class：叶节点预测类别。各准则得分量纲不同。")

new_page("Part 3 kNN 实现与选参", subtitle="")
heading("实现与实验协议")
paragraph("L2 双循环逐对计算 <math>sqrt(Σ(x_j - z_j)²)</math>。向量化计算采用 <math>||x||² + ||z||² - 2x·z</math>，先将平方距离截断为非负值再开方，避免舍入误差产生 NaN。余弦距离为 <math>1 - (x·z)/(||x|| ||z||)</math>，先归一化再矩阵相乘，并将相似度限制在 <math>[-1, 1]</math>。零向量与任意向量的相似度约定为 0，距离为 1。")
paragraph("对距离作稳定排序，取前 k 个近邻，以出现次数最多的整数标签作为预测；标签票数相同则返回最小标签。等距离的近邻按原训练顺序排列。k 必须是 1 到训练样本数之间的整数。")
paragraph("候选 k 扩展为 <math>1≤k≤250</math> 的所有整数，也包括偶数。仅用 250 条训练样本预测验证集并选择准确率最高的 k，平局时取较小 k。之后合并训练集与验证集共 375 条样本，用已选 k 预测测试集。边界图额外展示 <math>k=375</math> 的退化情况，该值不参与验证集选参。")
heading("验证曲线的数值（代表性 k）")
report_ks = {1, 11, 12, 13, 51, 101, 151, 201, 225, 250}
table([["k", "L2 验证准确率", "Cosine 验证准确率"]] + [
    [k, f'{DATA["knn"]["l2"]["val_accs"][i]:.4f}',
     f'{DATA["knn"]["cosine"]["val_accs"][i]:.4f}']
    for i, k in enumerate(DATA["knn"]["l2"]["ks"])
    if k in report_ks
], [64, 222, W - 2 * M - 286])
table([["距离", "最优 k", "验证正确数 / 准确率", "测试正确数 / 准确率"]] + [
    [metric, d["best_k"], f'<math>{round(max(d["val_accs"])*125)}/125 = {max(d["val_accs"]):.4f}</math>',
     f'<math>{d["test_correct"]}/125 = {d["test_correct"]/125:.4f}</math>']
    for metric, d in DATA["knn"].items()
], [62, 70, 185, W - 2 * M - 317])
paragraph("L2 的验证最优值变为 <math>k=12</math>、0.8560；对应测试准确率为 0.7760，低于原 <math>k=11</math> 的 0.8080。Cosine 在 <math>k=13, 18, 26, 28, 29, 30, 31, 32</math> 时均达到 0.8240，按平局规则仍选 13，测试准确率保持 0.7680。选参依据始终为验证集，验证表现提高不保证测试表现同步提高。", SMALL)

new_page("Part 3 Validation Accuracy vs k", wide=True,
         subtitle="")
width = (W - 2 * M - 20) / 2
top = Y
heights = []
for i, metric in enumerate(("l2", "cosine")):
    CV.setFont("Song", 13)
    CV.setFillColor(BLUE)
    CV.drawString(M + i * (width + 20), top, metric.upper() if metric == "l2" else "Cosine")
    heights.append(image_at(ROOT / f"k_nearest_neighbors/output/knn_k_curve_{metric}.png",
                            M + i * (width + 20), top - 12, width))
Y = top - max(heights) - 27
paragraph("L2 在 <math>k=12</math> 达到 0.8560，之后存在波动；增大到 <math>k=225</math> 时验证准确率降至 0.6400。Cosine 在 <math>k=13</math> 首次达到最高值 0.8240，增大到 <math>k=151</math> 时降至 0.7040，<math>k=225</math> 时为 0.4880。两种距离在 <math>k=250</math> 时均降至 0.2480，已经明显劣化。")
paragraph("验证时训练集只有 250 条样本，<math>k=250</math> 会对全部训练标签投票，所有验证点均预测为类别 1，因此正确数为 <math>31/125</math>。过大的邻域掩盖局部类别结构，模型退化为固定类别预测，表现出明显欠拟合。", SMALL)

for metric in ("l2", "cosine"):
    new_page(f"Part 3 · 决策边界 / {metric.upper() if metric == 'l2' else 'Cosine'}", wide=True,
             subtitle="圆点：原训练样本；三角形：测试样本；黑色叉号：误分类测试点；背景色：预测类别。")
    height = image_at(ROOT / f"k_nearest_neighbors/output/knn_boundary_grid_{metric}.png", M, Y, W - 2 * M)
    Y -= height + 12
    if metric == "l2":
        paragraph("<math>k=1</math> 时边界出现较多细碎区域和局部小岛，对单个训练样本较敏感，体现较高方差及过拟合倾向。k 增大后区域更平滑，小岛减少；选出的 <math>k=12</math> 在测试集上预测正确 97/125 条（77.6%），仍有 28 个误分类点。", SMALL)
        paragraph("更大的 k 会使局部投票更稳定，也可能抹平真正的局部结构。边界模型使用 375 条训练及验证样本，<math>k=250</math> 的测试准确率仍为 0.7600；继续到 <math>k=375</math>，边界完全变为类别 1，测试准确率降至 <math>31/125 = 0.2480</math>，表明明显欠拟合。", SMALL)
    else:
        paragraph("Cosine 只比较方向，同一条从原点出发的射线上，非零查询点拥有相同的距离排序，因此边界呈放射状。<math>k=1</math> 时有许多窄条带；较大的 k 减少条带数量，但仍丢失径向位置差异。<math>k=13</math> 测试正确 96/125 条（76.8%），误分类 29 条。", SMALL)
        paragraph("在这组数据上，新选出的 L2 模型比 Cosine 多预测正确 1 条（高 0.8 个百分点）。继续增大邻域，<math>k=250</math> 时 Cosine 的测试准确率为 0.7360；<math>k=375</math> 时所有点均预测为类别 1，测试准确率降至 <math>31/125 = 0.2480</math>，边界退化为单一区域。", SMALL)

# new_page("测试日志与复现记录", subtitle="下面保留原脚本的实际输出节选；完整日志位于 Assignment1/local_results/logs/。")
# heading("Part 2 · test_decision_tree.py 日志节选")
# dt_log = (ROOT / "local_results/logs/decision_tree.txt").read_text()
# log(dt_log[dt_log.index("=== Criterion:"):], size=9, leading=12)
# heading("Part 3 · test_knn.py 日志节选")
# knn_log = (ROOT / "local_results/logs/knn.txt").read_text()
# selection = knn_log[knn_log.index("[DIST]"):]
# selected = [line for line in selection.splitlines()
#             if not line.startswith(("---", "===", " - ks &")) and line]
# log("\n".join(selected), size=8.5, leading=12)
# paragraph("额外校验：15 passed, 1 warning。检查包括与 sklearn 指标、距离及分类预测的对照，三分类准则的解析算例、空子节点、负标签平票、零向量、未排序 k 的平局处理及非法 k。警告为临时环境的物理 CPU 核数探测提示。", SMALL)
# env = DATA["environment"]
# paragraph("运行环境：" + "；".join(f"{key} {value}" for key, value in env.items()) + "。", SMALL)
# paragraph("复现：分别在 classification/ 执行 python test.py，在 decision_tree/ 执行 python test_decision_tree.py，在 k_nearest_neighbors/ 执行 python test_knn.py 和 python knn_student.py。本机以 Agg 后端绘图；决策树测试仅将字体设为 DejaVu Sans 后调用原测试函数，模型代码、阈值和数据保持一致。", SMALL)
# paragraph("测量汇总可由 local_results/summarize_results.py 重建；报告可由 local_results/build_report.py 重建。依据为 Assignment1 及各 Part 的 README。题目九样本的指标计算与基础自测的小样例是不同输入。", SMALL)

CV.save()
for part, filename in (("classification", "accuracy_error.py"),
                       ("classification", "evaluation_metrics.py"),
                       ("decision_tree", "criterion.py"),
                       ("k_nearest_neighbors", "knn_student.py")):
    shutil.copy2(ROOT / part / filename, DEST / filename)
print(f"Created {DEST / 'report.pdf'} ({PAGE} pages)")
