r"""
B2 演示脚本：把推理过程"看得见"
  - 打印环境信息（Python / torch / GPU 名称）
  - 打印模型结构摘要（层数、参数量、GFLOPs）
  - 逐图输出检测结果表格
  - 保存完整日志到 runs\predict\log.txt
  - 生成「原图 | 结果」左右对比图到 runs\compare\

用法：
  & 'D:\yolo_env\Scripts\python.exe' "D:\deepseek harness work\AI-Plan\B2_演示.py"
"""
import os
import sys
import time
from datetime import datetime

import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

WEIGHT = r'D:\YOLO\work\weights\yolov10n.pt'
SRC = r'D:\YOLO\work\images'
PROJ = r'D:\YOLO\work\runs'
CMP = os.path.join(PROJ, 'compare')
os.makedirs(CMP, exist_ok=True)

log_lines = []


def log(msg=''):
    print(msg)
    log_lines.append(str(msg))


# ---------- 1. 环境信息 ----------
log('=' * 66)
log(f'运行时间：{datetime.now():%Y-%m-%d %H:%M:%S}')
log(f'Python     : {sys.version.split()[0]}  ({sys.executable})')
log(f'PyTorch    : {torch.__version__}')
log(f'CUDA 可用   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    log(f'GPU        : {torch.cuda.get_device_name(0)}')
    log(f'显存       : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    log(f'CUDA 版本   : {torch.version.cuda}')
log('=' * 66)

# ---------- 2. 模型结构 ----------
log('\n>>> 加载模型')
model = YOLO(WEIGHT)

log('\n>>> 模型结构摘要')
log(f'模型文件   : {WEIGHT}')
try:
    # info() 会打印每层的参数量与 GFLOPs 汇总
    model.info(detailed=False, verbose=True)
except TypeError:
    model.info()

log('\n>>> 网络层列表（前 12 层，完整版看控制台）')
try:
    for i, (name, mod) in enumerate(list(model.model.named_children())[:12]):
        log(f'  {i:2d}. {name:<12} {type(mod).__name__}')
except Exception as e:
    log(f'  （无法展开层列表：{e}）')

# ---------- 3. 推理 ----------
imgs = sorted(f for f in os.listdir(SRC) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
log(f'\n>>> 开始推理，共 {len(imgs)} 张图')

results = model.predict(
    source=SRC, conf=0.25, iou=0.45,
    save=True, project=PROJ, name='predict', exist_ok=True,
    device=0, verbose=False,
)

# ---------- 4. 逐图结果表 ----------
log('\n' + '=' * 66)
log(f'{"图片":<22}{"目标数":>6}  检测内容')
log('-' * 66)

t0 = time.time()
total = 0
detail = []
for r in results:
    boxes = r.boxes
    n = len(boxes)
    total += n
    names = [model.names[int(b.cls)] for b in boxes]
    # 统计每个类别出现几次
    counts = {}
    for nm in names:
        counts[nm] = counts.get(nm, 0) + 1
    summary = ', '.join(f'{k}×{v}' for k, v in sorted(counts.items(), key=lambda x: -x[1]))
    if not summary:
        summary = '（无检测结果）'
    log(f'{os.path.basename(r.path):<22}{n:>6}  {summary}')
    detail.append((r, n))

log('-' * 66)
log(f'合计：{len(results)} 张图，{total} 个目标，耗时 {time.time() - t0:.2f} 秒')
log(f'平均每张 {total / max(len(results), 1):.1f} 个目标')
log('=' * 66)

# ---------- 5. 对比图 ----------
log('\n>>> 生成「原图 | 结果」对比图')
for r, n in detail:
    name = os.path.basename(r.path)
    src_path = r.path
    out_path = os.path.join(PROJ, 'predict', name)
    if not os.path.exists(out_path):
        continue
    a = Image.open(src_path).convert('RGB')
    b = Image.open(out_path).convert('RGB')
    # 统一高度
    h = min(a.height, b.height)
    a = a.resize((int(a.width * h / a.height), h))
    b = b.resize((int(b.width * h / b.height), h))
    canvas = Image.new('RGB', (a.width + b.width + 10, h + 28), 'white')
    canvas.paste(a, (0, 28))
    canvas.paste(b, (a.width + 10, 28))
    d = ImageDraw.Draw(canvas)
    d.text((6, 8), f'original  {name}', fill='black')
    d.text((a.width + 16, 8), f'YOLOv10-N  ({n} objects)', fill='black')
    canvas.save(os.path.join(CMP, name), quality=90)
log(f'对比图已保存：{CMP}')

# ---------- 6. 日志落盘 ----------
log_path = os.path.join(PROJ, 'predict', 'log.txt')
with open(log_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
print(f'\n完整日志：{log_path}')
