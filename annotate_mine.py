r"""
标注你自己的图片

用法：
  1. 把图片丢进  D:\YOLO\work\inbox\
  2. 运行：
     & 'D:\YOLO\env\Scripts\python.exe' "D:\deepseek harness work\AI-Plan\标注我的图.py"
  3. 结果在      D:\YOLO\work\out\
     （原图 + 带框图 并排的对比图，也存一份纯带框图）

支持的格式：jpg / jpeg / png / bmp / webp
"""
import os
import shutil

from PIL import Image, ImageDraw
from ultralytics import YOLO

WEIGHT = r'D:\YOLO\work\weights\yolov10n.pt'
INBOX = r'D:\YOLO\work\inbox'
OUT = r'D:\YOLO\work\out'
RAW = os.path.join(OUT, 'raw')          # 纯带框图
CONF = 0.25                              # 置信度阈值，想多检出就调低到 0.15

os.makedirs(INBOX, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
os.makedirs(RAW, exist_ok=True)

# ---------- 收集图片 ----------
EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
files = sorted(f for f in os.listdir(INBOX) if f.lower().endswith(EXTS))

if not files:
    print(f'[空] 没有图片。把图片放到：{INBOX}')
    print('     然后重新运行本脚本。')
    raise SystemExit(0)

print(f'找到 {len(files)} 张图片，开始推理（conf={CONF}）\n')
model = YOLO(WEIGHT)

results = model.predict(
    source=INBOX,
    conf=CONF,
    save=True,
    project=OUT,
    name='predict',
    exist_ok=True,
    device=0,
    verbose=False,
)

# ---------- 结果表 ----------
print('-' * 66)
print(f'{"图片":<26}{"目标数":>6}  内容')
print('-' * 66)

rows = []
total = 0
for r in results:
    boxes = r.boxes
    n = len(boxes)
    total += n
    counts = {}
    for b in boxes:
        nm = model.names[int(b.cls)]
        counts[nm] = counts.get(nm, 0) + 1
    summary = ', '.join(f'{k}×{v}' for k, v in sorted(counts.items(), key=lambda x: -x[1]))
    name = os.path.basename(r.path)
    print(f'{name[:26]:<26}{n:>6}  {summary or "（无检测结果）"}')
    rows.append((r, name, n))

print('-' * 66)
print(f'合计 {len(results)} 张，{total} 个目标\n')

# ---------- 生成对比图 ----------
print('>>> 生成 原图 | 结果 对比图')
made = 0
for r, name, n in rows:
    src = r.path
    pred_dir = os.path.join(OUT, 'predict')
    pred = os.path.join(pred_dir, name)
    if not os.path.exists(pred):
        continue

    # 纯带框图另存一份
    shutil.copy(pred, os.path.join(RAW, name))

    a = Image.open(src).convert('RGB')
    b = Image.open(pred).convert('RGB')
    h = min(a.height, b.height)
    a2 = a.resize((int(a.width * h / a.height), h))
    b2 = b.resize((int(b.width * h / b.height), h))
    canvas = Image.new('RGB', (a2.width + b2.width + 10, h + 28), 'white')
    canvas.paste(a2, (0, 28))
    canvas.paste(b2, (a2.width + 10, 28))
    d = ImageDraw.Draw(canvas)
    d.text((6, 8), f'original  {name}', fill='black')
    d.text((a2.width + 16, 8), f'detected  {n} objects', fill='black')
    canvas.save(os.path.join(OUT, f'对比_{name}'), quality=92)
    made += 1

print(f'\n完成：{made} 张')
print(f'  对比图     -> {OUT}\\对比_*.jpg')
print(f'  纯带框图   -> {RAW}')
print(f'  原始输出   -> {os.path.join(OUT, "predict")}')
