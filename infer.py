r"""
B2 复现主脚本：YOLOv10 在 COCO val2017 图片上推理 + 可视化

用法：
  & 'D:\YOLO\env\Scripts\python.exe' "D:\deepseek harness work\AI-Plan\B2_推理.py"

前置：先跑 B2_下载COCO图.py（权重必须本地下载，GitHub 不通无法自动下载）

输出：
  D:\YOLO\work\runs\predict\    ← 20 张带检测框的图

答辩必问的三个参数：
  conf=0.25  置信度阈值：低于它的框丢弃。调低→召回高但误检多；调高→精度高但漏检多
  iou=0.45   NMS 去重阈值：两框 IoU 超过它就合并保留高分框
             ⚠️ YOLOv10 的卖点是"推理不需要 NMS"，这个参数主要是框架兼容
  device=0   用第 0 块 GPU；'cpu' 则用 CPU
"""
import os
from ultralytics import YOLO

WEIGHT = r'D:\YOLO\work\weights\yolov10n.pt'
SRC = r'D:\YOLO\work\images'
PROJ = r'D:\YOLO\work\runs'

# 防呆：权重不在就直接报错，避免 ultralytics 去 GitHub 下载然后卡死
if not os.path.exists(WEIGHT):
    raise SystemExit(f'权重不存在：{WEIGHT}\n先运行 B2_下载COCO图.py')
if not os.path.isdir(SRC):
    raise SystemExit(f'图片文件夹不存在：{SRC}')

imgs = [f for f in os.listdir(SRC) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f'找到 {len(imgs)} 张图片')

# 1. 加载本地权重
model = YOLO(WEIGHT)

# 2. 推理
results = model.predict(
    source=SRC,
    conf=0.25,
    iou=0.45,
    save=True,
    save_txt=False,
    project=PROJ,
    name='predict',
    exist_ok=True,
    device=0,
    verbose=False,
)

# 3. 打印结果，答辩时用来自我检查
print('\n' + '=' * 62)
total_obj = 0
for r in results:
    n = len(r.boxes)
    total_obj += n
    print(f'\n{r.path}')
    print(f'  检测到 {n} 个目标')
    for b in r.boxes[:5]:
        print(f'    {model.names[int(b.cls)]:<15} conf={float(b.conf):.2f}  '
              f'box={[round(float(v), 1) for v in b.xyxy[0]]}')
print('\n' + '=' * 62)
print(f'共 {len(results)} 张图，检测到 {total_obj} 个目标')
print(f'可视化结果：{os.path.join(PROJ, "predict")}')
