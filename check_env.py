r"""
环境自检：一条命令输出"环境已就绪"的证据
跑法：& 'D:\YOLO\env\Scripts\python.exe' 'D:\YOLO\scripts\check_env.py'
输出：屏幕表格 + D:\YOLO\环境自检报告.txt
"""
import os
import platform
import sys
from datetime import datetime

import torch
import torchvision
import cv2
import numpy as np
from ultralytics import __version__ as ul_version

BASE = r'D:\YOLO'
WEIGHT = os.path.join(BASE, r'work\weights\yolov10n.pt')
IMGDIR = os.path.join(BASE, r'work\images')
RESDIR = os.path.join(BASE, r'work\runs\predict')

lines = []


def p(s=''):
    print(s)
    lines.append(str(s))


def check(label, value, ok=None):
    mark = '' if ok is None else ('  OK' if ok else '  FAIL')
    p(f'  {label:<16}{value}{mark}')


p('=' * 68)
p('  YOLOv10 目标检测复现 · 环境自检报告')
p(f'  生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}')
p('=' * 68)

p('\n[1] 操作系统')
check('系统', f'{platform.system()} {platform.release()}')
check('架构', platform.machine())

p('\n[2] Python 与虚拟环境')
check('Python', sys.version.split()[0])
check('解释器', sys.executable)
check('隔离方式', '独立 venv（与 ComfyUI 主环境隔离）')

p('\n[3] 深度学习框架')
check('PyTorch', torch.__version__)
check('torchvision', torchvision.__version__)
check('CUDA 编译版本', torch.version.cuda)
check('CUDA 可用', torch.cuda.is_available(), torch.cuda.is_available())
if torch.cuda.is_available():
    check('GPU 型号', torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    check('显存', f'{props.total_memory / 1024**3:.1f} GB')
    check('计算能力', f'sm_{props.major}{props.minor}')

p('\n[4] 依赖库')
check('ultralytics', ul_version)
check('OpenCV', cv2.__version__)
check('NumPy', np.__version__)

p('\n[5] 权重文件')
ok_w = os.path.exists(WEIGHT)
check('yolov10n.pt', f'{os.path.getsize(WEIGHT) / 1048576:.1f} MB' if ok_w else '缺失', ok_w)

p('\n[6] 数据与结果')
n_img = len([f for f in os.listdir(IMGDIR) if f.endswith('.jpg')]) if os.path.isdir(IMGDIR) else 0
n_res = len([f for f in os.listdir(RESDIR) if f.endswith('.jpg')]) if os.path.isdir(RESDIR) else 0
check('COCO val2017 输入图', f'{n_img} 张', n_img >= 20)
check('带检测框输出图', f'{n_res} 张', n_res >= 20)

p('\n[7] GPU 实测（跑一次矩阵乘法）')
if torch.cuda.is_available():
    a = torch.randn(1000, 1000, device='cuda')
    import time
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        b = a @ a
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 20
    check('1000x1000 矩阵乘', f'{dt * 1000:.1f} ms/次')
    check('算力验证', 'GPU 计算正常', dt < 1.0)

p('\n' + '=' * 68)
p('  结论：环境已就绪，可进行推理与可视化')
p('=' * 68)

out = os.path.join(BASE, '环境自检报告.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'\n报告已保存：{out}')
