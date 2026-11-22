r"""
B2 准备脚本（网络受限 + 不稳定版）：
  1. 下载 YOLOv10-N 权重（HF 镜像，GitHub 不通）
  2. 下载 20 张 COCO val2017 图片（COCO 官方 http）

用法：
  & 'D:\YOLO\env\Scripts\python.exe' "D:\deepseek harness work\AI-Plan\B2_下载COCO图.py"

特性：断点续传 + 断线自动重试（最多 10 次），中断后随时重跑接着下。
"""
import json
import os
import random
import socket
import time
import zipfile
import urllib.request
import urllib.error

socket.setdefaulttimeout(45)          # 单次读操作最长等 45 秒，超时就重试

WORK = r'D:\YOLO\work'
IMG_DIR = os.path.join(WORK, 'images')
W_DIR = os.path.join(WORK, 'weights')

WEIGHT_URL = 'https://hf-mirror.com/kadirnar/yolov10n/resolve/main/yolov10n.pt'
WEIGHT_PATH = os.path.join(W_DIR, 'yolov10n.pt')

ANN_ZIP = os.path.join(WORK, 'annotations_trainval2017.zip')
ANN_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
IMG_URL = 'http://images.cocodataset.org/val2017/'

for d in (WORK, IMG_DIR, W_DIR):
    os.makedirs(d, exist_ok=True)


def human(n):
    return f'{n / 1048576:.1f} MB'


def _try_once(url, path, desc, expect_size=None):
    """一次下载尝试。返回 True=完成，False=未完成（需重试）"""
    tmp = path + '.part'
    pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0

    if expect_size and pos >= expect_size:
        os.replace(tmp, path)
        print(f'\n  [完成] {desc} {human(pos)}')
        return True

    req = urllib.request.Request(url)
    if pos:
        req.add_header('Range', f'bytes={pos}-')

    with urllib.request.urlopen(req, timeout=45) as resp:
        if pos and resp.status != 206:
            pos = 0
        total = int(resp.headers.get('Content-Length') or 0) + pos
        done = pos
        last = -1
        with open(tmp, 'ab' if pos else 'wb') as f:
            while True:
                chunk = resp.read(262144)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = int(done * 100 / total)
                    if pct != last:
                        print(f'\r  {desc}: {human(done)}/{human(total)} ({pct:3d}%)',
                              end='', flush=True)
                        last = pct

    if total and done < total:
        print(f'\n  [中断] 已下 {human(done)}，待重试')
        return False
    os.replace(tmp, path)
    print(f'\n  [完成] {desc} {human(done)}')
    return True


def download(url, path, desc='', attempts=10):
    """带重试的下载；每失败一次就续传一次"""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f'[skip] {os.path.basename(path)}（{human(os.path.getsize(path))}）')
        return path

    for i in range(1, attempts + 1):
        try:
            if _try_once(url, path, desc):
                return path
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, socket.timeout, ConnectionError, OSError) as e:
            print(f'\n  [第 {i}/{attempts} 次失败] {type(e).__name__}: {e}')
        time.sleep(2 * i)              # 退避：2s, 4s, 6s ...
    raise RuntimeError(f'{desc} 下载失败，已重试 {attempts} 次。网络恢复后重跑本脚本可续传。')


# ---------- 1. 权重 ----------
print('>>> 第 1 步：YOLOv10-N 权重')
download(WEIGHT_URL, WEIGHT_PATH, 'yolov10n.pt')

# ---------- 2. 标注包 ----------
print('\n>>> 第 2 步：COCO 标注包（253MB，会自动重试）')
download(ANN_URL, ANN_ZIP, 'annotations')

# ---------- 3. 清单 ----------
print('\n>>> 第 3 步：解析文件名清单')
with zipfile.ZipFile(ANN_ZIP) as z:
    with z.open('annotations/instances_val2017.json') as f:
        meta = json.load(f)
names = sorted(img['file_name'] for img in meta['images'])
print(f'val2017 共 {len(names)} 张')

random.seed(20261027)
picked = random.sample(names, 20)

# ---------- 4. 图 ----------
print('\n>>> 第 4 步：下载 20 张图')
ok, fail = 0, []
for i, n in enumerate(picked, 1):
    try:
        download(IMG_URL + n, os.path.join(IMG_DIR, n), f'[{i:2d}/20] {n}', attempts=5)
        ok += 1
    except Exception as e:
        print(f'  [失败] {n}: {e}')
        fail.append(n)

print()
print('=' * 56)
print(f'权重：{WEIGHT_PATH}')
print(f'图片：成功 {ok} 张，失败 {len(fail)} 张 -> {IMG_DIR}')
print('=' * 56)
