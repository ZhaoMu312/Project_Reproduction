# YOLOv10 目标检测复现（ICISee 二轮 B2）

在 MS COCO val2017 图片上使用 YOLOv10-N 预训练权重进行推理与可视化。

## 1. 环境

| 项 | 版本 |
|---|---|
| OS | Windows 11 |
| Python | 3.13.12 |
| GPU | NVIDIA RTX 5070 Laptop（8GB） |
| PyTorch | 2.12.1 + cu130 |
| Ultralytics | 8.4.190 |

虚拟环境路径：`D:\yolo_env`（从现有环境复制出的独立副本，与主环境隔离）。

## 2. 安装

```powershell
# 已有 venv 的情况下只补 ultralytics（--no-deps 避免重装 torch）
python -m pip install ultralytics --no-deps
python -m pip install pandas polars ultralytics-thop py-cpuinfo
```

或按 `requirements.txt` 从零安装：

```powershell
python -m pip install -r requirements.txt
```

## 3. 权重与数据

**注意：本机网络环境下 `github.com` 不可访问**，因此不能使用 `YOLO('yolov10n.pt')` 的自动下载功能，权重需手动获取：

| 资源 | 来源 | 说明 |
|---|---|---|
| `yolov10n.pt` | HuggingFace 镜像 `hf-mirror.com/kadirnar/yolov10n` | 约 10.9 MB |
| COCO val2017 图片 | `http://images.cocodataset.org/val2017/` | 用 **http**，https 证书验证失败 |
| 图片文件名清单 | `annotations_trainval2017.zip` 中的 `instances_val2017.json` | 约 253 MB |

数据准备脚本：`B2_下载COCO图.py`（支持断点续传与断线重试）。

从 5000 张 val2017 中固定随机种子抽取 20 张，保证结果可复现。

## 4. 运行

```powershell
& 'D:\yolo_env\Scripts\python.exe' infer.py
```

输出：`runs/predict/` 下 20 张带检测框的图片。

## 5. 关键参数（答辩要点）

| 参数 | 取值 | 含义 |
|---|---|---|
| `conf` | 0.25 | 置信度阈值。调低 → 召回升高但误检增多；调高 → 精度升高但漏检增多 |
| `iou` | 0.45 | NMS 去重阈值，两个框 IoU 超过该值则合并保留高分框 |
| `device` | 0 | 使用第 0 块 GPU |

关于 `iou`：**YOLOv10 的核心卖点是在推理阶段不需要 NMS**（通过一致双重分配实现端到端输出），此处的 `iou` 参数主要是框架兼容保留项，并非实际生效的去重步骤。

## 6. 结果

- 输入：20 张 COCO val2017 图片
- 检测目标总数：**98 个**
- 典型结果：运动场景图中正确检出 person（0.91 / 0.88 / 0.85）、baseball bat、baseball glove

## 7. 目录结构

```
AI1262黄仁睿_B/
├── README.md
├── requirements.txt
├── infer.py              # 推理主脚本
├── B2_下载COCO图.py       # 数据与权重准备脚本
├── images/               # 20 张输入图
└── results/              # 20 张带框输出图
```

## 8. 已知问题与说明

1. GPU 为移动端 8GB 显存，batch size 保持默认 1，未做批处理优化。
2. 未进行模型训练或微调，仅使用官方 COCO 预训练权重推理（符合考核范围）。
3. 部分密集小目标场景存在漏检，与 YOLOv10-N 的最小参数量设计相关（N 版侧重速度）。


---

## 代码仓库

https://github.com/ZhaoMu312/Project_Reproduction

