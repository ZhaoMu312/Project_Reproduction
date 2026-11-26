# YOLOv10 目标检测复现（ICISee 二轮 · B 方向）

在 MS COCO val2017 上使用 YOLOv10-N 官方预训练权重完成推理与可视化，并配套一份关于 YOLOv10 的文献调研报告。

## 项目内容

| 部分 | 说明 |
|---|---|
| 复现 | YOLOv10-N 推理 + 20 张 COCO val2017 图片的可视化 |
| 报告 | `文献调研报告.docx`：目标检测发展脉络、YOLOv10 关键设计、评价指标 |
| 结果 | `results/` 带检测框图；`检测结果对比图/` 原图与结果并排 |
| 工程 | 环境自检、权重获取、数据准备脚本 |

## 目录结构

```
.
├── infer.py                     # 推理主脚本
├── demo.py                      # 演示：逐图检测表 + 对比图生成
├── check_env.py                 # 环境自检（版本、GPU、CUDA、依赖）
├── annotate_mine.py             # 对自定义图片推理
├── B2_下载COCO图.py             # 下载权重与 COCO 图片（支持断点续传）
├── run_yolo.cmd / open_results.cmd
├── requirements.txt
├── 环境自检报告.txt
├── 文献调研报告.docx            # 文献调研报告
├── images/                      # 20 张 COCO val2017 输入图
├── results/                     # 20 张带检测框输出图
└── 检测结果对比图/               # 20 张「原图 | 结果」并排图
```

## 环境

| 项 | 版本 |
|---|---|
| OS | Windows 11 |
| Python | 3.13.12（独立 venv） |
| PyTorch | 2.12.1 + cu130 |
| ultralytics | 8.4.190 |
| GPU | RTX 5070 Laptop（8GB, sm_120） |

安装：

```bash
pip install ultralytics --no-deps
pip install pandas polars ultralytics-thop py-cpuinfo
```

> GPU 版 torch 需从官方 index 安装（Blackwell 架构需 cu128 以上）：
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`

## 运行

```bash
# 1. 环境自检（打印版本、CUDA、GPU、依赖、数据量）
python check_env.py

# 2. 推理（20 张 COCO val2017）
python infer.py

# 3. 演示模式（逐图检测表 + 生成对比图 + 日志）
python demo.py
```

输出：`runs/predict/`（带框图）、`runs/compare/`（原图|结果并排）

## 关键参数

| 参数 | 取值 | 说明 |
|---|---|---|
| `conf` | 0.25 | 置信度阈值，调低召回高但误检多 |
| `iou` | 0.45 | NMS 合并阈值；YOLOv10 推理端已不需要 NMS，此项为框架兼容 |
| `device` | 0 | 使用第 0 块 GPU |

## 结果

- 输入：COCO val2017 随机抽取 20 张（固定随机种子，可复现）
- 输出：20 张带框可视化，共检出 **98 个目标**
- 模型：YOLOv10-N，223 层 / 2,775,520 参数 / 8.8 GFLOPs
- 定性观察：大目标与常见类别检出稳定；密集人群与远处小目标存在漏检
- **说明**：本次只做推理与可视化，未复算 mAP（需完整 5000 张 val2017 + GT 匹配）

## 复现中解决的两个工程问题

1. 本机无法访问 GitHub，`YOLO('yolov10n.pt')` 自动下载失败 → 改为从 HuggingFace 镜像手动获取权重，脚本加载本地路径。
2. COCO 官方图片服务的 https 证书校验失败 → 改用 http 协议下载。

## 参考

- Wang A, Chen H, Liu L, et al. **YOLOv10: Real-Time End-to-End Object Detection**. NeurIPS, 2024. ([arXiv:2405.14458](https://arxiv.org/abs/2405.14458))
- 官方代码：https://github.com/THU-MIG/yolov10

## 说明

模型结构、预训练权重与性能数值均来自作者发布的论文与官方实现；本仓库的工作是环境配置、数据准备、推理与可视化脚本，以及据此完成的文献调研。
