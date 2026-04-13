# YOLOv11-VisDrone-Detection

基于 YOLOv11 + VisDrone 无人机数据集的行人与车辆检测实验

> 深圳大学 电子与信息工程学院 人工智能课程实验报告项目

## 项目简介

本项目使用 **YOLOv11n** 模型，基于 **VisDrone 无人机航拍数据集** 进行微调训练，实现对俯拍场景下行人和车辆的精准检测。通过新旧模型对比实验，验证了训练数据视角匹配对检测效果的决定性影响。

## 实验环境

| 配置项 | 详细信息 |
|--------|----------|
| GPU | NVIDIA GeForce RTX 3060 (6GB) |
| Python | 3.10 |
| 框架 | Ultralytics YOLOv11 |
| CUDA | 11.8 |
| 开发工具 | PyCharm |

## 项目结构

```
├── train_visdrone.py        # VisDrone 微调训练脚本
├── test_compare.py          # 新旧模型对比测试脚本
├── test-video.MOV           # 校园俯拍测试视频
├── results/                 # 检测结果视频
│   ├── old_model_result.mp4 # 旧模型（通用数据集）检测效果
│   └── new_model_result.mp4 # 新模型（VisDrone 微调）检测效果
└── docs/                    # 实验报告 LaTeX 源文件
```

## 实验思路

### 问题发现
使用通用数据集训练的模型在校园俯拍视频上效果不佳——漏检多、分类不准。

### 解决方案
校园俯拍视角接近无人机航拍，因此选择 **VisDrone 数据集**（天津大学发布的无人机航拍交通数据集，含 10 类目标）进行微调，使模型学习到俯拍视角的目标特征。

### 训练配置
- 底座模型：`yolo11n.pt`（YOLOv11 nano 官方预训练权重）
- 训练轮数：15 epochs
- 输入分辨率：640×640
- Batch Size：8

### 对比测试
使用同一段校园俯拍视频、相同推理参数（conf=0.15, imgsz=1280），分别用旧模型和新模型进行推理，直观对比检测效果。

另外场景的测试：
场景1：https://github.com/user-attachments/assets/49361d84-7596-4e68-a609-3366489215d9

场景2：https://github.com/user-attachments/assets/06ee8b92-62ca-45af-8be9-7aee923464a1

## 核心结论

1. **数据集视角匹配** 比单纯增加训练轮数更重要
2. VisDrone 微调后，小目标检测能力和分类准确率显著提升
3. RTX 3060 (6GB) 足以完成 VisDrone 数据集的微调训练
4. 仅 15 轮微调即可让模型快速适应俯拍场景

## 快速复现

```bash
# 1. 创建环境
conda create -n yolo11 python=3.10 -y
conda activate yolo11

# 2. 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics

# 3. 训练（需自行下载 VisDrone 数据集并配置 visdrone.yaml）
python train_visdrone.py

# 4. 对比测试（需修改脚本中的模型权重路径）
python test_compare.py
```

## 参考资料

- [Ultralytics YOLOv11 官方文档](https://docs.ultralytics.com/)
- [VisDrone 数据集](https://github.com/VisDrone/VisDrone-Dataset)
- [深圳大学实验报告 LaTeX 模板](https://github.com/MrR0922/SZU-Lab-Report-LaTeX)
