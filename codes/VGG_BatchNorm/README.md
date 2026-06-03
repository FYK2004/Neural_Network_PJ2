# VGG-A vs VGG-A+BatchNorm（CIFAR-10）

Project 2 任务二主实验：对比 VGG-A 与 VGG-A+BN 的训练曲线、梯度探测图与 loss landscape。

## 项目结构

```
VGG_BatchNorm/
├── run_task2.py              # 主实验一键脚本
├── train_compare.py          # 步骤 1
├── VGG_Loss_Landscape.py     # 步骤 2
├── requirements.txt
├── models/vgg.py
├── data/loaders.py
└── utils/
    ├── grad_probe.py
    ├── grad_plots.py
    ├── probe_select.py
    ├── optim.py
    └── nn.py
```

运行后生成 `outputs/`（权重、日志、图表；已加入 `.gitignore`，克隆后需本地重跑）。

## 环境

```powershell
cd codes/VGG_BatchNorm
pip install -r requirements.txt
```

需 PyTorch（建议 GPU）。Windows 下 DataLoader 报错时加 `--num_workers 0`。

## 主实验

```powershell
python -u run_task2.py
```

等价于依次执行：

| 步骤 | 脚本 | 内容 |
|------|------|------|
| 1 | `train_compare.py` | 100 epoch，VGG-A 与 VGG-A+BN；SGD lr=0.1、cosine warmup(5)、增强、dropout=0.5、label smoothing=0.1；多 epoch 梯度探测并自动选作图 epoch |
| 2 | `VGG_Loss_Landscape.py` | 20 epoch × lr `{0.1, 0.05, 0.01, 0.005}`，cosine、无 warmup；绘制多 lr loss 带 |

### 主实验默认配置

**步骤 1 训练**

| 项 | 值 |
|----|-----|
| 优化器 | SGD，momentum=0.9，weight_decay=5e-4 |
| 学习率 / 调度 | 0.1，cosine_warmup，warmup 5 epoch |
| 数据 | CIFAR-10，RandomCrop+Flip，标准归一化 |
| 探测 epoch | 5, 10, 15, 20, 25, 30, 40, 50（`utils/grad_probe.py` 中 `PROBE_EPOCHS`） |
| 报告图 epoch | 自动选取（`outputs/probe_selection.json`） |

**步骤 2 loss landscape**

| 项 | 值 |
|----|-----|
| 学习率 | 0.1, 0.05, 0.01, 0.005 |
| 调度 | cosine，warmup=0 |
| 其余训练设置 | 与步骤 1 一致（SGD、增强、dropout、label smoothing） |

### 主实验输出

```
outputs/
├── figures/
│   ├── vgg_vs_bn_training.png
│   ├── gradient_predictiveness.png      # Taylor 误差，epoch 自动选取
│   └── max_grad_diff.png
├── probe_selection.json
├── summary.json
├── vgg_a/   vgg_a_bn/
└── loss_landscape/figures/
    ├── loss_landscape_panels.png
    └── loss_landscape_combined.png
```
