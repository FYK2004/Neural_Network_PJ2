# CIFAR-10 ResNet-10（任务一）

任务一主实验：在 CIFAR-10 上训练 **ResNet-10** baseline，并完成优化器 / 损失 / 激活的 2×2×2 全因子对比（共 8 组，含 baseline）。

## 项目结构

```
Network_on_CIFAR-10/
├── train.py                 # 训练
├── run_ablation.py          # 主实验：跑齐其余 7 组配置
├── eval.py                  # 测试集评估
├── visualize.py             # 训练曲线、第一层特征图
├── requirements.txt
├── models/resnet_cifar.py
├── data/cifar10_loader.py
└── utils/train_utils.py
```

运行后生成 `outputs/`（已加入 `.gitignore`，克隆后需本地重跑）。

## 环境

```powershell
cd codes/Network_on_CIFAR-10
pip install -r requirements.txt
```

需 PyTorch（建议 GPU）。Windows 下 DataLoader 报错时加 `--num_workers 0`。

## 主实验

### 1. Baseline 训练

```powershell
python train.py --preset cifar100 --output_dir outputs/resnet10_100ep --num_workers 0
```

### 2. 全因子消融（其余 7 组）

Baseline 为 `sgd + ce + relu`（已写入 `outputs/resnet10_100ep`）。再执行：

```powershell
python run_ablation.py
```

若 baseline 仍在训练，可先等其结束：

```powershell
python run_ablation.py --wait-baseline
```

`run_ablation.py` 内部对每组调用 `train.py --preset cifar100`，未完成或中断的可 `--resume` 续训。

### 3. 评估与可视化

```powershell
python eval.py --checkpoint outputs/resnet10_100ep/best.pth
python visualize.py --checkpoint outputs/resnet10_100ep/best.pth --history outputs/resnet10_100ep/history.json
python visualize.py --all-ablation
```

## 主实验默认配置

| 项 | 值 |
|----|-----|
| 模型 | ResNet-10（`--model resnet10`，每 stage 1×BasicBlock，约 5.6M 参数） |
| 训练轮数 | 100（`--preset cifar100`） |
| 优化器 | SGD，lr=0.1，momentum=0.9，weight_decay=5e-4 |
| 学习率调度 | cosine_warmup，warmup 5 epoch |
| 数据 | CIFAR-10，RandomCrop+Flip，标准归一化 |
| batch_size | 128 |

**8 组实验因子**

| 因子 | 取值 |
|------|------|
| optimizer | sgd，adamw |
| loss | ce，label_smooth |
| activation | relu，gelu |

Baseline：`sgd` + `ce` + `relu` → `outputs/resnet10_100ep`  
其余组合 → `outputs/resnet10_ablation/{optimizer}_{loss}_{activation}/`

## 主实验输出

```
outputs/
├── resnet10_100ep/              # baseline
│   ├── best.pth
│   └── history.json
├── resnet10_ablation/           # 7 组消融
│   └── ...
└── resnet10_ablation/summary.json   # run_ablation 结束后汇总（若已生成）
```
