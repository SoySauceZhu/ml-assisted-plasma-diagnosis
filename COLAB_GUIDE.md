# Google Colab 运行指南

## 概述

本指南帮助你在 Google Colab 上运行 Phase 2 超参数调优代码。该代码支持 **GPU 加速**（如果可用）。

## 准备工作

### 1. 上传数据文件

在 Colab 中运行前，需要上传数据文件 `oes_ml_dataset_1nm.csv`：

**方法 A：直接从本地上传**
```python
from google.colab import files
uploaded = files.upload()
# 选择 oes_ml_dataset_1nm.csv 文件
```

**方法 B：挂载 Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
# 将数据文件放在 Colab Notebooks 文件夹中
```

### 2. 安装依赖

在 Colab 单元格中运行：

```python
# 安装所需包
!pip install optuna scikit-learn pandas numpy matplotlib torch
```

## 运行代码

### 方法 1：直接运行完整流程（推荐）

```python
# 上传 colab.py 和数据文件后
import sys
sys.path.insert(0, '/content')

from colab import main

# 运行完整流程（包括调优和评估）
all_results, results_df, tuned_params, studies = main(
    csv_path='oes_ml_dataset_1nm.csv',
    models=['RF', 'MLP', 'CNN'],  # 可选择 ['RF'], ['MLP'], ['CNN'] 等
    pca_k=11,
    n_trials_rf=200,   # RF 调优试验次数
    n_trials_mlp=100,  # MLP 调优试验次数
    n_trials_cnn=100   # CNN 调优试验次数
)
```

### 方法 2：仅运行调优

```python
from colab import main

# 仅调优，不进行最终评估
tuned_params, studies = main(
    csv_path='oes_ml_dataset_1nm.csv',
    tune_only=True,
    models=['RF', 'MLP', 'CNN']
)
```

### 方法 3：仅运行评估（使用已保存的参数）

```python
from colab import main

# 使用之前保存的调优参数进行评估
all_results, results_df = main(
    csv_path='oes_ml_dataset_1nm.csv',
    eval_only=True
)
```

### 方法 4：命令行运行

```python
!python colab.py --models RF MLP CNN --pca-k 11 --n-trials-rf 200 --n-trials-mlp 100 --n-trials-cnn 100
```

## GPU 加速说明

代码会自动检测并使用 GPU：

- **有 GPU**：输出 `Using GPU: Tesla T4`（或类似）
- **无 GPU**：输出 `Using CPU`，使用 CPU 进行训练

MLP 和 CNN 模型会自动将张量移动到 GPU 设备上。

## 输出文件

运行后会在以下位置生成结果：

```
results/
├── tables/
│   ├── tuned_hyperparameters.json    # 最佳超参数
│   ├── phase2_loocv_results_summary.csv
│   └── phase2_predictions_detail.csv
└── figures/
    ├── optuna_history_*.png         # 调优历史
    ├── optuna_importance_*.png      # 特征重要性
    ├── predicted_vs_actual_*.png    # 预测 vs 实际
    └── phase1_vs_phase2_comparison.png
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `csv_path` | `"oes_ml_dataset_1nm.csv"` | 数据文件路径 |
| `models` | `["RF", "MLP", "CNN"]` | 要调优的模型 |
| `pca_k` | `11` | PCA 主成分数量 |
| `n_trials_rf` | `200` | RF Optuna 试验次数 |
| `n_trials_mlp` | `100` | MLP Optuna 试验次数 |
| `n_trials_cnn` | `100` | CNN Optuna 试验次数 |
| `tune_only` | `False` | 仅调优，不评估 |
| `eval_only` | `False` | 仅评估，使用已保存参数 |

## 快速开始示例

```python
# 完整示例代码 - 复制到 Colab 单元格中运行

# 1. 安装依赖
!pip install optuna scikit-learn pandas numpy matplotlib torch -q

# 2. 上传文件（从本地选择 colab.py 和 oes_ml_dataset_1nm.csv）
from google.colab import files
print("请上传 colab.py 文件")
files.upload()
print("请上传 oes_ml_dataset_1nm.csv 文件")
files.upload()

# 3. 运行代码
import sys
sys.path.insert(0, '/content')

from colab import main

# 运行调优（可减少试验次数加快速度）
all_results, results_df, tuned_params, studies = main(
    csv_path='oes_ml_dataset_1nm.csv',
    models=['RF', 'MLP', 'CNN'],
    n_trials_rf=50,   # 减少次数加快演示
    n_trials_mlp=30,
    n_trials_cnn=30
)

# 4. 查看结果
print("\n=== 最终结果 ===")
print(results_df)
```

## 注意事项

1. **运行时间**：完整调优可能需要较长时间（取决于试验次数）
   - RF 200 次试验 ≈ 10-30 分钟
   - MLP/CNN 各 100 次试验 ≈ 20-60 分钟

2. **内存**：如果遇到内存问题，可以减少试验次数或只选择部分模型

3. **断点续训**：如果调优中断，可以设置 `eval_only=True` 使用已保存的参数继续评估
