# 📐 DXF Shape Classifier with GNN (Educational)

这是一个基于图神经网络（GNN）的 **DXF CAD 图纸几何形状分类器**。

本项目是一种有监督的深度学习。通过将复杂的工程图纸识别任务简化为经典的 **“三角形 vs 矩形”** 二分类问题，演示了如何将 CAD 矢量数据转化为图数据，并利用 PyTorch Geometric 进行训练的全过程。

## 🎯 项目目标

*   **输入**：包含几何图形（三角形或矩形）的 `.dxf` 文件。
*   **处理**：解析 DXF 线条 -> 构建图结构 (Graph Construction) -> 特征归一化 (Normalization)。
*   **模型**：基于 GCN (Graph Convolutional Network) 的图分类器。
*   **输出**：识别结果（Triangle / Rectangle）。

## 📂 文件结构

```text
.
├── 1_gen_dxf_v2.py          # [数据生成] 生成随机旋转、大小不一的不等边三角形和矩形 DXF
├── 2_dxf_to_graph_norm.py   # [预处理] 将 DXF 转换为 PyG 图数据 (含归一化与去重)
├── 3_train.py               # [模型训练] 训练 GCN 模型并保存为 model.pt
├── 4_visualize.py           # [可视化] 使用 PCA 对训练后的特征空间进行降维可视化
├── 5_predict.py             # [推理预测] 加载模型对单个 DXF 文件进行预测
├── diagnose.py              # [诊断工具] 打印 DXF 的拓扑结构（节点数/边数）排查数据问题
├── data/                    # 存放数据
│   ├── dxf/                 # 生成的 .dxf 文件
│   └── dataset_normalized.pt # 处理后的图数据集
└── model.pt                 # 训练好的模型权重
```

## 🚀 快速开始

### 1. 环境依赖
请确保安装了以下库：
```bash
pip install torch torch-geometric ezdxf numpy matplotlib scikit-learn
```

### 2. 生成数据
运行脚本，在 `data/dxf` 目录下生成 200 个随机形状的 DXF 文件。
这些图形包含随机旋转、平移和不规则形变，用于模拟真实的绘图环境。
```bash
python 1_gen_dxf_v2.py
```

### 3. 数据预处理（关键步骤）
将 DXF 线条转换为图节点。
*   **节点特征**：线段长度 + 端点坐标。
*   **归一化**：消除图形大小和平移的影响。
*   **方向去敏**：对端点坐标排序，确保 A->B 和 B->A 生成相同的特征。
```bash
python 2_dxf_to_graph_norm.py
```

### 4. 训练模型
运行图神经网络进行训练。通常在 50 个 Epoch 内即可达到 100% 准确率。
```bash
python 3_train.py
```

### 5. 可视化结果
查看模型学到的特征分布。红色（三角形）和蓝色（矩形）应当在空间中被完美分开。
```bash
python 4_visualize.py
```

### 6. 预测新文件
识别一个具体的 DXF 文件是什么形状。
```bash
python 5_predict.py
```

## 🧠 核心知识点与避坑指南

在开发过程中，本项目解决了以下几何深度学习中的典型问题：

1.  **模型坍塌 (Model Collapse)**：
    *   *现象*：只使用“线长”作为特征，模型无法收敛，准确率停留在 50%。
    *   *原因*：正三角形和正方形的边长特征可能完全一致，GNN 无法区分局部同构图。
    *   *解决*：引入**坐标特征**。

2.  **平移不变性 (Translation Invariance)**：
    *   *现象*：使用绝对坐标时，模型无法识别画在不同位置的同一图形。
    *   *解决*：实施**坐标归一化 (Normalization)**，将所有图形中心移至原点，并缩放到统一尺度。

3.  **画线方向敏感性 (Direction Sensitivity)**：
    *   *现象*：手动重画图形后识别错误。
    *   *原因*：线段 `(0,0)->(1,1)` 和 `(1,1)->(0,0)` 生成的特征向量不同，导致模型困惑。
    *   *解决*：在生成特征前对端点坐标进行**排序**，确保特征具有方向不变性。

4.  **拓扑清洗 (Topology Cleaning)**：
    *   *现象*：视觉上闭合的图形被识别错误。
    *   *原因*：CAD 中存在肉眼不可见的重叠线或断线。
    *   *解决*：在预处理阶段增加去重和微小间隙连接逻辑。

## 📜 License
MIT License
