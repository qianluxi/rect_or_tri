import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

# ============================================================
# 1. 配置路径与设备
# ============================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data/dataset_normalized.pt")
MODEL_PATH = os.path.join(ROOT_DIR, "model.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. 定义模型 (必须与训练时完全一致，但稍微修改 forward)
# ============================================================
class ShapeClassifierViz(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 对应训练时的参数：输入2维，隐藏层16维
        self.conv1 = GCNConv(5, 16)
        self.conv2 = GCNConv(16, 16)
        # 注意：这里我们不需要最后的 classifier (Linear层)
        # 因为我们要看的是分类前的“特征空间”

    def forward(self, x, edge_index, batch):
        # 卷积层提取特征
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # 全局池化：把整张图变成一个向量 [batch_size, 16]
        # 这就是图的“指纹”
        embedding = global_mean_pool(x, batch)
        
        return embedding

# ============================================================
# 3. 主流程
# ============================================================
if __name__ == "__main__":
    # --- A. 加载数据 ---
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到数据: {DATA_PATH}，请先运行步骤2。")
        exit()
        
    dataset = torch.load(DATA_PATH)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"📂 加载了 {len(dataset)} 个 DXF 数据样本")

    # --- B. 加载模型 ---
    model = ShapeClassifierViz().to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        # 过滤掉不匹配的层（因为我们把 classifier 层去掉了，或者为了安全）
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model_dict = model.state_dict()
        # 只保留能匹配上的权重
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"✅ 成功加载模型权重: {MODEL_PATH}")
    else:
        print("⚠ 警告: model.pt 不存在！展示的将是随机初始化的结果（必定混在一起）。")

    model.eval()

    # --- C. 提取特征 ---
    all_embeddings = []
    all_labels = []

    print("🔍 正在提取图特征...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            
            # 获取 16维 的图向量
            emb = model(batch.x, batch.edge_index, batch.batch)
            
            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

    # 拼接数据
    X = np.vstack(all_embeddings) # 形状 [N, 16]
    y = np.concatenate(all_labels) # 形状 [N]

    # --- D. PCA 降维 (16维 -> 2维) ---
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    # 计算方差解释率（看这两个轴能代表多少信息）
    explained = pca.explained_variance_ratio_
    print(f"📊 PCA 降维完成。前两个主成分解释了 {sum(explained)*100:.2f}% 的特征差异")

    # --- E. 绘图 ---
    plt.figure(figsize=(8, 6))
    
    # 画三角形 (Label=0)
    idx_tri = (y == 0)
    plt.scatter(X_2d[idx_tri, 0], X_2d[idx_tri, 1], c='red', label='Triangle', alpha=0.7, s=30, marker='^')
    
    # 画矩形 (Label=1)
    idx_rect = (y == 1)
    plt.scatter(X_2d[idx_rect, 0], X_2d[idx_rect, 1], c='blue', label='Rectangle', alpha=0.7, s=30, marker='s')

    plt.title("GNN Classification Result (PCA Visualization)\nRed=Triangle, Blue=Rectangle")
    plt.xlabel(f"Principal Component 1 ({explained[0]:.2f})")
    plt.ylabel(f"Principal Component 2 ({explained[1]:.2f})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()