import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ============================================================
# 1. 配置
# ============================================================
DATA_PATH = "data/dataset_normalized.pt"
dataset = torch.load(DATA_PATH)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# 2. 对比学习模型 (Encoder + Projection Head)
# ============================================================
class ContrastiveGNN(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Encoder: 提取特征 (我们最后要用的部分)
        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, 16)
        
        # Projection Head: 专门为了计算 Loss 用的辅助层 (训练完丢弃)
        # 将特征映射到另一个空间，有助于学习
        self.proj_head = torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16)
        )

    def forward(self, x, edge_index, batch):
        # --- 编码 ---
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        embedding = global_mean_pool(x, batch) # [Batch, 16]
        
        # --- 投影 ---
        proj = self.proj_head(embedding)       # [Batch, 16]
        return embedding, proj

# ============================================================
# 3. 核心：数据增强与 Loss
# ============================================================
def augment_data(x):
    """ 数据增强：给坐标加入随机抖动 (Jittering) """
    noise = torch.randn_like(x) * 0.05  # 5% 的噪声
    return x + noise

def contrastive_loss(z_i, z_j, temperature=0.05):
    """ NT-Xent Loss (SimCLR的核心) """
    batch_size = z_i.size(0)
    
    # 1. 拼接两组特征 (原始图特征 + 增强图特征)
    z = torch.cat([z_i, z_j], dim=0)
    
    # 2. 计算相似度矩阵 (Cosine Similarity)
    z = F.normalize(z, dim=1)
    sim_matrix = torch.mm(z, z.t()) / temperature
    
    # 3. 构造标签：每个图的正样本是它的增强版
    # (i, i+batch_size) 和 (i+batch_size, i) 是正对
    sim_i_j = torch.diag(sim_matrix, batch_size)
    sim_j_i = torch.diag(sim_matrix, -batch_size)
    
    # 把正样本的相似度拿出来
    positives = torch.cat([sim_i_j, sim_j_i], dim=0)
    
    # 4. 计算 Loss (InfoNCE)
    # 分母是所有样本的相似度之和，分子是正样本的相似度
    # 目标：让正样本相似度最大，负样本相似度最小
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    # 排除自己和自己的相似度
    negatives = sim_matrix[~mask].view(2 * batch_size, -1)
    
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z.device) # 正样本在第0列
    
    return F.cross_entropy(logits, labels)

# ============================================================
# 4. 训练
# ============================================================
model = ContrastiveGNN(in_channels=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("🚀 开始图对比学习 (SimCLR)...")

for epoch in range(2000): # 对比学习通常需要更久
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # 1. 原始图的特征
        _, z1 = model(batch.x, batch.edge_index, batch.batch)
        
        # 2. 增强图的特征 (给 x 加噪声，拓扑结构不变)
        # 这里的逻辑是：即便坐标歪了，它还是同一个形状
        x_aug = augment_data(batch.x)
        _, z2 = model(x_aug, batch.edge_index, batch.batch)
        
        # 3. 计算 Loss
        loss = contrastive_loss(z1, z2)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")

print("✅ 训练完成！")

# ============================================================
# 5. 验证聚类
# ============================================================
model.eval()
all_embeddings = []
true_labels = []

with torch.no_grad():
    for batch in loader: # 用 loader 遍历一次
        batch = batch.to(device)
        # 注意：最后聚类用的是 encoder 输出的 embedding，不是 projection head 的输出
        emb, _ = model(batch.x, batch.edge_index, batch.batch)
        all_embeddings.append(emb.cpu().numpy())
        true_labels.append(batch.y.cpu().numpy())

X = np.vstack(all_embeddings)
y_true = np.concatenate(true_labels)

# K-Means 聚类
kmeans = KMeans(n_clusters=2).fit(X)
y_pred = kmeans.labels_

# 可视化
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(6, 5))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
plt.title("Graph Contrastive Learning Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# ============================================================
# 6. 保存模型 (Encoder Only)
# ============================================================
print("💾 正在保存模型...")

# 我们只保存 state_dict (参数字典)
# 注意：虽然 ContrastiveGNN 包含了 proj_head，但在推理时我们只需要 conv1 和 conv2
# 保存整个 model.state_dict() 是最安全的
SAVE_PATH = "model_contrastive.pt"
torch.save(model.state_dict(), SAVE_PATH)

print(f"✅ 模型已保存至: {SAVE_PATH}")
print("👉 下一步：使用 'predict_contrastive.py' 进行新文件的推理。")