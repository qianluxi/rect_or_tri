import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

# 1. 加载数据
dataset = torch.load("data/dataset_normalized.pt")
# 打乱数据
random_indices = torch.randperm(len(dataset))
dataset = [dataset[i] for i in random_indices]

# 80% 训练，20% 测试
train_size = int(len(dataset) * 0.8)
train_loader = DataLoader(dataset[:train_size], batch_size=16, shuffle=True)
test_loader = DataLoader(dataset[train_size:], batch_size=16)

# 2. 定义简单的 GNN 模型
class ShapeClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 输入特征维度是2 (长度, 常数)，隐藏层 16
        self.conv1 = GCNConv(5, 16)
        self.conv2 = GCNConv(16, 16)
        # 最终分类为 2 类 (三角形 vs 矩形)
        self.classifier = torch.nn.Linear(16, 2)

    def forward(self, x, edge_index, batch):
        # 第一层图卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二层图卷积
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # 全局池化：把一张图里所有节点特征取平均，变成一个向量
        x = global_mean_pool(x, batch)
        
        # 分类
        return self.classifier(x)

# 3. 训练流程
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ShapeClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

print("🚀 开始训练...")
for epoch in range(200):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

# 4. 测试流程
model.eval()
correct = 0
total = 0
for batch in test_loader:
    batch = batch.to(device)
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)

print(f"✅ 测试集准确率: {100 * correct / total:.2f}%")
torch.save(model.state_dict(), "model.pt")
print("💾 模型已保存为 model.pt")