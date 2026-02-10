import torch
import os
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import random

# ============================================================
# 1. 配置
# ============================================================
DATA_PATH = "data/dataset_normalized.pt"
ENCODER_PATH = "checkpoints/encoder.pt"
SAVE_PATH = "checkpoints/classifier.pt"

if not os.path.exists("checkpoints"): os.makedirs("checkpoints")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# 2. 定义完整模型 (Encoder + Classifier)
# ============================================================
class FineTunedModel(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Encoder 部分 (必须与预训练时一致)
        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, 16)
        
        # Classifier 部分 (新增的分类头)
        self.classifier = torch.nn.Linear(16, 2) # 16维特征 -> 2类 (Tri/Rect)

    def forward(self, x, edge_index, batch):
        # Encoder 前向传播
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        embedding = global_mean_pool(x, batch)
        
        # Classifier 前向传播
        logits = self.classifier(embedding)
        return logits

# ============================================================
# 3. 加载数据与模型
# ============================================================
dataset = torch.load(DATA_PATH)
# ✅ 正确做法：原地打乱列表
random.shuffle(dataset) 
train_subset = dataset[:1000]

# 🔍 [诊断] 打印一下看看样本分布对不对
labels = [data.y.item() for data in train_subset]
num_tri = labels.count(0)
num_rect = labels.count(1)
print(f"📊 微调样本分布: 柱={num_tri}, 墙={num_rect}")
if num_tri == 0 or num_rect == 0:
    print("⚠️ 警告：样本极度不平衡，训练将失败！")

loader = DataLoader(train_subset, batch_size=32, shuffle=True)

model = FineTunedModel(in_channels=5).to(device)

# 加载预训练好的 Encoder 权重
if os.path.exists(ENCODER_PATH):
    state_dict = torch.load(ENCODER_PATH, map_location=device)
    # 过滤掉 proj_head 的权重，只加载 conv 层
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("✅ 成功加载预训练 Encoder")
else:
    print("❌ 未找到预训练模型，请先运行 Step 3")
    exit()

# ============================================================
# 4. 微调训练 (Linear Probing)
# ============================================================
# 冻结 Encoder 参数 (只训练分类头)       需要解冻，让所有参数都参与训练
# for param in model.conv1.parameters(): param.requires_grad = False
# for param in model.conv2.parameters(): param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("🚀 开始微调分类头 (Linear Probing)...")

for epoch in range(100): # 很快就能收敛
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(logits, batch.y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        pred = logits.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
        
    if epoch % 10 == 0:
        acc = correct / total
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f} | Acc: {acc*100:.2f}%")

# ============================================================
# 5. 保存最终模型
# ============================================================
torch.save(model.state_dict(), SAVE_PATH)
print(f"💾 最终模型已保存至: {SAVE_PATH}")
print("👉 现在可以使用 '5_predict_final.py' 直接预测新文件了！")