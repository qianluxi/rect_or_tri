import torch
import os
import math
import numpy as np
import ezdxf
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

# ============================================================
# 1. 模型定义 (与微调时一致)
# ============================================================
class FineTunedModel(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, 16)
        self.classifier = torch.nn.Linear(16, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        embedding = global_mean_pool(x, batch)
        logits = self.classifier(embedding)
        return logits

# ============================================================
# 2. 数据处理函数 (必须与训练时完全一致)
# ============================================================
def process_single_dxf(dxf_path):
    if not os.path.exists(dxf_path): return None
    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        lines = []
        for e in msp:
            if e.dxftype() == "LINE":
                lines.append(((e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)))
    except: return None
    
    if len(lines) == 0: return None

    # 归一化
    all_points = [p for l in lines for p in l]
    pts = np.array(all_points)
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    distances = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    scale = np.max(distances) if np.max(distances) > 1e-6 else 1.0

    # 特征构建 (修复了双层循环 Bug)
    node_features = []
    for (p1, p2) in lines:
        raw_len = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
        nx1, ny1 = (p1[0] - cx) / scale, (p1[1] - cy) / scale
        nx2, ny2 = (p2[0] - cx) / scale, (p2[1] - cy) / scale
        
        pt1 = (nx1, ny1)
        pt2 = (nx2, ny2)
        if pt1 > pt2: start_pt, end_pt = pt2, pt1
        else: start_pt, end_pt = pt1, pt2
            
        feat = [raw_len / scale, start_pt[0], start_pt[1], end_pt[0], end_pt[1]]
        node_features.append(feat)
    
    x = torch.tensor(node_features, dtype=torch.float)

    # 边构建
    edges = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            connected = False
            for p1 in lines[i]:
                for p2 in lines[j]:
                    if math.hypot(p1[0]-p2[0], p1[1]-p2[1]) < 0.1:
                        connected = True
            if connected:
                edges.append([i, j])
                edges.append([j, i])
    
    if not edges: edge_index = torch.zeros((2, 0), dtype=torch.long)
    else: edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, batch=torch.zeros(x.size(0), dtype=torch.long))

# ============================================================
# 3. 主程序
# ============================================================
if __name__ == "__main__":
    MODEL_PATH = "checkpoints/classifier.pt"
    TEST_FILE = "data/dxf/rect_10000.dxf" # 替换你想测的文件
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FineTunedModel(in_channels=5).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ 成功加载最终模型 (classifier.pt)")
    else:
        print("❌ 模型未找到，请先运行 Step 4 微调")
        exit()
    model.eval()

    print(f"📄 正在识别: {TEST_FILE}")
    data = process_single_dxf(TEST_FILE)
    
    if data:
        data = data.to(device)
        with torch.no_grad():
            logits = model(data.x, data.edge_index, data.batch)
            probs = F.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            
            label_map = {0: "三角形", 1: "矩形"}
            print("-" * 30)
            print(f"🤖 结果: {label_map[pred_class]}")
            print(f"📊 置信度: {probs[0][pred_class]*100:.2f}%")
            print("-" * 30)
    else:
        print("❌ 文件解析失败")