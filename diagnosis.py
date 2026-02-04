import torch
import os
import math
import numpy as np
import ezdxf
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

# ============================================================
# 模型定义 (保持不变)
# ============================================================
class ShapeClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(5, 16)
        self.conv2 = GCNConv(16, 16)
        self.classifier = torch.nn.Linear(16, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.classifier(x)

# ============================================================
# 诊断版处理函数 (增加了打印功能)
# ============================================================
def process_dxf_debug(dxf_path):
    print(f"\n🔍 [诊断] 正在解析: {dxf_path}")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    lines = []
    for e in msp:
        if e.dxftype() == "LINE":
            lines.append(((e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)))
    
    print(f"   --> 发现 {len(lines)} 条线段 (Lines)")

    # 归一化
    all_points = [p for l in lines for p in l]
    pts = np.array(all_points)
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    distances = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    scale = np.max(distances)
    if scale < 1e-6: scale = 1.0

    # 特征构建
    node_features = []
    for (p1, p2) in lines:
        raw_len = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
        feat = [
            raw_len / scale,
            (p1[0] - cx) / scale, (p1[1] - cy) / scale,
            (p2[0] - cx) / scale, (p2[1] - cy) / scale
        ]
        #node_features.append(feat)
        # --- 修改后的代码 (修复方向敏感性) ---
        for (p1, p2) in lines:
            raw_len = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
            
            # 归一化坐标
            nx1, ny1 = (p1[0] - cx) / scale, (p1[1] - cy) / scale
            nx2, ny2 = (p2[0] - cx) / scale, (p2[1] - cy) / scale

            # ⭐ 关键修改：保证点总是按固定顺序排列 (比如按 x 坐标排序，如果 x 相同按 y 排序)
            # 这样 A->B 和 B->A 都会变成 A-B
            pt1 = (nx1, ny1)
            pt2 = (nx2, ny2)
            
            if pt1 > pt2: # Python 元组比较：先比第一个元素，再比第二个
                start_pt, end_pt = pt2, pt1
            else:
                start_pt, end_pt = pt1, pt2
                
            # 特征向量现在是唯一的了
            feat = [
                raw_len / scale,
                start_pt[0], start_pt[1],  # x较小的那个点
                end_pt[0],   end_pt[1]     # x较大的那个点
            ]
            node_features.append(feat)   

    x = torch.tensor(node_features, dtype=torch.float)

    # 边构建 (关键点!)
    edges = []
    print("   --> 正在检查连接关系 (Tolerance=0.1)...")
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            connected = False
            # 检查两线是否相接
            for p1 in lines[i]:
                for p2 in lines[j]:
                    dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                    if dist < 0.1: # 阈值
                        connected = True
            if connected:
                edges.append([i, j])
                edges.append([j, i])
    
    # 打印拓扑信息
    num_edges = len(edges) // 2 # 除以2是因为无向图存了双向
    print(f"   --> 📊 拓扑结构诊断: 节点数={len(lines)}, 边数={num_edges}")
    
    if len(lines) == 4 and num_edges < 4:
        print("   ⚠️  警告: 这是一个矩形(4条线)，但边数少于4！说明有角断开了！")
        print("       GNN 会把它看成折线，极易误判为三角形。")
    
    if not edges: edge_index = torch.zeros((2, 0), dtype=torch.long)
    else: edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, batch=torch.zeros(x.size(0), dtype=torch.long))

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    MODEL_PATH = "model.pt"
    # 使用你刚才出错的那个文件
    TEST_FILE = "data/dxf/rect_150.dxf" 
    
    device = torch.device('cpu') # 调试用 CPU 足够
    model = ShapeClassifier().to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("❌ model.pt 不存在")
        exit()
    model.eval()

    data = process_dxf_debug(TEST_FILE)
    if data:
        with torch.no_grad():
            logits = model(data.x, data.edge_index, data.batch)
            probs = F.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            
            label_map = {0: "三角形 (Triangle)", 1: "矩形 (Rectangle)"}
            print("\n" + "="*30)
            print(f"🤖 最终预测: {label_map[pred_class]}")
            print(f"📊 概率分布: 三角形={probs[0][0]:.4f}, 矩形={probs[0][1]:.4f}")
            print("="*30)