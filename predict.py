import torch
import os
import math
import numpy as np
import ezdxf
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

# ============================================================
# 1. 定义完全相同的模型架构 (必须与训练时一致)
# ============================================================
class ShapeClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # ⚠️ 注意：输入维度必须是 5 (因为我们用了归一化后的5个特征)
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
# 2. 定义完全相同的数据预处理逻辑
#    (必须把 2_dxf_to_graph_norm.py 里的逻辑搬过来)
# ============================================================
def process_single_dxf(dxf_path):
    # --- 读取 DXF ---
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

    # --- 归一化 (核心!) ---
    all_points = [p for l in lines for p in l]
    pts = np.array(all_points)
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1]) # 中心
    distances = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    scale = np.max(distances)
    if scale < 1e-6: scale = 1.0

    # --- 构建特征 ---
    node_features = []
    for (p1, p2) in lines:
        raw_len = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
        # [归一化长度, 归一化x1, y1, x2, y2]
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

    # --- 构建边 ---
    edges = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            # 判断连接
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

    # 这里的 y 是假的，因为预测时我们不知道答案，但 PyG 需要这个结构
    # batch 是必须的，指明这属于第 0 张图
    return Data(x=x, edge_index=edge_index, batch=torch.zeros(x.size(0), dtype=torch.long))

# ============================================================
# 3. 假想的主程序：实际使用
# ============================================================
if __name__ == "__main__":
    # A. 准备工作
    MODEL_PATH = "model.pt"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # B. 加载模型
    model = ShapeClassifier().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ 成功加载大脑 (model.pt)")
    else:
        print("❌ 没找到 model.pt，请先训练！")
        exit()
    
    model.eval() # 切换到评估模式

    # C. 模拟用户上传了一个文件
    # 我们随便找一个生成好的文件来测试
    TEST_FILE = "data/dxf/rect_150.dxf" # 找一个矩形测试
    if not os.path.exists(TEST_FILE):
        print("测试文件不存在，请检查路径")
        exit()

    print(f"📄 正在识别文件: {TEST_FILE} ...")

    # D. 处理数据
    data = process_single_dxf(TEST_FILE)
    if data is None:
        print("无法解析该文件")
        exit()
    
    data = data.to(DEVICE)

    # E. 预测
    with torch.no_grad():
        # 模型输出的是两个数字 [score_tri, score_rect]
        logits = model(data.x, data.edge_index, data.batch)
        
        # 转化为概率 (Softmax)
        probs = F.softmax(logits, dim=1)
        
        # 获取最大概率的类别
        pred_class = logits.argmax(dim=1).item()
        confidence = probs[0][pred_class].item()

    # F. 输出人话
    label_map = {0: "三角形 (Triangle)", 1: "矩形 (Rectangle)"}
    print("-" * 30)
    print(f"🤖 识别结果: {label_map[pred_class]}")
    print(f"📊 置信度: {confidence * 100:.2f}%")
    print("-" * 30)