import os
import math
import ezdxf
import torch
import numpy as np
from torch_geometric.data import Data

# ============================================================
# 配置路径
# ============================================================
DXF_DIR = "data/dxf"
PROCESSED_FILE = "data/dataset_normalized.pt"  # 保存为新文件

if not os.path.exists(DXF_DIR):
    print(f"❌ 找不到文件夹 {DXF_DIR}，请先运行步骤1造数据。")
    exit()

# ============================================================
# 辅助函数
# ============================================================
def get_lines(dxf_path):
    """ 读取 DXF 中的所有 LINE 实体 """
    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        lines = []
        for e in msp:
            if e.dxftype() == "LINE":
                start = (e.dxf.start.x, e.dxf.start.y)
                end = (e.dxf.end.x, e.dxf.end.y)
                lines.append((start, end))
        return lines
    except Exception as e:
        print(f"读取错误 {dxf_path}: {e}")
        return []

def lines_touch(l1, l2, tol=0.1):
    """ 判断两条线是否相连 """
    for p1 in l1:
        for p2 in l2:
            dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
            if dist < tol:
                return True
    return False

# ============================================================
# 核心逻辑：归一化处理
# ============================================================
def process_dxf_normalized(path, label):
    lines = get_lines(path)
    if len(lines) == 0:
        return None

    # 1. 收集所有点，计算中心点 (Centroid)
    all_points = []
    for p1, p2 in lines:
        all_points.append(p1)
        all_points.append(p2)
    
    # 转换为 numpy 方便计算
    pts = np.array(all_points)
    
    # 计算几何中心 (cx, cy)
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])

    # 2. 计算缩放尺度 (Scale)
    # 也就是找出离中心最远的点，用这个距离做分母
    # 这样所有图形都会被缩放到半径为 1 的圆内
    distances = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    scale = np.max(distances)
    if scale < 1e-6: scale = 1.0  # 防止除以0

    # 3. 构建特征 (归一化后的坐标)
    # 特征维度 = 5: [归一化长度, 归一化x1, 归一化y1, 归一化x2, 归一化y2]
    node_features = []
    
    for (p1, p2) in lines:
        # 原始长度
        raw_len = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
        
        # 归一化处理： (坐标 - 中心) / 尺度
        norm_len = raw_len / scale
        n_x1 = (p1[0] - cx) / scale
        n_y1 = (p1[1] - cy) / scale
        n_x2 = (p2[0] - cx) / scale
        n_y2 = (p2[1] - cy) / scale
        
        #node_features.append([norm_len, n_x1, n_y1, n_x2, n_y2])
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

    # 4. 构建边 (拓扑结构)
    num_nodes = len(lines)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if lines_touch(lines[i], lines[j]):
                edges.append([i, j])
                edges.append([j, i])
    
    # 如果没有边（比如只有一条线），加自环防止报错
    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 5. 标签
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

# ============================================================
# 主执行流
# ============================================================
if __name__ == "__main__":
    print("🔄 开始处理数据 (启用相对坐标归一化)...")
    
    data_list = []
    files = os.listdir(DXF_DIR)
    
    for f in files:
        if not f.endswith(".dxf"): continue
        
        # 标签逻辑：矩形=1，三角形=0
        label = 1 if "rect" in f else 0
        path = os.path.join(DXF_DIR, f)
        
        graph = process_dxf_normalized(path, label)
        if graph:
            data_list.append(graph)

    # 保存
    torch.save(data_list, PROCESSED_FILE)
    print(f"✅ 处理完成！")
    print(f"📊 保存路径: {PROCESSED_FILE}")
    print(f"🔢 图形总数: {len(data_list)}")
    print(f"ℹ️  特征维度: 5 (Length, x1, y1, x2, y2 - 全部归一化)")