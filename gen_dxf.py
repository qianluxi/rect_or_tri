import os
import math
import random
import ezdxf

# ============================================================
# 配置
# ============================================================
SAVE_DIR = "data/dxf"
NUM_SAMPLES = 200  # 总样本数

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ============================================================
# 几何辅助函数
# ============================================================
def rotate_point(x, y, angle_deg):
    """ 将点 (x, y) 绕原点 (0,0) 旋转 angle_deg 度 """
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    # 旋转公式
    new_x = x * cos_a - y * sin_a
    new_y = x * sin_a + y * cos_a
    return new_x, new_y

# ============================================================
# 核心生成逻辑
# ============================================================
def create_shape(filename, shape_type):
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    # 随机中心点 (ox, oy)
    ox, oy = random.uniform(0, 100), random.uniform(0, 100)
    
    points = []
    
    if shape_type == "triangle":
        # === 生成不等边三角形 ===
        # 方法：在圆周上随机取3个角度，且每个角的半径也不一样
        
        # 1. 随机生成3个角度并排序，确保画线顺序顺畅，不会交叉
        angles = sorted([random.uniform(0, 360) for _ in range(3)])
        
        # 2. 为每个角生成不同的半径 (2.0 ~ 6.0)
        radii = [random.uniform(2.0, 6.0) for _ in range(3)]
        
        # 3. 转换为坐标
        local_points = []
        for ang, r in zip(angles, radii):
            lx = r * math.cos(math.radians(ang))
            ly = r * math.sin(math.radians(ang))
            local_points.append((lx, ly))
            
        # 4. 加上中心点偏移
        points = [(lx + ox, ly + oy) for lx, ly in local_points]

    else:
        # === 生成旋转矩形 ===
        # 1. 随机长宽 (长宽比随机)
        w = random.uniform(3.0, 8.0)
        h = random.uniform(2.0, 5.0)
        
        # 2. 定义未旋转的四个角 (相对于中心)
        # 顺序：左下 -> 右下 -> 右上 -> 左上 (形成闭环)
        local_corners = [
            (-w/2, -h/2),
            ( w/2, -h/2),
            ( w/2,  h/2),
            (-w/2,  h/2)
        ]
        
        # 3. 随机旋转角度 (0~360度)
        rotation_angle = random.uniform(0, 360)
        
        # 4. 旋转并平移
        points = []
        for lx, ly in local_corners:
            rx, ry = rotate_point(lx, ly, rotation_angle)
            points.append((rx + ox, ry + oy))
    
    # === 画线 (首尾相连) ===
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i+1) % len(points)] # 连回起点
        msp.add_line(p1, p2)
        
    doc.saveas(filename)

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print(f"🔨 开始生成 {NUM_SAMPLES} 个随机形状 (不规则三角形 & 旋转矩形)...")
    
    # 清空旧数据（可选）
    # for f in os.listdir(SAVE_DIR): os.remove(os.path.join(SAVE_DIR, f))

    for i in range(NUM_SAMPLES):
        # 前一半是三角形(label=0)，后一半是矩形(label=1)
        is_rect = i >= NUM_SAMPLES // 2
        label = "rect" if is_rect else "tri"
        
        # 文件名
        fname = os.path.join(SAVE_DIR, f"{label}_{i}.dxf")
        
        # 生成
        create_shape(fname, "rectangle" if is_rect else "triangle")

    print(f"✅ 生成完成！保存在: {SAVE_DIR}")
    print("👉 提示：现在可以重新运行 '2_dxf_to_graph_norm.py' 来处理这些更复杂的数据了。")