import os
import math
import random
import ezdxf

# ============================================================
# 配置
# ============================================================
SAVE_DIR = "data/dxf"
NUM_SAMPLES = 20000  # 总样本数

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ============================================================
# 几何辅助函数
# ============================================================
def rotate_point(x, y, angle_deg):
    """将点 (x, y) 绕原点 (0,0) 旋转 angle_deg 度"""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    new_x = x * cos_a - y * sin_a
    new_y = x * sin_a + y * cos_a
    return new_x, new_y

# ============================================================
# 核心生成逻辑
# ============================================================
def create_shape(filename, label_type):
    doc = ezdxf.new()
    msp = doc.modelspace()
    ox, oy = random.uniform(0, 100), random.uniform(0, 100)
    
    # 根据标签类型设置长宽比
    if label_type == "fat":
        # 长宽比 <= 4 (胖矩形)
        ratio = random.uniform(1.0, 4.0)
    else:  # thin
        # 长宽比 > 4 (细长矩形)
        ratio = random.uniform(4.1, 10.0)
    
    # 基准尺寸
    base_size = random.uniform(2.0, 5.0)
    
    # 随机决定谁是长边
    if random.random() > 0.5:
        width = base_size * ratio
        height = base_size
    else:
        width = base_size
        height = base_size * ratio
    
    # 创建矩形的四个角点（以原点为中心）
    half_w = width / 2
    half_h = height / 2
    local_corners = [
        (-half_w, -half_h),  # 左下
        (half_w, -half_h),   # 右下
        (half_w, half_h),    # 右上
        (-half_w, half_h)    # 左上
    ]
    
    # 随机旋转角度 (0~360度)
    rotation_angle = random.uniform(0, 360)
    
    # 旋转并平移
    points = []
    for lx, ly in local_corners:
        rx, ry = rotate_point(lx, ly, rotation_angle)
        points.append((rx + ox, ry + oy))
    
    # 画线（首尾相连形成闭合矩形）
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        msp.add_line(p1, p2)
    
    # 保存文件
    doc.saveas(filename)

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print(f"🔨 开始生成 {NUM_SAMPLES} 个随机矩形...")
    
    # 清空旧数据（可选）
    # for f in os.listdir(SAVE_DIR):
    #     if f.endswith(".dxf"):
    #         os.remove(os.path.join(SAVE_DIR, f))

    for i in range(NUM_SAMPLES):
        # 前一半是胖矩形(fat)，后一半是细长矩形(thin)
        is_thin = i >= NUM_SAMPLES // 2
        label = "thin" if is_thin else "fat"
        
        # 文件名
        fname = os.path.join(SAVE_DIR, f"{label}_{i}.dxf")
        
        # 生成
        create_shape(fname, label)

    print(f"✅ 生成完成！保存在: {SAVE_DIR}")
    print(f"生成了 {NUM_SAMPLES // 2} 个胖矩形(fat)和 {NUM_SAMPLES - NUM_SAMPLES // 2} 个细长矩形(thin)")