# 导入所需库
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 录入数据 ----------------------
# 训练轮数 Epoch
epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
          110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
          210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
          310, 320, 330, 340, 350, 360, 370, 380, 390, 400,
          410, 420, 430, 440, 450, 460, 470, 480, 490, 500,
          510, 520, 530, 540, 550, 560, 570, 580, 590]

# 对应损失值 Loss
losses = [2.3830, 1.0868, 1.0265, 1.0016, 0.9843, 0.9792, 0.9682, 0.9641, 0.9521, 0.9452, 0.9391,
          0.9318, 0.9312, 0.9267, 0.9233, 0.9290, 0.9287, 0.9238, 0.9216, 0.9113, 0.9166,
          0.9202, 0.9109, 0.9115, 0.9160, 0.9148, 0.9145, 0.9096, 0.9114, 0.9081, 0.9090,
          0.9115, 0.9101, 0.9052, 0.9091, 0.9056, 0.9058, 0.9021, 0.9002, 0.9019, 0.8987,
          0.8992, 0.9051, 0.8999, 0.8969, 0.8997, 0.8946, 0.8917, 0.9065, 0.8975, 0.8998,
          0.8946, 0.8957, 0.8939, 0.8926, 0.9007, 0.8923, 0.8884, 0.8936, 0.8867]

# ---------------------- 2. 核心修改：筛选从 Epoch 50 开始的数据 ----------------------
# 找到 Epoch=50 对应的索引位置
start_idx = epochs.index(50)
# 切片截取数据
filtered_epochs = epochs[start_idx:]
filtered_losses = losses[start_idx:]

# ---------------------- 3. 配置绘图样式，解决中文/负号显示问题 ----------------------
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
# 创建画布
plt.figure(figsize=(12, 6), dpi=100)

# ---------------------- 4. 绘制筛选后的损失曲线 ----------------------
plt.plot(filtered_epochs, filtered_losses, color="#2E86AB", linewidth=2, marker="o", markersize=4,
         markerfacecolor="#A23B72", markeredgecolor="white", label="训练损失")

# ---------------------- 5. 图表标注与美化 ----------------------
plt.title("模型训练损失值变化曲线（Epoch ≥ 50）", fontsize=16, pad=20)
plt.xlabel("训练轮数 (Epoch)", fontsize=12)
plt.ylabel("损失值 (Loss)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)

# 标注筛选后数据中的最小损失值点
min_loss = min(filtered_losses)
min_epoch = filtered_epochs[filtered_losses.index(min_loss)]
plt.annotate(f"最小Loss: {min_loss:.4f}\nEpoch: {min_epoch}",
             xy=(min_epoch, min_loss),
             xytext=(min_epoch+30, min_loss+0.015),
             arrowprops=dict(arrowstyle="->", color="red"),
             fontsize=10, color="red")

# 适配新数据范围调整坐标轴
plt.xlim(45, 600)
plt.ylim(0.88, 1.0)

# ---------------------- 6. 输出图表 ----------------------
plt.tight_layout()
plt.savefig("epoch_loss_curve_50+.png", dpi=300)
plt.show()