
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ramanspy as rp

# 1. 定位项目目录和光谱文件
project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data/raw/sample_raw.txt"

# 2. 读取原始光谱数据
data = np.loadtxt(
    file_path,
    delimiter=",",
    comments="#"
)

# 3. 将两列数据分别取出来
raman_shift = data[:, 0]
intensity = data[:, 1]

# 4. 观察数据
print("前5个数据点：")
print(data[:5])

print("光谱数据点数：", len(raman_shift))
print("最小波数：", np.min(raman_shift))
print("最大波数：", np.max(raman_shift))

# 5. 创建 RamanSPy 光谱对象
spectrum = rp.Spectrum(
    intensity,
    raman_shift
)

# 6. 绘制原始光谱
ax = spectrum.plot(title="My First Raman Spectrum")

# 7. 保存绘图结果
output_dir = project_dir / "results"
output_dir.mkdir(exist_ok=True)

ax.figure.savefig(
    output_dir / "01_raw.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()
