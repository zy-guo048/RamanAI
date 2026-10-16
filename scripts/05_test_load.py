
from pathlib import Path
from raman_utils import load_spectrum

# 1. 获取项目根目录
project_dir = Path(__file__).resolve().parent.parent

# 2. 设置光谱文件路径
file_path = project_dir / "data/raw/sample_raw.txt"

# 3. 调用我们自己编写的函数
spectrum = load_spectrum(file_path)

# 4. 检查读取结果
print("光谱读取成功！")

print(
    "数据点数量:",
    len(spectrum.spectral_axis)
)

print(
    "波数范围:",
    spectrum.spectral_axis.min(),
    "~",
    spectrum.spectral_axis.max()
)

print(
    "前5个强度值:",
    spectrum.spectral_data[:5]
)
