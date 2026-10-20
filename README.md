# RamanAI

RamanAI 是一个面向科研演示与基础分析的拉曼光谱预处理和候选峰分析系统。当前 V1 使用 Streamlit 提供单条 RRUFF RAW TXT 光谱的上传、预处理、候选峰检测、峰特征展示、CSV 下载及简体中文/英语双语界面。

> 当前系统输出的是算法检测到的候选峰，不等同于矿物特征归属，也不代表已经建立或独立验证了矿物识别准确率。

## 核心功能

- 在内存中读取并校验 RRUFF 风格的两列 TXT 光谱，不写入原始数据目录或 `results/`。
- 使用 Savitzky–Golay（SG）方法去噪，并使用 AsLS 方法进行基线校正。
- 在未归一化的基线校正光谱上检测候选峰和提取峰特征。
- 对基线校正光谱执行最大值归一化，用于显示及后续分析。
- 同时显示原始、SG 去噪、AsLS 基线校正、最大值归一化光谱及候选峰标记。
- 支持调节候选峰突出度阈值，并显示本次实际使用的参数。
- 显示候选峰特征表，并在内存中生成适合 Windows Excel 打开的 UTF-8-SIG CSV。
- 支持简体中文和英语即时切换，覆盖主要界面、图表、峰表、参数说明及输入错误提示。
- 上传文件或突出度改变后清除旧分析结果；单独切换语言时保留当前文件、突出度和分析结果。

## 项目结构

```text
Raman AI/
├─ app.py                         # Streamlit V1 单光谱分析界面
├─ requirements.txt              # Python 运行依赖
├─ raman_app/
│  ├─ __init__.py
│  ├─ upload.py                   # 上传内容的内存解析与输入校验
│  ├─ pipeline.py                 # 单条光谱的统一分析 Pipeline
│  ├─ export.py                   # 候选峰 CSV 的内存导出
│  └─ i18n.py                     # 中英文翻译字典、语言回退与错误翻译
├─ scripts/
│  ├─ raman_utils.py              # 可复用的预处理、寻峰、特征与归一化函数
│  └─ 01_*.py ～ 06_*.py          # 实验 1–6 的历史脚本
├─ tests/
│  ├─ test_upload.py
│  ├─ test_pipeline.py
│  ├─ test_export.py
│  ├─ test_i18n.py
│  └─ test_app_helpers.py
├─ data/raw/sample_raw.txt        # 本地存在时用于真实样本集成测试与演示
├─ docs/
│  ├─ EXPERIMENT_06_SUMMARY.md
│  ├─ EXPERIMENT_07_SUMMARY.md
│  └─ DEMO_GUIDE.md
└─ results/                       # 历史实验结果；V1 上传分析不会自动写入
```

## Windows 安装与启动

当前开发环境已使用 Python 3.13.3 验证。建议在项目目录内创建独立虚拟环境。

```powershell
cd "D:\【程序】\Raman AI"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

如果 PowerShell 不允许执行激活脚本，可以不激活环境，直接运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

启动后，在浏览器中打开 Streamlit 输出的本地地址。停止服务时在终端按 `Ctrl+C`。

## 中英双语界面

界面默认使用简体中文。页面顶部标题行右侧提供“⚙️ 设置 / Settings”按钮，点击后通过 Streamlit `st.dialog()` 打开设置窗口。

在设置窗口中选择：

- `简体中文 / Simplified Chinese`
- `English / 英语`

选择后整个界面立即切换，不需要重启应用。语言状态保存在当前浏览器会话的 `st.session_state` 中，因此普通 Streamlit rerun 后仍会保持；关闭浏览器会话或重启服务后重新使用默认简体中文。

语言切换覆盖页面标题与说明、上传控件、突出度设置、按钮、状态与错误提示、分析参数、Matplotlib 图表标题与坐标轴、图例、候选峰表列名、科学限制说明及 CSV 下载按钮。技术缩写和单位按科学惯例保留。

切换语言只刷新界面文本，不会重新执行光谱 Pipeline，也不会清除当前上传文件、用户设置的 `prominence` 或已经展示的分析结果。CSV 的原始字段名、数值和计算过程不随界面语言改变。

## 输入数据格式

上传文件必须满足以下要求：

- 文件扩展名为 `.txt`，且文件名不能包含路径。
- 文本编码为 UTF-8；支持带 UTF-8 BOM 的文件。
- 文件大小不超过 `5 × 1024 × 1024` 字节（界面提示为 5 MB）。
- 数据为逗号分隔的两列有限数值；第一列为 Raman Shift，单位约定为 cm⁻¹，第二列为 Intensity，通常记为 a.u.。
- 以 `#` 开头的整行作为注释忽略，空行也会忽略。
- Raman Shift 必须严格递增，不能重复或降序。
- 至少包含 21 个有效采样点，以满足 V1 的 SG 窗口长度要求。
- 空文件、仅含注释的文件、列数错误、非数值、NaN、Inf 或不支持的编码会被拒绝。

示例：

```text
##NAMES=Example
# Raman Shift (cm-1), Intensity (a.u.)
100.0,1250.0
101.0,1268.5
102.0,1241.2
```

## 分析流程与参数

V1 Pipeline 的顺序和默认参数如下：

1. 解析并校验上传内容。
2. SG 去噪：`window_length=21`，`polyorder=3`。
3. AsLS 基线校正：`lam=1e6`，`p=0.01`。
4. 在未归一化的基线校正光谱上检测候选峰：`prominence=1000`，`distance=10`。
5. 从同一条基线校正光谱提取候选峰特征。
6. 对基线校正光谱进行 `max` 最大值归一化，生成独立的显示结果。

`distance=10` 表示相邻候选峰至少间隔 10 个采样点，不是 10 cm⁻¹。最大值归一化不会参与上述 `prominence` 阈值的计算，因此不会改变原始强度阈值的含义。

`scripts/raman_utils.py` 中的历史底层函数 `denoise_spectrum()` 仍保留 `window_length=9` 的函数默认值；V1 Pipeline 为保持实验设计，始终显式传入 `window_length=21`，没有修改历史函数行为。

## 调节突出度阈值

页面中的“候选峰突出度阈值”默认值为 1000。较低强度光谱可以尝试降低阈值，但阈值越低，越可能把噪声波动识别为候选峰。

该参数不会被系统自动优化。每次修改后需要重新点击“开始分析”，页面会基于新阈值重新分析；结果参数区会显示实际使用的 `prominence`。阈值必须是大于或等于 0 的有限数值。

## 峰特征表与 CSV 下载

页面表格重点显示峰编号、峰位、校正后强度、突出度和半突出度宽度。下载的 CSV 保留以下完整字段：

- `Peak_ID`
- `Position_cm-1`
- `Intensity`
- `Prominence`
- `Half_Prominence_Width_cm-1`
- `Left_cm-1`
- `Right_cm-1`
- `Width_Height`

CSV 使用 UTF-8-SIG 编码，文件名格式为 `<上传文件名主体>_candidate_peak_features.csv`。没有检测到候选峰时，页面仍显示空表，下载文件仅包含有效表头。

峰宽由 `scipy.signal.peak_widths(..., rel_height=0.5)` 计算，表示半突出度宽度。它不是经过单峰拟合得到的物理 FWHM，不能在所有情况下直接等同于 FWHM。

## 自动化测试

在项目根目录运行完整测试：

```powershell
python -m unittest discover -s tests -v
```

未激活虚拟环境时可运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖上传校验、Pipeline 参数与数值一致性、默认及自定义突出度、零候选峰、真实样本回归、CSV 编码与空表导出、界面请求状态失效，以及中英文翻译键完整性、非法语言回退、语言切换状态保持、科学计算与 CSV 的语言无关性。真实样本测试仅在 `data/raw/sample_raw.txt` 存在时运行。

可使用以下命令检查已安装依赖之间是否存在冲突：

```powershell
python -m pip check
```

## 科学限制与注意事项

- 候选峰结果依赖采样质量、基线、噪声水平和人工设定的突出度，不是自动优化后的最优结果。
- 候选峰尚未进行矿物峰归属、参考谱库确认或人工专家确认。
- 当前真实样本回归用于验证软件流程和历史参数的一致性，不证明材料分类能力。
- 当前系统没有经过独立数据集上的矿物识别准确率验证，也没有分类模型。
- 半突出度宽度受离散采样、峰重叠和局部背景影响，不应无条件解释为物理 FWHM。
- V1 面向单条光谱交互分析，不会自动处理约 11,415 条的外部数据集。
- 原始上传内容、分析结果和下载 CSV 均不由 Pipeline 自动写入 `data/` 或 `results/`。

## V1 范围与未来扩展

**V1 已实现**：单条 TXT 上传、格式校验、SG 去噪、AsLS 基线校正、最大值归一化、可调突出度寻峰、候选峰特征、绘图、中英文界面与错误提示、CSV 下载和自动化回归测试。

**尚未实现，可作为后续研究方向**：多文件批量上传、参考光谱相似度匹配、峰归属辅助、矿物分类模型、独立测试集性能评估、11,415 条外部光谱的数据治理与批处理、参数优化建议和多页面分析工作流。
