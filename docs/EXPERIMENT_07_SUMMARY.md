# RamanAI 实验七总结

## 1. 实验目标

实验七在实验 1–6 已有拉曼光谱处理函数的基础上，构建 RamanAI V1 单光谱交互分析系统。目标包括：

- 从内存解析并校验用户上传的 RRUFF RAW TXT 光谱。
- 复用既有 SG 去噪、AsLS 基线校正、SciPy 寻峰、峰特征提取和最大值归一化算法。
- 通过统一 Pipeline 固定处理顺序并返回结构化结果。
- 使用 Streamlit 展示各处理阶段、候选峰和峰特征表。
- 支持可调突出度阈值、中英文异常提示和 UTF-8-SIG CSV 下载。
- 通过设置弹窗提供简体中文和英语两种界面语言，并保持分析状态不受语言切换影响。
- 通过自动化测试验证历史参数、真实样本结果和主要边界条件。

实验七没有增加矿物分类模型，也没有处理约 11,415 条的外部数据集。

## 2. 系统架构

```text
Streamlit 界面 app.py
        │
        ├─ raman_app.i18n：中英文界面文本与错误翻译
        ├─ 上传内容与文件名
        ▼
raman_app.pipeline.analyze_uploaded_spectrum()
        │
        ├─ raman_app.upload：内存解析与校验
        ├─ scripts.raman_utils：SG → AsLS → 寻峰 → 峰特征 → max 归一化
        └─ AnalysisResult：光谱、候选峰、特征和实际参数
        │
        ├─ app.py：绘图、参数与表格展示
        └─ raman_app.export：UTF-8-SIG CSV 内存导出
```

寻峰位于 AsLS 基线校正之后、最大值归一化之前。归一化结果用于独立显示和后续分析，不用于 `prominence` 阈值寻峰。

## 3. 模块职责

### 3.1 上传解析 `raman_app/upload.py`

- 接收二进制上传内容，不创建临时原始数据文件。
- 支持 UTF-8 与 UTF-8 BOM、`#` 注释行、空行及逗号分隔的两列数值。
- 校验文件非空、大小不超过 `5 × 1024 × 1024` 字节、数据有限、至少 21 点、Raman Shift 严格递增。
- 将第一列作为 Raman Shift、第二列作为 Intensity，返回 RamanSPy `Spectrum`。

### 3.2 分析 Pipeline `raman_app/pipeline.py`

- 校验 `.txt` 文件名，拒绝路径形式的文件名。
- 校验 `prominence` 为大于或等于 0 的有限数值。
- 显式使用 SG `window_length=21`、`polyorder=3`。
- 使用 AsLS `lam=1e6`、`p=0.01`。
- 在未归一化 corrected 光谱上使用 SciPy 寻峰；默认 `prominence=1000`，`distance=10` 个采样点。
- 提取半突出度宽度等峰特征，并对 corrected 光谱执行 `max` 归一化。
- 通过 `AnalysisResult` 返回各阶段光谱、候选峰、特征与实际参数，不写入 `results/`。

### 3.3 Streamlit 界面 `app.py`

- 上传单个 `.txt` 文件并通过“开始分析”按钮调用统一 Pipeline。
- 显示文件名、数据点数、候选峰数和实际分析参数。
- 绘制原始、SG 去噪、AsLS 基线校正、最大值归一化光谱，以及 corrected 光谱上的候选峰。
- 提供可调 `prominence` 输入，并提示低阈值可能引入噪声峰。
- 在标题行右侧提供“⚙️ 设置 / Settings”按钮，并使用 `st.dialog()` 显示语言设置窗口。
- 默认使用简体中文，可切换为英语；页面、图表、峰表、参数和已知输入错误随语言切换。
- 未知内部错误记录到服务器日志，不在页面暴露完整 traceback。
- 文件内容、文件名或突出度改变时清除旧结果，避免显示过期峰表。
- 语言选择保存在 `st.session_state`；切换语言只 rerun 界面，不重新运行 Pipeline，也不清除上传文件、突出度或现有结果。
- Matplotlib 字体按 `Microsoft YaHei`、`SimHei`、`DejaVu Sans` 的顺序回退，并关闭 Unicode 负号替换问题。

### 3.4 结果导出 `raman_app/export.py`

- 直接复用 `AnalysisResult.peak_features`，不重新运行寻峰。
- 生成 UTF-8-SIG CSV 字节，便于 Windows Excel 打开。
- 保留 8 个峰特征字段；零候选峰时生成只有表头的有效 CSV。
- 生成与上传文件可区分的下载名，不自动写入磁盘。

### 3.5 国际化 `raman_app/i18n.py`

- 集中维护 `zh_CN` 和 `en` 两套翻译字典，避免复制两份 Streamlit 应用。
- 默认语言为 `zh_CN`；非法或未知语言代码安全回退到简体中文。
- 为设置窗口提供双语语言选项，并为已知上传和参数错误提供对应英文信息。
- 保持技术缩写、科学参数和 CSV 原始字段不变；语言只影响用户可见文本。
- Matplotlib 的 `cm$^{-1}$` MathText 花括号在格式化模板中进行转义，确保中英文坐标轴均可正常生成。

## 4. 科学参数与字段解释

V1 的默认处理参数为：

| 环节 | 参数 |
|---|---|
| SG 去噪 | `window_length=21`，`polyorder=3` |
| AsLS 基线校正 | `lam=1e6`，`p=0.01` |
| 候选峰检测 | `prominence=1000`，`distance=10` 个采样点 |
| 峰宽 | `peak_widths(rel_height=0.5)` 的半突出度宽度 |
| 归一化 | `method="max"` |

`distance` 不是 cm⁻¹。`Half_Prominence_Width_cm-1` 是在离散 Raman Shift 轴上插值得到的半突出度宽度，不应无条件称为物理 FWHM。

为保持历史代码行为，`scripts/raman_utils.py` 中 `denoise_spectrum()` 的底层函数默认窗口仍为 9；V1 Pipeline 始终显式传入 21，因此 V1 实际使用的是 SG(21, 3)。

## 5. Annite 真实样本验证

本地 `data/raw/sample_raw.txt` 存在时，集成测试会执行以下验证：

1. 上传解析结果的 Raman Shift 与 Intensity 和历史 `load_spectrum()` 一致。
2. 使用 SG(21, 3)、AsLS(1e6, 0.01)、`prominence=1000`、`distance=10` 得到 8 个候选峰。
3. 候选峰位与历史回归值一致：267.406、388.149、534.344、667.586、714.981、765.538、934.406、1018.056 cm⁻¹。
4. 半突出度宽度按测试中保存的历史数值进行浮点容差比较。
5. 将同一样本的 `prominence` 降至 100 时，当前回归测试得到 17 个候选峰，用于确认参数确实影响寻峰结果。

上述验证证明的是当前软件流程与历史样本结果的一致性，不证明 Annite 自动识别能力，也不构成矿物分类准确率验证。

## 6. 自动化测试概况

测试目录包括：

- `tests/test_upload.py`：上传格式、大小、编码内容、有限值、递增轴、最少点数及真实样本读取一致性。
- `tests/test_pipeline.py`：处理顺序、参数、数值一致性、corrected 光谱寻峰、零峰、非法突出度、磁盘无副作用和真实样本回归。
- `tests/test_export.py`：8 个 CSV 字段、UTF-8-SIG、空表头和下载文件名。
- `tests/test_app_helpers.py`：上传内容、文件名或突出度变化时请求键失效。
- `tests/test_i18n.py`：两种语言、翻译键完整性、非法语言回退、错误翻译、MathText 坐标轴、状态保持、科学结果与 CSV 的语言无关性。

2026-10-20 在 Python 3.13.3 环境中执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

实际结果：运行 40 项测试，用时 0.153 秒；全部通过，无跳过、无失败、无错误（`OK`）。

同时执行：

```powershell
.\.venv\Scripts\python.exe -m pip check
```

实际结果：`No broken requirements found.`

## 7. 遇到的问题与解决办法

### 7.1 Matplotlib 中文字体

问题：默认字体环境下，中文标题和标签可能出现缺字，负号也可能显示异常。

处理：界面按 `Microsoft YaHei`、`SimHei`、`DejaVu Sans` 配置字体回退，并设置 `axes.unicode_minus=False`。前两种字体优先用于 Windows 中文显示，`DejaVu Sans` 作为回退；具体显示效果仍取决于运行系统实际安装的字体。

### 7.2 不同强度光谱的固定突出度阈值

问题：历史 Annite 样本在 `prominence=1000` 下得到 8 个候选峰，但校正后最大强度约 108 a.u. 的低强度光谱可能得到零候选峰。

处理：保留 1000 作为默认值以维持历史回归，同时允许用户输入非负有限的 `prominence`。寻峰仍只在未归一化的 corrected 光谱上进行，页面明确警告较低阈值可能引入噪声峰。测试中的低强度合成光谱在默认阈值下为零峰，降低到 10 后可以检测到候选峰。

### 7.3 中英文界面与状态保持

问题：语言切换必须覆盖图表与错误信息，同时不能触发不必要的光谱计算或丢失当前上传状态。

处理：新增 `raman_app/i18n.py` 集中管理翻译；`app.py` 使用独立语言状态键和 `st.dialog()`。语言回调只更新语言并执行应用级 rerun，分析请求键不包含语言，因此当前上传文件、`prominence` 和 `AnalysisResult` 可以继续使用。自动化测试验证状态保持、科学结果一致和 CSV 不受语言影响。

人工验收结果：默认简体中文、右上方设置弹窗、中英文即时切换、主要界面及图表翻译、现有分析结果保持和 CSV 下载均已通过人工验证。

## 8. 当前版本局限性

- 仅支持单条、UTF-8 编码、逗号分隔的两列 `.txt` 光谱。
- 当前界面语言仅包含简体中文和英语，语言选择只在当前 Streamlit 会话中保持。
- 参数由历史设置和用户输入决定，不提供自动寻优或质量评分。
- 候选峰没有经过峰形拟合、峰重叠分解、矿物归属或参考谱库确认。
- 半突出度宽度不能无条件视为物理 FWHM。
- 没有批量上传、数据库、用户管理、云端持久化或大型数据集处理。
- 没有矿物分类模型，也没有独立测试集上的识别准确率结论。
- 自动化测试覆盖翻译和状态逻辑；实际浏览器中的双语设置弹窗与语言切换已完成人工验收。

## 9. 后续可研究方向

- 建立经过版本控制和质量审核的参考光谱库，研究相似度匹配。
- 评估不同仪器、采样间隔和信噪比下的阈值策略，但避免把自动建议表述为通用最优参数。
- 增加峰形拟合、重叠峰分解和不确定度估计。
- 在明确数据划分、标签质量和独立测试集后研究矿物分类。
- 设计批量处理、任务进度和可追溯导出，再评估约 11,415 条外部光谱。
- 增加 Streamlit 端到端浏览器测试和多操作系统字体验收。
- 在保持翻译键一致性的前提下扩展第三种语言。

## 10. 资料边界

检查时 `docs/EXPERIMENT_06_SUMMARY.md` 为空。本总结没有据此编造实验六的数据或结论；实验六的详细记录需要由项目负责人另行补充或确认。
