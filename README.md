# RamanAI

RamanAI 是一个基于 Python 的拉曼光谱科研软件项目。项目以稳定的 V1
光谱分析流程为基础，正在扩展 RRUFF
相关光谱数据的审计、批量预处理、相似度检索和严格评价流程。

> **项目状态说明：** 本 README 根据截至实验 11.2 的项目交接信息、Codex
> 运行报告和已提供的结果文件整理。当前本地原始 `README.md`
> 未在本轮提供，因此这是一份可用于审阅和合并的整合稿，而不是对现有
> README 的逐行修改。具体 Git
> 状态、依赖版本和本地文件路径应以项目当前工作区为准。

## 项目目标

1.  为单条拉曼光谱提供可复现的预处理、寻峰和峰特征分析。
2.  对 RRUFF 相关数据集进行可追溯的元数据审计与质量筛选。
3.  建立统一波数网格上的实验性批量光谱缓存。
4.  提供文件、Sample_ID 和参考标签三个层级的 Top-K 相似光谱检索。
5.  使用按 Sample_ID
    隔离的开发集与最终测试集协议，评价检索方法，降低重复测量造成的数据泄漏。
6.  在证据充分后再讨论辅助矿物识别能力，不把相似度分数直接解释成概率或已验证的矿物身份。

## 已实现功能

### V1.0.0：稳定单谱分析

-   Streamlit 应用。
-   中英文双语界面。
-   Savitzky--Golay（SG）去噪。
-   AsLS 基线校正。
-   自动寻峰。
-   峰特征提取。
-   CSV 导出。

已知发布记录：

-   Git 标签：`v1.0.0`。
-   稳定提交：`5ceb9cf`。
-   当时自动化测试：40 项通过。
-   Annite 样例：1,194 个数据点、8 个候选峰。

V1 算法是稳定基线。实验性公共网格插值、批处理和相似度检索不应被误认为 V1
原有功能。

### 实验 8：RRUFF 数据审计和候选筛选

脚本：

-   `scripts/08_audit_rruff_dataset.py`
-   `scripts/08_build_rruff_metadata_index.py`
-   `scripts/08_select_rruff_candidates.py`

测试：

-   `tests/test_rruff_audit.py`
-   `tests/test_rruff_metadata_index.py`
-   `tests/test_rruff_candidates.py`

审计范围为 11,415 个 `.txt` 文件：

  指标                 结果
  ---------------- --------
  RAW                 5,628
  Processed           5,787
  格式解析成功       11,414
  格式解析失败            1
  V1 上传兼容        11,395
  V1 上传不兼容          20
  RAW 预处理候选      5,611
  初始参考候选        5,531

元数据索引统计：5,805 个 `measurement_stem`；Both 5,610、RAW-only
18、Processed-only 177、Unclassified
0。元数据缺失、冲突和待核实记录可能重叠。后续修补后的待核实记录为 RAW
97、Processed 132，共 229。

**重要：** `reference_candidate`
是经过当前准入规则筛选的候选参考标签，不代表已独立验证的矿物真值。

### 实验 9：覆盖率、采样、插值与批量预处理

主要脚本：

-   `scripts/09_profile_raw_coverage.py`
-   `scripts/09_inspect_raw_sampling.py`
-   `scripts/09_validate_interpolation.py`
-   `scripts/09_validate_dense_peak_fidelity.py`
-   `scripts/09_batch_preprocess_rruff.py`

对应测试位于 `tests/`，包括
`test_raw_coverage.py`、`test_raw_sampling.py`、`test_interpolation_validation.py`、`test_dense_peak_fidelity.py`
和 `test_batch_preprocessing.py`。

覆盖率统计基于波数端点，不证明区间内部不存在缺口：

  波段                  覆盖候选数       比例
  ---------------- --------------- ----------
  300--850 cm⁻¹      5,609 / 5,611   99.9644%
  300--1150 cm⁻¹     5,587 / 5,611   99.5723%
  232--1466 cm⁻¹       370 / 5,611    6.5942%

插值实验比较了 0.5、1.0 和 2.0 cm⁻¹ 网格。已测试的真实谱往返误差和合成
Gaussian
连续峰形保真结果整体支持：较细网格在这些指标上损失较小，尤其对窄峰更有利。但这些结果**尚不足以证明
0.5 cm⁻¹ 是所有仪器、矿物与下游任务的最终科学最优网格**。

批量预处理顺序：

``` text
V1 内存解析
→ SG(21, 3)
→ AsLS(lam=1e6, p=0.01)
→ 在原始采样轴上 max 归一化
→ 线性插值至 300–850 cm⁻¹、步长 0.5 cm⁻¹
```

5,611 条 RAW 预处理候选的实际结果：

-   成功：5,609。
-   失败：2，原因均为无法完整覆盖目标波段；未外推、补零或静默截断。
-   缓存矩阵：`(5609, 1101)`。
-   缓存值全部有限。
-   强度范围约为 `-0.115895` 至 `1.0`。

缓存文件：

-   `results/exp09_batch_preprocessing_full/rruff_preprocessed_300_850_step0p5.npz`
-   `results/exp09_batch_preprocessing_full/rruff_preprocessing_index.csv`
-   `results/exp09_batch_preprocessing_full/rruff_preprocessing_summary.json`

这里的公共网格插值属于实验 9 的处理步骤，并不等于 V1 上传 Pipeline
的原样复制。AsLS 后出现负值并不自动表示错误，因为 max 归一化不是 min-max
归一化。

### 实验 10：Top-K 相似光谱检索

脚本：

-   `scripts/10_search_rruff_topk.py`
-   `tests/test_rruff_topk.py`

当前检索参考库从成功缓存中进一步筛选：

-   参考光谱：5,529。
-   Sample_ID：3,070。
-   参考标签：1,853。
-   公共网格点数：1,101。

支持三个层级：

1.  **文件级**：以单条参考光谱排序。
2.  **Sample_ID 级**：对参考身份去重。
3.  **参考标签级**：每个标签的分数为排除规则执行后，该标签全部有效参考谱的最高余弦相似度。

支持自身排除、排除相同已知
Sample_ID、稳定并列排序、未知身份降级去重和外部 RAW
查询预处理。标签规范化不自动合并别名、连字符差异或可能不同的材料名称。

人工诊断命令：

``` powershell
.\.venv\Scripts\python.exe scripts\10_search_rruff_topk.py `
  --diagnostic `
  --k 10 `
  --output-dir "results\exp10_topk_manual_new"
```

输出目录应是尚不存在的新目录。

**解释限制：**
余弦相似度表示指定预处理和波段下的光谱形状相似度，不是概率、置信度或矿物识别准确率。检索标签仍属于数据库候选参考标签。

### 实验 11：评价协议和 Development 基线

#### 11.1：冻结分组协议

脚本：

-   `scripts/11_build_evaluation_protocol.py`
-   `tests/test_evaluation_protocol.py`

冻结文件目录：`docs/evaluation/exp11_v1/`

严格闭集包含 60 个标签，每个标签至少有 5 个合格
Sample_ID。固定分组如下：

  分组                  Sample_ID   光谱记录
  ------------------- ----------- ----------
  Gallery                     342        724
  Development query            60        118
  Final-test query             60        129

分组按 Sample_ID 隔离，且检查 `relative_path`、`measurement_stem`
和身份标记的跨组泄漏。Final-test
在方法、阈值和分析方案冻结前必须保持封存。

#### 11.2：Development Top-K 基线

脚本：

-   `scripts/11_evaluate_development_topk.py`
-   `tests/test_development_topk.py`

当前基线：

  检索层级          Top-1    Top-3    Top-5
  -------------- -------- -------- --------
  文件级           88.06%   93.33%   93.33%
  Sample_ID 级     88.06%   93.33%   93.33%
  参考标签级       88.06%   95.00%   96.67%

以上为 **60 个查询 Sample_ID 等权**的命中率。对应的 118
条逐光谱结果中，参考标签级 Top-1 / Top-3 / Top-5 分别为 89.83% / 96.61%
/ 98.31%。

评价约定：同一 Sample_ID 的多条查询先求命中指示平均，再让 Sample_ID
等权。由于每个标签恰有一个 development Sample_ID，Sample_ID
等权平均与标签宏平均相同。逐光谱微平均是另一个指标，不能混称。

当前结果文件：

-   `results/exp11_development_baseline/development_query_record_results.csv`
-   `results/exp11_development_baseline/development_sample_id_results.csv`
-   `results/exp11_development_baseline/development_label_stats_and_errors.csv`
-   `results/exp11_development_baseline/development_baseline_summary.json`

本轮已对提供的脚本和结果文件进行内部一致性复核。完整源数据、锁文件和缓存的端到端重新运行仍应由本地环境单独复现。

## 数据与科学解释原则

-   RAW 与 Processed 不是两个独立标本；Processed
    可能是同一测量的派生版本。
-   Sample_ID
    是当前数据协议采用的身份分组单位，但不自动等于已验证的独立物理标本。
-   参考标签未经过独立验证时，只能称为候选参考标签。
-   端点覆盖不代表波段内部没有缺口。
-   往返插值误差不是独立实测真值。
-   合成 Gaussian 的数学峰形不等于真实仪器响应。
-   余弦相似度不能直接解释为矿物识别概率。
-   对长尾标签分布、样本量不足或相关性分析，应报告实际分母和局限，不作超出证据的因果推断。
-   不要根据 final-test 成绩选模型、调阈值、重选样本或修改协议。

## 环境与运行

建议使用项目虚拟环境。

### Windows PowerShell

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

也可以不激活环境，直接运行虚拟环境解释器。

### 启动 Streamlit 应用

``` powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### 运行全项目测试

``` powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

最近一次报告的全量回归为 167 项通过；应以当前本地运行结果为准。

### 重跑 Development 基线（需要时）

``` powershell
.\.venv\Scripts\python.exe scripts\11_evaluate_development_topk.py `
  --output-dir "results\exp11_development_baseline_manual_new"
```

请使用新的输出目录，避免覆盖现有结果。不要为了确认结果而提前运行
final-test。

## Git 与实验产物

-   V1 标签：`v1.0.0`。
-   V1 稳定提交：`5ceb9cf`。
-   实验 8 提交记录：`f267762`。
-   之后的 HEAD 与工作区状态可能已变化；运行以下命令确认：

``` powershell
git status --short
git log -1 --oneline
git tag --list
```

不要把 `results/`
中的本地输出自动视为已纳入版本控制。新增脚本、测试、冻结协议和重要文档应按项目既定
Git 规则管理；提交、打标签和推送前先检查差异。

## V2.0.0 Release Candidate

当前工作区已具备 V2 候选功能：

- 保留 V1 中英双语单谱预处理、候选峰分析和峰特征 CSV；
- 在同一 Streamlit 应用中提供“光谱匹配 / Spectrum Matching”；
- 支持 Top-3、Top-5、Top-10 候选参考标签；
- 展示最佳参考文件、Sample_ID、激发波长、支持身份和查询/参考叠加图；
- 导出与当前页面结果一致的 UTF-8-SIG CSV 和可追溯 JSON；
- JSON 保存上传文件 SHA-256、预处理设置和实际 NPZ/索引来源指纹；
- 参考文件变化、缺失或身份映射异常时使旧结果失效，拒绝继续展示或导出。

设置按钮位于页面右上区域，可在简体中文和 English 之间切换。语言切换不改变光谱计算结果。

### 匹配数据的本地放置

匹配功能默认读取：

```text
results/exp09_batch_preprocessing_full/rruff_preprocessed_300_850_step0p5.npz
results/exp09_batch_preprocessing_full/rruff_preprocessing_index.csv
```

这两个文件及 RRUFF 原始数据默认不属于公开发布包。再分发许可尚未确认时，请勿将其复制到
公开制品。缺失时应从已验收的本地实验产物恢复，并使用来源 SHA-256 和
`scripts/14_release_check.py` 核验；不要用同名未知文件代替。

### 发布状态

V2 当前只能称为 **V2.0.0 Release Candidate**。冻结 final-test 已按一次性授权成功完成：
60 个查询 Sample_ID 等权的参考标签 Top-1/3/5 分别为 `0.9361111111`、`0.9833333333`、
`0.9916666667`；95% percentile Bootstrap 区间分别为 `[0.8777777778, 0.9833333333]`、
`[0.9500000000, 1.0000000000]`、`[0.9750000000, 1.0000000000]`。这些数值仅表示相对于候选
参考数据库标签的冻结闭集检索命中率。相关说明见：

- `docs/EXPERIMENT_13_SUMMARY.md`
- `docs/V2_RELEASE.md`
- `docs/SCIENTIFIC_LIMITATIONS.md`

只读发布检查：

```powershell
.\.venv\Scripts\python.exe scripts\14_release_check.py `
  --final-output-dir "results\exp13_final_test_e99dc611-5687-4a82-aff7-2cd459dd598d"
```

该命令不会重新运行 final-test。只有全量测试、依赖检查、只读发布检查和用户浏览器验收
全部完成后，才可决定是否正式发布 V2.0.0。

## 当前进度与下一步

截至实验 13 授权准备阶段：

-   实验 8：RRUFF 审计、索引和候选筛选完成。
-   实验 9：覆盖率、采样与插值研究，以及批量预处理完成。
-   实验 10：三层 Top-K 检索原型完成。
-   实验 11.1：冻结评价协议完成。
-   实验 11.2：Development 基线完成，核心输出内部一致性已复核。
-   实验 11.3--11.5：Development 诊断、证据解释、匹配 UI 与来源完整性保护已完成。
-   实验 12：正式方法、Bootstrap、输入与可执行候选锁已准备。
-   实验 13：唯一一次冻结 final-test 已成功完成，禁止再次运行。
-   实验 14：V2 发布文档和只读发布检查已完成，仍需用户浏览器验收。
