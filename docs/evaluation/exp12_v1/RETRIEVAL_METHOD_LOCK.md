# RamanAI 实验 12-A：最终评价方法锁草案

> 文档状态：`draft_pending_human_review`  
> 方法配置状态：`decided_in_draft_not_yet_approved`  
> 可执行源码状态：`not_frozen_pending_experiment_12_b`  
> final-test 状态：`prohibited_in_experiment_12`  
> 草案日期：2026-10-27

本文件预注册 RamanAI 最终闭集检索评价的方法、数据边界、统计指标和完整性要求。
它不是已批准的最终锁，也不授权读取 final-test 光谱向量、计算相似度、生成预测或报告成绩。
机器可核验的对应配置位于同目录的 `RETRIEVAL_METHOD_LOCK.json`。

## 1. 评价问题与科学边界

正式评价的问题是：在实验 11.1 冻结的 724 条 Gallery 光谱上，使用固定检索方法时，
129 条 final-test 查询光谱所属的 60 个 Sample_ID 能否在 Top-1、Top-3 和 Top-5
候选参考标签中命中其冻结的 `reference_label_group_key`。

这里的 `reference_label` 是数据库候选参考标签，不是经独立矿物学验证的真值。
结果只能称为“相对于当前数据库候选参考标签的检索命中率”。余弦相似度不是概率、
置信度或识别准确率。本评价也不支持外推到全部矿物、全部仪器、外部实验室数据或开放集识别。

## 2. 冻结数据范围

复用实验 11.1 的冻结 Manifest 和实验 11.2 的数据源锁，不重新分组：

| 分组 | 光谱记录 | 已知 Sample_ID | 候选参考标签 |
|---|---:|---:|---:|
| `reference_gallery` | 724 | 342 | 60 |
| `development_query` | 118 | 60 | 60 |
| `final_test_query` | 129 | 60 | 60 |

正式评价只能以 724 条 `reference_gallery` 作为参考库。不得使用应用级 5,529 条参考库替代
冻结 Gallery，不得把 Development、final-test 或稀疏标签记录加入 Gallery。

实验 12 只能用冻结 Manifest 中的元数据和身份哈希核对 final-test 的数量、Sample_ID、
路径和标签分组键。不得读取 final-test 对应 NPZ 行用于相似度计算，不得生成预测标签、
排名、得分或任何检索成绩。

实验 9.3 NPZ 是包含多类分组记录的单一物理文件，因此当前隔离是基于
`cache_row_index` 和 Manifest 的逻辑隔离，而非物理文件级隔离。实验 12-B 的预运行检查必须
在选择任何查询向量前验证分组；没有执行批准时不得选择 final-test 向量。

## 3. 方法 A 的可执行级定义

正式方法标识为 `A_max_spectrum_cosine`。

### 3.1 冻结预处理谱来源

最终评价直接使用实验 9.3 已冻结 NPZ 中的预处理向量，不在实验 13 重新读取或预处理 RAW。
这些缓存向量的生成谱系固定为：

1. V1 兼容解析；
2. Savitzky–Golay：`window_length=21` 个采样点，`polyorder=3`；
3. AsLS：`lam=1e6`，`p=0.01`；
4. 在原始有效波数轴上进行 `max` 归一化；
5. 线性插值到 300–850 cm⁻¹、步长 0.5 cm⁻¹、共 1,101 点；
6. 禁止外推和补零。

SG 窗口按采样点计，不同原始采样间隔对应不同物理宽度。0.5 cm⁻¹ 网格是本评价的冻结
工程配置，不代表科研上已经证明的最终最优网格。

### 3.2 查询和 Gallery 映射

- 记录身份以 `relative_path` 为主键，并核对唯一性。
- NPZ 行必须由冻结 Manifest 的 `cache_row_index` 精确选择。
- 选择后必须逐行核对 `relative_path`、Sample_ID、`measurement_stem`、
  `reference_label_group_key` 及缓存身份映射。
- Gallery 矩阵必须为 724 行、1,101 列；正式 final-test 查询矩阵必须为 129 行、1,101 列。
- 所有向量必须有限且范数大于 0；异常时整个正式运行失败，不静默跳过。

### 3.3 余弦相似度

对查询向量 \(q\) 与每条 Gallery 光谱 \(g_i\) 计算：

\[
s_i = \frac{g_i \cdot q}{\lVert g_i\rVert_2\lVert q\rVert_2}.
\]

允许负相似度。非有限值、零范数或实质超出 `[-1, 1]` 的结果必须报错。仅允许实验 10
既有规则对不超过 `1e-12` 的浮点边界误差裁剪到 `[-1, 1]`。

一次查询只计算该查询与 724 条 Gallery 光谱的相似度，不构造数据库全体两两相似度矩阵。

### 3.4 参考标签规范化

沿用实验 10 的既有规则：

1. 对 `reference_label` 做 Unicode NFKC 规范化；
2. 去除首尾空白，并把内部连续空白折叠为单个空格；
3. 分组键为 `"reference_label:" + normalized_label.casefold()`；
4. 不增加矿物同义词、别名、化学式或人工映射合并；
5. 正式 60 标签均必须具有冻结的可用分组键，缺失标签不得临时修复。

### 3.5 标签分数与完整排名

对每条查询，先计算其与全部 724 条合格 Gallery 光谱的相似度。对每个标签 \(L\)：

\[
score(L,q)=\max_{i:\,label(g_i)=L} cosine(g_i,q).
\]

必须在全部合格 Gallery 文件上取最大值，不能先取文件级 Top-K 再汇总。不得采用方法 B，
不得对文件数、Sample_ID 数或激发波长加权，也不得进行新的聚合参数搜索。

每个标签同时保留产生最大值的支持文件作为可追溯证据。标签分数仅用于排序，不解释为概率。

### 3.6 确定性并列排序

文件候选先按以下键升序排列：

1. `-cosine_similarity`；
2. `relative_path.casefold()`；
3. 原始 `relative_path`。

同一标签的最佳文件在完全相同分数时使用上述路径顺序。标签完整排名按以下键升序排列：

1. `-label_score`；
2. `reference_label_group_display.casefold()`；
3. `reference_label_group_key`；
4. 最佳支持文件 `relative_path.casefold()`；
5. 最佳支持文件原始 `relative_path`。

不得使用随机方式打破并列。

## 4. 正式指标

### 4.1 命中定义

查询记录的正确标签由冻结 Manifest 中的 `reference_label_group_key` 给出。若正确标签出现在
完整标签排名前 K 名，则 `hit_at_K=1`，否则为 0。正式 K 为 1、3、5。

### 4.2 主要和次要指标

- 主要指标：reference-label Top-1，60 个 final-test 查询 Sample_ID 等权。
- 次要指标：reference-label Top-3 和 Top-5，60 个 final-test 查询 Sample_ID 等权。
- 补充描述：129 条查询光谱的逐记录 Top-K、60 标签宏平均、正确标签完整排名、混淆对和错误案例。

### 4.3 Sample_ID 等权聚合

对 Sample_ID \(j\) 的 \(n_j\) 条测量，先计算：

\[
h_{j,K}=\frac{1}{n_j}\sum_{r=1}^{n_j} hit_{j,r,K}.
\]

再计算 60 个查询 Sample_ID 的等权平均：

\[
H_K=\frac{1}{60}\sum_{j=1}^{60}h_{j,K}.
\]

同一 Sample_ID 的多条测量不得作为多个独立样本重复计权。当前每个严格闭集标签恰有一个
final-test Sample_ID，因此标签宏平均与 Sample_ID 等权结果理论上应一致；如果程序输出不一致，
必须停止并调查，不能把二者报告成两组独立证据。

## 5. Bootstrap 区间预注册

对 Top-1、Top-3、Top-5 的 Sample_ID 等权指标使用同一方案：

- 重抽样单位：查询 Sample_ID；
- 总单位数：60；
- 每次有放回抽取 60 个 Sample_ID；
- 重抽样次数：10,000；
- 随机种子：42；
- 每次使用预先计算的 `h_j,K`，不拆分 Sample_ID 内重复测量；
- 区间：双侧 95% percentile bootstrap；
- 下界取 10,000 个重抽样统计量的 2.5 百分位；
- 上界取 97.5 百分位；
- 分位数实现固定为
  `np.quantile(replicate_metrics, [0.025, 0.975], axis=0, method="linear")`；
- Top-1、Top-3、Top-5 共用同一组重抽样索引；
- 12-B 必须用固定合成夹具验证点估计、抽样索引哈希和区间数值。

该区间只是对这 60 个预选标签、每标签一个 final-test Sample_ID 的条件重抽样描述。
它不能提供对全部矿物、所有仪器、外部实验数据或标签真实性的置信保证。

## 6. 已确认的冻结数据哈希

下列值已于 2026-10-27 从工作区实际文件重新计算并与实验 11 锁核对：

| 对象 | SHA-256 |
|---|---|
| Manifest | `1467911decc715ed2ab6a4c5214e4ecb170db4b10d7c89a1c2d8dc438ac9aeee` |
| 实验 9.3 NPZ | `8ad10760593498ef8b256c44571165eab1ed73987e90db587f0b26088de65610` |
| 实验 9.3 索引 CSV | `6f3f64884fe653dc32143f7a495a0a457d0b3b4a15a6b5dab57f58ae26b61a87` |
| 实验 11 评价协议 | `9fb3d74fa25dc58532b105075ccb0feba6ca6c6baabfc15c3f89a5b1c4474b74` |
| 实验 11 分组审计 | `dd9537e220594599d1563163e96d069b9f95e4cedab2f412b17e181103a34551` |
| 标签/Sample_ID 分布表 | `3fa9973c0d196ea3c6be76406ccbbd37868fa2332f524f584b54682b250c414b` |
| 实验 11 数据源锁文件本身 | `6b1945fe7648ade97134c4d218a18ab5fa73d6d1b1d424e8666aa077df8ba6fe` |

冻结 final-test 身份哈希为：

`d94b630e4cf60ffb532583b52c78997e4276faa743752f1ba974bc30258fde2c`

其身份哈希负载由每条记录的
`sample_id|evaluation_group|relative_path|reference_label_group_key` 构成，按行排序后以 UTF-8
编码并计算 SHA-256。12-A 仅使用 Manifest 元数据复核该定义，不读取 final-test 光谱向量。

## 7. 源码冻结的两阶段状态

### 7.1 已决定的方法配置

本文件第 2–5 节的方法、分组、指标和 bootstrap 配置已经作为 12-A 草案确定，不允许在
12-B 中进行方法 B 选择或参数搜索。人类审核通过前，其批准状态仍为 `pending`。

### 7.2 尚未最终冻结的可执行源码

12-A 只记录现有源码清单和观察哈希。因为 12-B 尚需实现预运行检查、bootstrap、正式输出
schema 和执行门禁，这些哈希不是最终可执行锁，也不能填入未来的批准文件冒充最终值。

12-B 完成后至少要锁定：

- 实验 10 余弦相似度、标签规范化和稳定排序实现；
- 冻结 Manifest/NPZ/索引映射与泄漏检查实现；
- 方法 A 完整排名实现；
- Sample_ID 等权与标签宏平均实现；
- bootstrap 实现；
- 正式输出和审计轨迹实现；
- 实验 13 的薄运行入口；
- 上述文件的 SHA-256、大小、Python 与 NumPy 版本及测试结果。

## 8. 审批和执行边界

12-A 和 12-B 均不得运行 final-test 预测。实验 13 首次正式运行必须同时满足：

1. 方法锁经用户明确批准；
2. 12-B 可执行源码锁状态为 `approved_and_frozen`；
3. 数据、协议、Manifest、final-test 身份和源码哈希全部匹配；
4. 11.5-C 浏览器人工验收结论已记录；
5. 使用新的、不存在的输出目录；
6. 用户针对首次正式运行给出明确授权。

缺少任一条件必须在加载 final-test 查询向量和计算相似度前停止。详细执行和失败规则见
`FINAL_TEST_EXECUTION_POLICY.md`。

## 9. 审核结论栏

- 方法配置审核：`pending`
- Bootstrap 方案审核：`pending`
- 逻辑隔离限制接受：`pending`
- 12-B 接口和测试方案审核：`pending`
- final-test 执行授权：`not_requested`

本草案不得因文件存在而被解释为已经批准冻结。
