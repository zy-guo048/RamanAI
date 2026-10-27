# RamanAI 实验 11.1：固定检索评价协议

## 1. 协议状态

本协议由实验 9.3 成功缓存和实验 10 的参考库过滤条件确定。固定随机种子为
`42`。本阶段未计算任何 Top-1、Top-3 或 Top-5 命中率，
也未运行 final test 查询。

参考标签仅是数据库中的待核实标签，不是独立验证的矿物真值；不同 Sample_ID 也不自动
等价于已经核实的独立物理标本。

## 2. 数据资格

记录必须同时满足：具有可用且无冲突的已知 Sample_ID、具有可用参考标签、无缺失或冲突
元数据标记、无需人工复核。同一 Sample_ID 的全部记录作为不可拆分单元；若同一 Sample_ID
关联多个参考标签，或组内任一记录不合格，则整个 Sample_ID 组不进入正式闭集评价。

严格闭集标签还必须具有至少
`5` 个合格 Sample_ID。
不合格及稀疏标签记录继续保留为探索性检索材料，但不进入正式指标分母。

实际参考谱 5529 条，参考标签
1853 个，已知 Sample_ID 3070 个。
严格闭集纳入 60 个标签、462 个
Sample_ID、971 条记录。

## 3. 固定分组

对每个严格闭集标签，将排序后的 Sample_ID 用种子 42 的确定性随机过程打乱：1 个进入
`development_query`，1 个进入 `final_test_query`，其余全部进入 `reference_gallery`。
同一 Sample_ID 的全部测量记录始终属于同一分组。

- reference_gallery：342 个 Sample_ID，
  724 条记录。
- development_query：60 个 Sample_ID，
  118 条记录。
- final_test_query：60 个 Sample_ID，
  129 条记录。

development 仅用于算法选择、阈值和检索规则确认。final test 在协议确认后冻结，不得根据
final-test 检索成绩重新选择样本、调整规则或优化参数。

## 4. 身份隔离与泄漏

gallery、development 和 final test 不共享已知 Sample_ID。还必须保证 relative_path 和
measurement_stem 不跨分组。实验 9.3 缓存仅包含准入 RAW；Processed 衍生文件不作为独立
参考样本。Unknown、缺失或冲突身份不得进入严格闭集分组。

光谱相似只允许作为探索性重复谱诊断，不得据此自动宣布两个文件来自同一物理样本。

## 5. 后续指标定义

后续可分别报告文件级、Sample_ID 级和参考标签级 Top-1、Top-3、Top-5 命中指标：

- 文件级：保留每条 gallery 文件作为证据，检查候选参考标签是否与查询参考标签一致。
- Sample_ID 级：每个已知 gallery Sample_ID 仅保留最高排名证据后评价。
- 参考标签级：每个标签使用其全部有效 gallery 参考谱中的最高余弦相似度。

闭集分母只包含所属标签在 gallery 中存在、且通过本协议资格检查的 query Sample_ID。
若查询标签不在 gallery，应单独报告为 out-of-gallery/open-set 情况，不计入闭集命中率分母。

同一 query Sample_ID 若有多条测量，应先在该 Sample_ID 内对各测量的命中指示求平均，
然后每个 Sample_ID 以相同权重进入总体指标，避免多测量 Sample_ID 重复计权。除总体微平均外，
必须对标签计算宏平均，使每个标签权重相同，并同时报告每个标签的 Sample_ID 数和查询数。

## 6. 标签库规模偏倚

参考标签级分数采用标签内最大相似度。参考文件越多，取得偶然高最大值的机会可能越大。
后续必须联合报告每个标签的文件数和 Sample_ID 数，并分析命中率或分数与库规模的关系。
任何按标签限额、重采样或替代聚合分数的方案只能在 development 上确定，并在 final test 前冻结。

## 7. 科学表述

所有指标只能称为“相对于当前数据库候选参考标签的检索命中率”。不得称为矿物识别概率、
置信度、独立验证准确率或真实材料分类准确率。
