# RamanAI 实验 13：冻结 final-test 正式评价

> 当前状态：首次正式运行成功完成；禁止再次运行。  
> 结果名称：相对于候选参考数据库标签的冻结闭集检索命中率。

## 固定设计

- 方法：`A_max_spectrum_cosine`。
- Gallery：724 条光谱、342 个已知 Sample_ID。
- Final-test：129 条光谱、60 个 Sample_ID。
- 严格闭集：60 个候选参考标签。
- 主要指标：查询 Sample_ID 等权 reference-label Top-1。
- 次要指标：同一定义下的 Top-3、Top-5。
- Bootstrap：Sample_ID 为单位，60 个身份有放回抽样，10,000 次，种子 42，
  `numpy.random.default_rng(42)`，三个 K 共用抽样索引，双侧 95% linear percentile 区间。

## 运行保护

正式运行要求唯一 UUID `run_id`、全输入与源码哈希核验、Python/NumPy/SciPy 环境核验、
外部一次性人工批准以及从未存在的新输出目录。每次尝试会在正式输出目录之外保留持久审计；
失败后不得自动重跑。

## 首次正式结果

- run_id：`e99dc611-5687-4a82-aff7-2cd459dd598d`
- 查询光谱：129 条；查询 Sample_ID：60 个。
- Gallery：724 条；相似度配对：93,396；预测：129。
- Sample_ID 等权 Top-1：`0.9361111111`，95% percentile 区间
  `[0.8777777778, 0.9833333333]`。
- Sample_ID 等权 Top-3：`0.9833333333`，区间 `[0.9500000000, 1.0000000000]`。
- Sample_ID 等权 Top-5：`0.9916666667`，区间 `[0.9750000000, 1.0000000000]`。
- 逐记录 Top-1/3/5：`0.9379844961` / `0.9844961240` / `0.9922480620`。
- Top-1 未命中记录：8 条；涉及 Top-1 非全命中的查询 Sample_ID：6 个。

方法锁与可执行锁分别为：

- `2bccf0553debc38e6e4e425dce19f9a9bfc6f1d4c47104181b6dbb8dcb4f8c5a`
- `fddd0be66cfda963be9d8cc54fc7869fb1ac1a3a4c1870a3d50a68e00fe186b6`

关键审计哈希：

- `final_test_summary.json`：`3a1402970e5585e572dd19b0b75b3acbdb57a3ec841d9b96f1cb633a4d14e170`
- `RUN_AUDIT.json`：`bebf9db1ead50c5328006a8d6ba1f7f4b384af78e78060978e2d609646cb2b4d`
- 持久尝试审计：`a85c259d159c970c15a1d54954dbe021bfe5eb8573e759dafdcce5630a361675`
- 一次性批准记录：`d4d790899517148f2eba455bf6305466bf37325708b0bc461ac510be3d472ee6`

这些结果不是外部验证的矿物识别准确率、概率或置信度。Bootstrap 区间是对预选 60 个查询
Sample_ID 的条件重抽样描述，不能代表全部矿物、仪器或外部数据。
