# RamanAI 实验 12-A：Final-test 执行政策草案

> 状态：`draft_pending_human_review`  
> 本文件不授权 final-test 运行。  
> 首次正式运行授权状态：`not_requested`

## 1. 适用范围

本政策约束实验 12-B 的实现和实验 13 的首次正式 final-test 运行。正式方法、数据范围、
指标和 bootstrap 参数由 `RETRIEVAL_METHOD_LOCK.md` 与
`RETRIEVAL_METHOD_LOCK.json` 定义。

实验 12 的所有测试和诊断只能使用合成数据或冻结 Development 分组。实验 12 中 final-test
相似度计算数、预测数和成绩输出数必须全部为 0。

## 2. 两阶段冻结状态

### 阶段一：方法配置决定

12-A 确定 `A_max_spectrum_cosine`、数据分组、指标和 bootstrap 设计，但文件仍是待审核草案。
审核不得被文件存在、测试通过或方法字段已填写自动替代。

### 阶段二：可执行源码冻结

12-B 完成后必须生成独立的可执行源码锁，至少记录：

- 每个正式运行源文件的相对路径、大小和 SHA-256；
- 机器可核验的方法锁 SHA-256；
- 实验 11 数据锁、Manifest、NPZ、索引和 final-test 身份哈希；
- Python、NumPy 及相关依赖版本；
- 正式 CLI 参数和输出 schema 版本；
- 专项测试、全项目回归和静态检查结果；
- `final_test_similarity_count=0` 的 12-B 预运行证据；
- 锁状态 `draft`、`approved_and_frozen` 或 `revoked`。

方法锁和可执行锁以“冻结候选、等待外部人工批准”状态确定最终字节和 SHA-256。实验 13
只有在另一个一次性批准记录精确绑定这两个最终哈希后，才可选择 final-test NPZ 行。

## 3. 12-B 必须提供的程序接口

具体函数名可依据代码结构调整，但必须具备下列可独立测试的职责：

1. `load_and_validate_method_lock`：校验 JSON schema、草案/批准状态和禁止变更字段。
2. `verify_locked_inputs`：逐文件核对路径、大小和 SHA-256。
3. `verify_manifest_identities`：核对分组数量、all-group 哈希和 final-test 身份哈希。
4. `validate_group_isolation`：检查 Sample_ID、`relative_path` 和 `measurement_stem` 不跨组。
5. `map_manifest_rows_to_cache`：以 `cache_row_index` 和身份字段核对 NPZ/索引映射。
6. `rank_reference_labels_method_a`：从全部冻结 Gallery 文件生成 60 标签完整确定性排名。
7. `aggregate_query_sample_ids`：先组内平均，再对查询 Sample_ID 等权。
8. `compute_percentile_bootstrap`：固定 60 单位、10,000 次、种子 42，并显式固定百分位算法。
9. `build_run_audit`：记录所有检查、输入哈希、代码哈希、计数、状态与失败阶段。
10. `write_outputs_no_overwrite`：以新目录事务式写出，任何已有目标都拒绝覆盖。
11. `require_final_test_approval`：在选择 final-test 向量前验证明确的人类批准材料。

实验 13 入口应当是调用这些冻结接口的薄层，不得另写一套余弦相似度、标签聚合或指标逻辑。

## 4. 12-B 专项测试清单

所有测试使用合成数据和 Development，不运行 final-test 预测：

- 方法 A 与实验 11.2 Development reference-label Top-1/3/5 完全一致；
- 方法 B、未知方法 ID 和参数搜索配置被拒绝；
- 标签分数从全部 Gallery 文件计算，而非文件 Top-K 截断结果；
- 标签规范化和并列排序与实验 10 一致；
- 724/118/129、342/60/60 和 60 标签计数校验；
- Manifest、数据锁、NPZ、索引或协议哈希不一致时，在向量选择前停止；
- final-test 身份哈希不一致时停止；
- Gallery 与 Development/final-test 的 Sample_ID、路径或 measurement_stem 泄漏时停止；
- NPZ、索引、Manifest 的 `cache_row_index` 或 `relative_path` 映射不一致时停止；
- 非有限向量、零范数及严重余弦越界被拒绝；
- 同一查询 Sample_ID 多测量先组内平均；
- 标签宏平均与 Sample_ID 等权一致性检查；
- bootstrap 固定种子可复现，且不拆分 Sample_ID 内测量；
- bootstrap 百分位算法由固定夹具验证；
- 缺少批准材料时 final-test 行选择函数不被调用；
- 输出目录或任一目标文件存在时拒绝覆盖；
- 模拟中途失败时保留审计信息和既有输出，不删除、不覆盖；
- 同一输入重复预运行得到相同的 Development 指标和 bootstrap 结果；
- 全项目既有测试不退化。

12-B 报告必须明确给出 final-test 查询向量选择次数、相似度计算次数、预测数和成绩数，四者
均应为 0。

## 5. 首次正式运行的批准门禁

实验 13 的首次正式运行必须由用户在运行前明确批准。12-B 应设计可机器核验的批准文件，
至少包含：

- `approval_status: approved_for_single_final_test_attempt`；
- 审批人或审批标识；
- 审批时间；
- 已批准方法锁 SHA-256；
- 已批准可执行源码锁 SHA-256；
- final-test 身份 SHA-256；
- NPZ 与索引 SHA-256；
- 指定的新输出目录；
- 唯一 `run_id`；
- Manifest 与实验 11 数据源锁 SHA-256。

`run_id` 必须是规范的小写 UUID，并同时绑定到批准记录、尝试审计和正式输出。批准记录对应的
`run_id` 一旦开始执行，无论成功、授权拒绝或技术失败，均不得再次使用；技术重跑必须获得新的
人工批准、UUID 和输出目录。

CLI 还应要求显式的 final-test 确认参数。批准文件和显式参数必须同时存在；任一缺失、过期、
哈希不匹配或状态非批准时，都必须在读取 final-test NPZ 行前退出。不能通过默认值、环境猜测
或交互式“按任意键继续”绕过门禁。

## 6. 正式运行前检查顺序

应按以下顺序执行，顺序本身属于安全边界：

1. 检查输出目录及运行 ID 未使用；
2. 读取方法锁、可执行源码锁和批准文件；
3. 核对方法锁与源码锁均为不可变冻结候选，并核对相互哈希引用；
4. 在选择 final-test 向量前核对 Python、NumPy、SciPy 版本与可执行锁一致；
5. 计算并核对协议、Manifest、NPZ、索引和源码 SHA-256；
6. 只读取 Manifest 元数据，复核分组计数、身份哈希和泄漏检查；
7. 加载 NPZ 时使用 `allow_pickle=False`，核对网格、矩阵和索引身份映射；
8. 构建且验证 724 条 Gallery；
9. 最后才允许按冻结 `cache_row_index` 选择 129 条 final-test 查询；
10. 计算正式结果并写入新的输出目录；
11. 写出完整运行审计、输出文件哈希和完成状态。

应用级 5,529 条参考库不得进入步骤 7。

## 7. 输出禁止覆盖

- 正式输出目录必须在运行开始前不存在。
- 不允许删除、清空、截断或复用历史正式结果目录。
- 运行使用唯一 `run_id`，并在所有输出和审计文件中保持一致。
- 每次尝试在正式输出目录之外的 `final_test_attempt_audits/<run_id>.json` 留下持久记录；该记录
  按生命周期原子更新，不能因正式输出写入失败而丢失。
- 正式结果产生后，任何再运行都必须使用新目录并保留前次审计；不得把后一次结果伪装成首次运行。
- 输出文件至少记录输入哈希、源码锁哈希、批准文件哈希、统计分母、失败数及生成时间。

## 8. 技术失败与恢复

技术失败包括但不限于文件读取失败、哈希不匹配、内存错误、非有限向量、映射错误和写出失败。

发生失败时：

1. 立即停止，不生成或补写正式成绩；
2. 写入或保留 `RUN_AUDIT.json`，状态标为 `failed`；
3. 记录失败阶段、异常类型、已完成检查和是否已经计算任何查询；
4. 不删除已生成的临时或正式输出；
5. 修复只能针对明确的技术故障，不能改变方法、数据、Gallery、标签或指标；
6. 若故障发生在任何成绩可见之后，后续运行必须声明为重跑并保留前次运行材料；
7. 是否允许重跑需再次获得用户明确批准。

不得捕获所有异常后静默继续，也不得跳过失败查询缩小正式分母。

审计计数必须区分 `final_test_query_row_count`、`gallery_reference_row_count`、
`similarity_pair_count` 和 `prediction_count`。冻结设计的预期值分别为 129、724、93,396 和
129，但运行时必须由实际选中矩阵与结果结构计算并核对，不得用查询行数代替相似度配对数。

## 9. 锁更新顺序

锁定不采用递归自哈希。人工批准前按以下顺序执行：

1. 完成并测试评价核心及正式入口；
2. 将方法锁固定为 `frozen_candidate_pending_external_human_approval`，确定最终方法锁 SHA-256；
3. 将该方法锁哈希和全部可执行源码哈希写入状态为
   `frozen_candidate_pending_human_approval` 的可执行锁；
4. 可执行锁内容确定后计算最终 SHA-256，之后不得再改写任一锁；
5. 最后由人工授权创建一次性批准记录，同时绑定两个最终锁哈希、输入哈希、final-test 身份哈希、
   规范输出目录和唯一 `run_id`。

外部批准记录是对两个冻结候选哈希的批准，不通过修改锁内状态表达，因此不会在批准后改变用户
已审核的哈希。任何源码或锁内容变化都会使批准失效，禁止程序自动静默更新锁或批准文件。

## 10. 结果产生后的禁止事项

首次正式成绩产生后，不得根据结果：

- 调整 SG、AsLS、归一化、插值网格或余弦计算；
- 选择方法 B 或新的聚合规则；
- 改变 Top-K、主要指标或 bootstrap 参数；
- 修改 Gallery、重新分组或更换随机种子；
- 合并标签同义词或修正不利标签；
- 删除错误、不利样本或失败查询；
- 用应用级 5,529 条参考库重算正式闭集指标；
- 将相似度或命中率改称概率、置信度或外部验证准确率。

发现科学或数据问题时，应保留原结果并发布勘误或新协议版本；不得覆盖原始正式运行。

## 11. 当前待审核事项

- 接受方法 A 作为唯一正式方法：`pending`
- 接受 bootstrap 设计及其限制：`pending`
- 接受单一 NPZ 下的逻辑隔离：`pending`
- 批准 12-B 门禁和测试接口：`pending`
- 批准首次 final-test 正式运行：`not_requested`

在这些状态按规定更新前，本政策始终是草案。
