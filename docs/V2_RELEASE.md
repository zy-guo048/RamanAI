# RamanAI V2.0.0 Release Candidate

> 状态：Release Candidate；首次冻结 final-test 已完成，尚待用户浏览器验收，因此未正式发布。

## 交付范围

V2 保留 V1 单谱分析，并增加应用级 RRUFF 候选参考库光谱匹配：

- RAW TXT 上传、SG 去噪、AsLS 基线校正、候选峰和峰表；
- 中英文界面及峰特征 CSV 下载；
- 300–850 cm⁻¹ 公共网格上的 Top-3/5/10 参考标签检索；
- 最佳参考光谱、Sample_ID、激发波长和支持证据展示；
- 查询谱与最佳参考谱叠加；
- 匹配 CSV、JSON 导出；
- 查询和参考库来源 SHA-256 指纹；
- 参考文件变化后的缓存失效与旧结果导出保护。

## 本地数据

公开发布包默认不包含 RRUFF 原始数据、批量 NPZ、索引 CSV或大型 `results/`。应用匹配功能
要求本地存在：

```text
results/exp09_batch_preprocessing_full/rruff_preprocessed_300_850_step0p5.npz
results/exp09_batch_preprocessing_full/rruff_preprocessing_index.csv
```

文件缺失、身份映射不一致或内容指纹变化时，应用必须明确报错。再分发许可未确认前，不应将
这些数据复制进公开发布包。

## 发布检查

冻结 final-test 首次结果：Sample_ID 等权 Top-1/3/5 分别为 `0.9361111111`、
`0.9833333333`、`0.9916666667`。其含义仅为相对于当前候选参考数据库标签的冻结闭集检索命中率。

授权 final-test 完成后运行：

```powershell
.\.venv\Scripts\python.exe scripts\14_release_check.py `
  --final-output-dir "results\exp13_final_test_e99dc611-5687-4a82-aff7-2cd459dd598d"
```

该命令只读核验，不会重新运行 final-test。
