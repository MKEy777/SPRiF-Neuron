# SPRiF 正文与补充材料内容规划

## 分配原则

AAAI 正文必须自包含。审稿人即使不阅读 technical appendix，也应能够理解 SPRiF 的方法、复现实验的核心设置，并判断两个主要机制主张是否得到支持。补充材料承接完整细节、扩展结果和审计记录，不能承接论文成立所必需的唯一证据。

## 内容分配

| 内容 | 正文 | 补充材料 | 说明 |
| --- | --- | --- | --- |
| 问题定义与核心动机 | 必须 | 不重复 | 正文完整建立 memory/readout/reset 耦合问题 |
| SPRiF slow/fast 状态结构 | 必须 | 完整推导 | 正文保留全部核心更新公式 |
| Projective reset `[1,λ]` | 必须 | reset identity 审计 | 正文不能只引用补充材料 |
| 参数约束与稳定性 | 核心定义 | 初始化范围和推导 | 正文保留 α、ρ、ω 的取值域 |
| 五个 benchmark 协议 | 精简但充分 | 完整预处理和超参数 | 正文报告公平比较所需设置 |
| 五任务主结果 | 必须 | per-seed 和扩展指标 | 核心结论必须在正文可见 |
| full vs merged | 必须 | 完整多种子表 | 对应 slow/fast separation 主张 |
| full vs λ=0 | 必须 | 完整多种子表 | 对应 projective reset 主张 |
| ω=0 等次要消融 | 可精简 | 推荐完整放置 | 不属于 SI-DMS 两项核心对照 |
| SI-DMS 任务和 intervention 定义 | 必须 | 生成算法和完整配置 | 正文需说明干预发生在 threshold 前 |
| SI-DMS 核心 delay×K 结果 | 必须 | 完整网格与 per-seed 数据 | 正文至少展示三项机制模型的主要趋势 |
| LIF、ASRNN、BRF SI-DMS 基线 | 视篇幅精简 | 完整结果 | 必须标记为 external baselines |
| forced-hit rate | 正文报告通过 | 完整逐条件审计 | `K>0` 必须为 1 才能解释 reset |
| reset residual / slow insulation | 核心结论或紧凑图 | 完整数值和测试 | 替代旧 trajectory reset 箭头 |
| impulse response | 核心代表图 | 完整 gallery | 支撑 temporal-kernel 解释 |
| frequency selectivity | 关键结论 | 完整结果 | 保留原实验，不被 SI-DMS 替代 |
| λ 分布与 reset analysis | 代表结果 | 完整分布 | 避免赋予 λ 符号未经证明的生物含义 |
| noise robustness / sequence noise | 关键结果或简述 | 完整表图 | 根据正文页数决定展示密度 |
| failure cases | 至少一句 limitation | 完整案例与选择规则 | 不得只展示成功案例 |
| 复现信息 | 必要摘要 | 完整 checklist | 不写本机绝对路径或身份信息 |

## 正文建议结构

1. Abstract
2. Introduction
3. Related Work
4. Method
   - Functional decomposition
   - Slow spectral state
   - Fast discharge and projective reset
   - Training and cost
5. Experiments
   - Benchmarks and protocol
   - Main benchmark results
   - Core mechanism ablations
   - SI-DMS controlled intervention
   - Dynamical and robustness analysis
6. Limitations
7. Conclusion
8. References

## Technical Appendix 建议结构

1. Supplementary Overview
2. Complete SPRiF Dynamics
3. Benchmark and Preprocessing Details
4. Training and Reproducibility Details
5. Complete Benchmark Results
6. Complete Mechanism Ablations
7. SI-DMS Protocol
8. SI-DMS Integrity Audits
9. Additional Dynamical and Robustness Analyses
10. Failure Cases and Limitations
11. Reproducibility Checklist
12. Supplementary References

## 正文待更新位置

当前 `SPRiF_AAAI2027.tex` 在正式 SI-DMS 结果生成后需要进行以下修改：

- Abstract 中的 “State trajectories directly verify ...” 必须改为 SI-DMS 的受控干预证据，且只能使用正式结果支持的措辞。
- Experiments 中新增 SI-DMS setting 和核心结果小节。
- Dynamical Analysis 不再使用旧 `trajectory_visualization` 的 reset 箭头；`trajectory_analysis`、impulse 和 λ analysis 可继续保留。
- Conclusion 中的 “trajectory ... make these mechanisms directly inspectable” 应改为实际 spike/reset intervention 所支持的结论。
- 所有 supplementary 引用必须是辅助性的，不能把核心公式、核心比较或唯一证据移出正文。

## 提交前硬检查

- 主文和 technical appendix 均保持匿名。
- appendix PDF 不出现作者、单位、实验室、非匿名仓库或本机路径。
- 所有占位注释在最终提交前转化为真实内容或删除。
- 不保留 `TBD`、空表、空图或未定义引用。
- 重新核验 AAAI-27 对 technical appendix 的文件大小、页数和上传格式要求。
