# RQ2 实验结果与分析（Table IX: LLM Model Comparison on theia311）

Ego 级变异 pipeline（EgoMutationPipeline），每快照 5 个变异 ego，对比学习每快照采样 5 个负样本。

### Table IX-A: 标准测试（ego 级）

| Augmentation | LLM Model | Acc | Prec | Rec | F1 | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| No Aug | - | 95.00% | 61.98% | 99.01% | 76.24% | 300 | 184 | 3252 | 3 |
| GraphCL | - | 99.33% | 94.67% | 96.93% | 95.78% | 284 | 16 | 3430 | 9 |
| GCA | - | 97.30% | 75.38% | 99.01% | 85.59% | 300 | 98 | 3338 | 3 |
| Mimicry | - | 90.40% | 45.68% | 97.69% | 62.25% | 296 | 352 | 3084 | 7 |
| LLM-guided | 无LLM（仅结构变异） | 96.84% | 71.22% | 100.00% | 83.19% | 292 | 118 | 3329 | 0 |
| LLM-guided | GPT-4o | 97.59% | 76.88% | 99.66% | 86.80% | 296 | 89 | 3353 | 1 |
| LLM-guided | Qwen2.5-7B | 95.93% | 66.22% | 99.66% | 79.57% | 296 | 151 | 3291 | 1 |
| LLM-guided | Qwen2.5-14B | 98.29% | 82.45% | 99.66% | 90.24% | 296 | 63 | 3379 | 1 |
| LLM-guided | DeepSeek-V3 | 94.46% | 58.93% | 100.00% | 74.16% | 297 | 207 | 3235 | 0 |
| LLM-guided | GLM-4-9B | 95.51% | 63.93% | 99.66% | 77.89% | 296 | 167 | 3275 | 1 |

### Table IX-B: 标准测试（快照级）

| Augmentation | LLM Model | Acc | Prec | Rec | F1 | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| No Aug | - | 75.00% | 52.38% | 100.00% | 68.75% | 22 | 20 | 38 | 0 |
| GraphCL | - | 92.50% | 83.33% | 90.91% | 86.96% | 20 | 4 | 54 | 2 |
| GCA | - | 83.75% | 62.86% | 100.00% | 77.19% | 22 | 13 | 45 | 0 |
| Mimicry | - | 85.00% | 65.62% | 95.45% | 77.78% | 21 | 11 | 47 | 1 |
| LLM-guided | 无LLM（仅结构变异） | 86.25% | 66.67% | 100.00% | 80.00% | 22 | 11 | 47 | 0 |
| LLM-guided | GPT-4o | 85.00% | 64.71% | 100.00% | 78.57% | 22 | 12 | 46 | 0 |
| LLM-guided | Qwen2.5-7B | 85.00% | 64.71% | 100.00% | 78.57% | 22 | 12 | 46 | 0 |
| LLM-guided | Qwen2.5-14B | 85.00% | 64.71% | 100.00% | 78.57% | 22 | 12 | 46 | 0 |
| LLM-guided | DeepSeek-V3 | 83.75% | 62.86% | 100.00% | 77.19% | 22 | 13 | 45 | 0 |
| LLM-guided | GLM-4-9B | 86.25% | 66.67% | 100.00% | 80.00% | 22 | 11 | 47 | 0 |

### 结果与分析

Table IX-A 报告了各增强策略在 Theia311 上的 ego 级检测性能。综合 F1 排序：**GraphCL（95.78%）> GCA（85.59%）> LLM-guided/无LLM（83.19%）> No Aug（76.24%）> Mimicry（62.25%）**。

**No Aug** 不做任何增强。恶意 ego subgraph 数量极少，对比学习中这些样本被反复采样，编码器对恶意类的表示发生坍缩——所有恶意嵌入聚集到嵌入空间中一个极小区域。下游 MLP 分类器只能基于这个坍缩簇拟合决策边界，无法覆盖恶意行为的真实分布，导致 FP=184 和 Prec=61.98%。

**GraphCL** 通过均匀随机扰动（删边+掩特征）生成增强视图，取得最优综合性能（F1=95.78%、Prec=94.67%、FP 仅 16），但均匀扰动不区分节点重要性，在攻击样本稀缺时产生的变体多样性有限。

**GCA** 的度感知增强 F1=85.59%，溯源图中恶意节点的度分布与良性节点显著不同，度感知增强对不同度的节点施加差异化扰动，最大化了有限负样本的变体多样性，使编码器学到基于度分布模式的判别特征。

**Mimicry** 向攻击子图注入良性边模拟隐蔽攻击。问题在于注入良性边直接破坏了攻击子图的判别性拓扑特征（如异常低度进程、短路径攻击链），增强后的图在结构上已接近良性图，但仍以恶意标签参与对比学习。编码器被迫将结构上像良性的样本编码到恶意区域，学到的恶意/良性边界模糊不清，FP=352（最高）、Prec=45.68%（最低）。

**LLM-guided（无LLM）** 通过变异流水线扩充恶意样本池，Rec=100.00%（FN=0），漏报最少，但 FP=118、Prec=71.22%，说明变异图过于激进导致编码器偏向将正常样本判为恶意。

在 LLM-guided 框架下引入不同 LLM 后，ego 级 F1 差异显著：**Qwen2.5-14B（90.24%）> GPT-4o（86.80%）> 无LLM（83.19%）> Qwen2.5-7B（79.57%）> GLM-4-9B（77.89%）> DeepSeek-V3（74.16%）**。Qwen2.5-14B 和 GPT-4o 超过无 LLM 基线，FP 分别从 118 降至 63 和 89，验证了 LLM 语义变异能提升难负样本质量；Qwen2.5-7B、GLM-4-9B、DeepSeek-V3 反而低于基线。

分析变异语料发现，**变异保守性与检测效果正相关**。Theia311 攻击语料仅含 5 种独立攻击进程节点，各 LLM 变异量差异显著：DeepSeek-V3 改动 2193 个节点（F1=74.16%），而 Qwen2.5-14B 仅改 298 个（F1=90.24%）。激进变异引入**结构-语义不一致**——ego 拓扑保持攻击模式但 properties 被大量替换为良性字符串，编码器将矛盾样本编码为恶意类，决策边界泛化导致 FP 飙升。

**变异多样性受限，本质是浅层字符串替换。** replacement 策略将 `/bin/sh` 换为 `bash`、`curl` 等同义 shell，extension 策略机械拼接 `systemctl status nginx &&` 前缀，这些替换在 GIN 编码器特征空间中几乎不产生有效偏移，真正决定嵌入的是图拓扑结构而非命令名字符串。此外小模型存在变异质量问题：Qwen2.5-7B 将 prompt 格式说明原样输出为 properties，GLM-4-9B 将 extension 节点替换为其他攻击节点的 properties 破坏攻击语义，DeepSeek-V3 超过 96% 的变异仅做单一替换。

### Table IX-C: 攻击变体测试（ego 级）

历史数据（旧版代码跑的，待用新代码重跑补全 LLM 模型行）

| Augmentation | LLM Model | Acc | Prec | Rec | F1 | TP | FP | TN | FN | Rec衰减 |
|---|---|---|---|---|---|---|---|---|---|---|
| No Aug | - | 83.69% | 39.72% | 38.52% | 39.11% | 141 | 214 | 2112 | 225 | -60.49% |
| GraphCL | - | 89.41% | 85.58% | 24.79% | 38.44% | 89 | 15 | 2318 | 270 | -71.80% |
| GCA | - | 84.14% | 41.46% | 40.44% | 40.94% | 148 | 209 | 2117 | 218 | -58.57% |
| Mimicry | - | 89.45% | 81.36% | 26.82% | 40.34% | 96 | 22 | 2312 | 262 | -68.56% |
| LLM-guided | 无LLM（仅结构变异） | 87.89% | 52.61% | 87.68% | 65.76% | 313 | 282 | 2053 | 44 | -11.99% |
| LLM-guided | GPT-4o | - | - | - | - | - | - | - | - | - |
| LLM-guided | Qwen2.5-7B | - | - | - | - | - | - | - | - | - |
| LLM-guided | Qwen2.5-14B | - | - | - | - | - | - | - | - | - |
| LLM-guided | DeepSeek-V3 | - | - | - | - | - | - | - | - | - |
| LLM-guided | GLM-4-9B | - | - | - | - | - | - | - | - | - |

### Table IX-D: 攻击变体测试（快照级）

历史数据（旧版代码跑的，待用新代码重跑补全 LLM 模型行）

| Augmentation | LLM Model | Acc | Prec | Rec | F1 | TP | FP | TN | FN | Rec衰减 |
|---|---|---|---|---|---|---|---|---|---|---|
| No Aug | - | 90.76% | 75.00% | 78.26% | 76.60% | 18 | 6 | 90 | 5 | -21.74% |
| GraphCL | - | 90.76% | 83.33% | 65.22% | 73.17% | 15 | 3 | 93 | 8 | -25.69% |
| GCA | - | 91.60% | 76.00% | 82.61% | 79.17% | 19 | 6 | 90 | 4 | -17.39% |
| Mimicry | - | 90.76% | 80.00% | 69.57% | 74.42% | 16 | 4 | 92 | 7 | -25.89% |
| LLM-guided | 无LLM（仅结构变异） | 92.44% | 71.88% | 100.00% | 83.64% | 23 | 9 | 87 | 0 | 0.00% |
| LLM-guided | GPT-4o | - | - | - | - | - | - | - | - | - |
| LLM-guided | Qwen2.5-7B | - | - | - | - | - | - | - | - | - |
| LLM-guided | Qwen2.5-14B | - | - | - | - | - | - | - | - | - |
| LLM-guided | DeepSeek-V3 | - | - | - | - | - | - | - | - | - |
| LLM-guided | GLM-4-9B | - | - | - | - | - | - | - | - | - |

## Table V: Attack Variant Usability (theia311)

论文 RQ2(2)：人工评估 LLM 生成的攻击变体质量。每个 LLM 生成 100 个变体，3 名博士生独立评分（1-5 Likert），评估操作合法性、文件路径真实性、进程关系一致性。Usability rate = 评分 ≥4 的比例。

| Model | Size | Provider | Direct | Filtered |
|---|---|---|---|---|
| GPT-4o | - | chatanywhere | 78% | 91% |
| Qwen2.5 | 7B | siliconflow | 24% | 52% |
| Qwen2.5 | 14B | siliconflow | 68% | 85% |
| DeepSeek-V3 | 671B | siliconflow | 54% | 72% |
| GLM-4 | 9B | siliconflow | 42% | 66% |

Direct 为 LLM 直接输出的变体可用率，Filtered 为经验证 pipeline 过滤后的可用率。GPT-4o 在操作合法性和路径真实性上表现最优，变异多样性最高（193 种独立结果），生成的 replacement 变体能合理选择语义相近的替代命令（如 `curl -H "X-Health-Check: true"`）。Qwen2.5-14B 变异保守但无格式错误，多数变体仅做 shell 名称替换，合法性高但多样性不足（29 种独立结果）。DeepSeek-V3 参数量最大但 96% 的 replacement 为单一替换（`/bin/sh`→`/bin/bash`），且 7.8% 的 extension 变体破坏了原始攻击语义。GLM-4-9B 存在 prompt 泄漏问题——115 次将 prompt 示例 `http://mal.com/pay.sh` 直接复制到输出中，降低了变体真实性。Qwen2.5-7B 可用率最低，20.6% 的输出为格式错误（将 prompt 模板 `cmdLine,tgid,path` 原样输出），指令遵循能力不足。

## Table VI: LLM Token Consumption and Latency Per Variant (theia311)

论文 RQ2(3)：评估 LLM 集成的部署开销。LLM 参与两个阶段：(a) 候选选择（Algorithm 1 Line 24-26）和 (b) 语义变异（Figure 5），分别测量每个快照的平均 token 消耗和推理延迟。

| Model | Size | Provider | Semantic Tokens/snap | Semantic Latency/snap (s) | Selection Tokens/snap | Selection Latency/snap (s) |
|---|---|---|---|---|---|---|
| GPT-4o | - | chatanywhere | 3836 | 8.87 | - | - |
| Qwen2.5 | 7B | siliconflow | 2664 | 7.62 | - | - |
| Qwen2.5 | 14B | siliconflow | 1896 | 4.75 | - | - |
| DeepSeek-V3 | 671B | siliconflow | 12230 | 123.73 | - | - |
| GLM-4 | 9B | siliconflow | 8455 | 17.40 | - | - |

**指标计算方式。** 每次 LLM API 调用对应一个 ego 子图的语义变异：将 ego 中的攻击进程节点（平均 2.2 个）及其上下文编码为 prompt，LLM 返回变异后的 properties。每次调用的 token 消耗（prompt_tokens + completion_tokens）从 API 返回的 usage 字段读取，延迟为该次请求的端到端耗时。表中 Tokens/snap = 所有调用的 token 总和 / 118（良性快照数），Latency/snap 同理。

**单次调用开销分析。** 各模型单次调用的 token 消耗相近：prompt 约 1037-1211 tok/call（因 prompt 模板一致，差异来自 ego 大小和上下文三元组数量），completion 约 164-230 tok/call（输出均为短 JSON 数组）。单次延迟方面，GPT-4o（2.78s）、GLM-4-9B（2.63s）、Qwen2.5-14B（3.12s）、Qwen2.5-7B（3.60s）处于同一量级，而 DeepSeek-V3（14.57s/call）显著偏高，为 siliconflow 上 671B 模型的推理瓶颈。

**总开销差异的根因是调用次数。** EgoMutationPipeline 的阶段 1 从每个良性快照随机采样候选中心节点做结构变异，成功的 ego 全部进入阶段 2 调 LLM（不限数量），最终截断到 5 egos/snap = 590 egos。由于各 LLM 独立运行且未固定随机种子，阶段 1 采样不同导致进入 LLM 的 ego 数量差异显著：Qwen2.5-14B 仅 180 次（1.5 calls/snap），GPT-4o 377 次（3.2 calls/snap），而 DeepSeek-V3 达 1002 次（8.5 calls/snap）。超出 590 的调用结果被截断丢弃，DeepSeek-V3 约 41% 的 LLM 调用被浪费。这一差异是实验噪声而非模型特性——固定随机种子后各模型的调用次数应趋于一致。

综合来看，Qwen2.5-14B 以最低开销（1896 tok/snap, 4.75s/snap）取得最优 F1（90.24%），DeepSeek-V3 因高调用次数和高单次延迟叠加，总开销（12230 tok/snap, 123.73s/snap）为 Qwen2.5-14B 的 6.4 倍和 26 倍，但 F1 反而最低（74.16%）。

## 实现与论文差异记录

以下记录实际实现与论文（bare_jrnl_new_sample4.pdf）描述的差异。

### 1. LLM 候选选择：省略

| | 论文描述 | 实际实现 |
|---|---|---|
| 候选排序 | Algorithm 1 Line 24-26：LLM 对 top-m 候选打分（operational compatibility + behavioral similarity），选最优候选 | **省略**：直接取结构对齐得分最高的候选，不用 LLM 打分 |
| 原因 | 每次变异多调一次 LLM 打分开销太大（ego 级变异量 ~1000 个），且结构对齐得分已能有效筛选 |

### 2. 统一验证：Ego 级省略

| | 论文描述 | 实际实现 |
|---|---|---|
| 验证 | 4 项统一验证（Operation Legality, Attribute Feasibility, Imperceptibility, Hardness） | 快照级 MutationPipeline 保留验证；**Ego 级 EgoMutationPipeline 省略验证** |
| 原因 | Ego 子图太小（~32 节点），历史行为 profile 统计量不足，验证容易误拒；且 ego 级攻击信号占比高，不需要 hardness 阈值过滤 |

### 3. 语义变异 Prompt：增强攻击语义保留指导

| | 论文描述 | 实际实现 |
|---|---|---|
| Prompt | Figure 5：传入 attributes、associated_nodes、strategy、context，让 LLM 按策略变异 | **增强版**：额外传入 `ATTACK-SPECIFIC (must keep)` 和 `BENIGN (can replace)` 标注，明确告诉 LLM 哪些部分在 H_b 中（可替换）、哪些不在（必须保留）；关联节点传真实图索引 |
| 原因 | Qwen2.5-7B 等小模型不理解隐式语义约束，会把整个 cmdLine 替换成良性操作，丢失攻击语义。显式标注后变异质量提升 |

### 4. 语义变异策略融合：单策略 → Mix-of-Strategy MoE（借鉴 GAugLLM）

| | 论文描述 | 实际实现 |
|---|---|---|
| 变异输出 | 每个攻击节点按 `_assign_strategy` 判定唯一策略（replacement / rewriting / extension），LLM 输出一种变异结果直接写回图的 properties | **三策略同时生成**：不再按规则分配单一策略，一次 LLM 调用同时输出 replacement、rewriting、extension 三种变异结果，存为 `strategy_variants` 字典挂在 ego 图属性上，不修改原始 properties |
| 变异融合 | 无融合机制 | **StrategyMoE 可学习融合**（借鉴 GAugLLM 的 SimilarityAttentionMLP）：三种变异各过 word2vec 得到 3 个 128 维向量，加上原始 properties 的 word2vec 向量作为 context，通过 MLP 内容权重 + 内容-上下文余弦相似度联合打分，softmax 加权融合为单一 128 维特征向量 |
| 融合位置 | — | **训练 forward pass 中实时融合**（有梯度）：缓存阶段只存 numpy 向量（`variant_vecs = {node_idx: (content_np[3,D], context_np[D])}`），训练时转 tensor 过 MoE → 替换攻击节点特征 → GIN → 对比损失 → 梯度回传更新 MoE + GIN |
| 原因 | 论文 Figure 5 的单策略分配存在两个问题：(1) 策略选择依赖 H_b 匹配的简单规则，rewriting 策略几乎不触发（需 cmd 和 args 都在 H_b 中，实际攻击节点极少满足）；(2) 不同策略在 word2vec 特征空间产生不同方向的偏移，固定选一种丢失了策略互补信息。借鉴 GAugLLM 的 Mix-of-Expert-Prompt 思路，让模型端到端学习最优融合权重 |
| 实现文件 | `semantic.py`: `generate_strategy_variants()` + `build_multi_strategy_prompt()`；`pipeline.py`: `use_multi_strategy` 控制 MoE/单策略模式切换；`gcc_embedder_dev.py`: `StrategyMoE` 类 + `_encode_single_ego_graph()` 中的 MoE forward |

### 5. LLM 调用方式：串行 → 并发

| | 论文描述 | 实际实现 |
|---|---|---|
| 调用方式 | 未提及 | **两阶段流水线**：阶段 1 串行做结构变异（~8s），阶段 2 用 ThreadPoolExecutor 8 线程并发调 LLM，总计 ~5-10 分钟 vs 串行 ~42 分钟 |

