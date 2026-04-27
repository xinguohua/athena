## 实现与论文差异记录

以下记录实际实现与论文描述之间的关键差异及其设计动机。每项差异均附有实验验证或理论分析。

### 完整网络结构

#### Stage 1: 自适应对比学习（训练 GIN Encoder + StrategyMoE）

**输入。** 溯源图快照序列 [G_0, G_1, ..., G_198]，其中良性快照 [0, 118]，恶意快照 [119, 198]。每个训练 step 处理一个良性快照 G_t。

**正样本构建。** 从 G_t 中按频次采样中心节点，以每个中心节点为起点 BFS 提取 ego 子图（max_nodes=5）。每个 ego 内的节点 properties 经 word2vec tokenize（正则切分为 token 序列）→ word2vec 查表 → mean pooling 得到 128 维特征向量。所有节点特征组成矩阵 X ∈ R^{n×128}，连同边索引 edge_index 和边类型 edge_feat（4 类：进程/文件/网络/内存操作）输入 GIN 编码器。编码前施加随机增强：以概率 p=0.2 删除边，以概率 p=0.2 掩盖节点特征维度。

**GIN Encoder（3 层 TypedGINConv）。** 每层按边类型分 4 组独立聚合邻居消息，拼接后与自身特征一起经 MLP 变换。维度变化：128 → 64 → 64 → 256。激活函数为 ReLU。输出 H ∈ R^{n×256}，经 mean pooling 得到图级嵌入 graph_emb ∈ R^{1×256}，再经 projection head（Linear(256→256) → ReLU → Linear(256→256)）和 L2 归一化得到 z_pos ∈ R^{1×256}。

**负样本构建（来源 1：_mal_ego_pool）。** 从攻击快照中以 label=1 的攻击节点为中心 BFS 提取 ego（max_nodes=5），随机采样 Bc 个，经相同的 word2vec → GIN → proj head 路径编码为 z_neg_1 ∈ R^{Bc×256}。

**负样本构建（来源 2：mutation_map 变异 ego）。** 结构变异（子图替换）生成的变异 ego，自适应大小（ego_size = n_attack × 2）。良性节点正常经 word2vec 编码；攻击节点在有 variant_vecs 时经 StrategyMoE 融合：

```
攻击节点处理流程：
  缓存中取出: content [3, 128] = 三种变异的 word2vec 向量
              context [128]    = 原始 properties 的 word2vec 向量
       ↓
  StrategyMoE:
    Step 1: 3 个 content 各过 Linear(128→64) + LeakyReLU，拼接后 Linear(192→3) = w_content [3]
    Step 2: sim_i = content_i · context（点积）= w_sim [3]
    Step 3: weights = softmax((w_content + w_sim) / 0.2) [3]
    Step 4: fused = Σ weights_i × content_i [128]
       ↓
  替换 x_t 中该攻击节点的特征向量
```

融合后的 x_t 经共享权重的 GIN Encoder → mean pooling → proj head → normalize 得到 z_neg_2。

**对比损失与梯度流。** 正负样本嵌入经加权对比损失（温度 τ=0.07）计算 loss，反向传播更新三组参数：GIN Encoder、projection head、StrategyMoE。梯度路径：loss → proj head → GIN → StrategyMoE MLP（通过 fused 向量回传）。

#### Stage 2: MLP 分类器（冻结 GIN）

**训练集。** train_ego_cache 中的所有 ego（良性 ~16673 + 攻击 ~657，含 590 变异 ego），1:1 平衡采样。

**特征提取。** 每个 ego 经增强（删边 p=0.2 + 掩特征 p=0.2）后，冻结的 GIN 编码，取 [center 节点嵌入 ‖ 全图 mean 嵌入] 拼接为 512 维特征向量。

**MLP 分类器。** Linear(512→128) → ReLU → Dropout(0.1) → Linear(128→2)，CrossEntropyLoss 训练 10 个 epoch，仅更新 MLP 参数。

#### 测试（推理）

恶意时间段快照 [119, 198] 中的每个节点（攻击节点全取 + 良性节点采样 50 个）→ BFS 提取 ego（max_nodes=5）→ word2vec → 冻结 GIN → [center ‖ mean] → MLP → softmax → 恶意/良性。Ego 级判定：子图含 label=1 节点的 ego 为真攻击。快照级判定：任一 ego 被判为恶意则整个快照为恶意。

#### 核心维度汇总

| 位置 | 维度 | 说明 |
|------|------|------|
| word2vec 输出 | 128 | prop_feat_dim，节点特征 |
| StrategyMoE 输入/输出 | 128 | 在 word2vec 空间操作，GIN 之前 |
| GIN 各层 | 128→64→64→256 | 3 层 TypedGINConv |
| graph embedding | 256 | enc_out_dim，mean pooling 后 |
| projection head | 256→256 | 对比学习投影 |
| MLP 输入 | 512 | center(256) + mean(256) 拼接 |
| MLP 输出 | 2 | 二分类（良性/恶意） |

---

### 1. 变异粒度：快照级 → Ego 级

**论文描述。** Algorithm 1 在整个快照 G_b（~500 节点）上执行子图替换，生成变异图 G̃_b 作为难负样本。

**实际实现。** 采用 EgoMutationPipeline，在 ego 子图粒度执行 WL 匹配与子图替换。ego 大小根据攻击子图动态确定（`ego_size = max(5, n_attack_nodes × 2)`），使攻击节点占比稳定在 30-60%。

**设计动机。** 快照级变异存在严重的信号稀释问题：攻击节点仅占快照的 ~0.4%（如 Theia311 中 2000 节点快照含 3-5 个攻击节点），结构变异注入的攻击子图被良性上下文淹没。实验发现固定 32 节点 ego 中攻击节点仅占 9.2%，其余 78.3% 为良性节点（MemoryObject 占 83.9%），GIN 的 mean pooling 将攻击特征稀释到可忽略的程度。自适应 ego 大小将攻击占比提升至 ~50%，同时将负样本池、训练正样本和测试集的 ego 提取统一到相同的尺度约束，避免训练-测试尺度不匹配导致的系统性偏差。

### 2. LLM 候选选择：省略

**论文描述。** Algorithm 1 Line 24-26 要求 LLM 对 top-m 候选进行 operational compatibility 和 behavioral similarity 双维度评分，选择最优候选。

**实际实现。** 省略 LLM 评分步骤，直接采用结构对齐（BFS + Jaccard 类型匹配）得分最高的候选。评分综合攻击节点覆盖率（attack_ratio，权重 2.0）与类型一致性（rho）。

**设计动机。** Ego 级变异的候选量为 ~590 个（5 egos/snapshot × 118 snapshots），每个候选若额外调用一次 LLM 评分，将引入 ~590 次 API 调用（约 10 分钟延迟）。实验表明结构对齐得分已能有效筛选高质量候选，结构变异成功率达 100%（590/590），额外 LLM 评分的边际收益有限。

### 3. 统一验证：Ego 级省略

**论文描述。** 变异后的图需通过 4 项统一验证：Operation Legality、Attribute Feasibility、Imperceptibility、Hardness。

**实际实现。** 快照级 MutationPipeline 保留完整验证；Ego 级 EgoMutationPipeline 省略验证。

**设计动机。** 验证依赖历史行为 profile（entity_ops、type_attrs）的统计显著性。Ego 子图仅含 5-10 个节点，统计量不足导致验证阈值不稳定，实测拒绝率 >60%，有效变异产出过低。自适应 ego 大小下攻击节点占比 ~50%，结构变异本身已保证足够的判别性，Hardness 过滤不再必要。

### 4. 语义变异 Prompt 增强

**论文描述。** Figure 5 的 prompt 传入 attributes、associated_nodes、strategy 和 r-hop context C，由 LLM 自行判断哪些属性可修改、哪些需保留。策略选择由 `_assign_strategy` 基于良性语料 H_b 的匹配规则预先决定（replacement / rewriting / extension），每次 LLM 调用仅输出一种策略的变异结果。

**实际实现。** 对 prompt 进行三项增强：

**(1) 显式攻击语义标注。** 在每个待变异节点的 prompt 段中，基于 H_b 匹配分析结果，显式标注 `ATTACK-SPECIFIC (must keep)` 和 `BENIGN (can replace)` 字段。具体来说，对进程节点的 properties（格式为 `cmdLine,tgid,path`）：解析 cmdLine 为命令名和参数，分别检查是否出现在良性语料集合 `benign_commands` 和 `benign_args` 中。不在 H_b 中的部分标记为 ATTACK-SPECIFIC（如 `-c ./gtcache &>/dev/null &`），在 H_b 中的部分标记为 BENIGN（如 `/bin/sh`）。LLM 被明确要求在所有变异输出中保留 ATTACK-SPECIFIC 部分。

**(2) 关联节点索引传递。** 将攻击节点的邻居中文件和网络类型节点的**真实图索引**传入 prompt（如 `id=10 type=file_object_block properties=/etc/firefox/native-messaging-hosts/gtcache`），使 LLM 在 `associated_updates` 中能精确指定需同步更新的关联节点，避免索引错位导致的更新失败。

**(3) 策略说明与示例强化。** 对三种变异策略提供 BAD/GOOD 示例，引导 LLM 产生足够深度的变异。例如 replacement 策略：BAD 示例为 `/bin/sh → /bin/bash`（仅 1 个 token 变化），GOOD 示例为 `/bin/sh -c ./gtcache &>/dev/null & → env LANG=C /usr/bin/perl -e 'exec("./gtcache")' &>/dev/null &`（命令结构重组）。

**设计动机。** 实验分析 5 种 LLM（GPT-4o、Qwen2.5-7B/14B、DeepSeek-V3、GLM-4-9B）在 Theia311 上生成的 2524 个变异节点发现，未增强 prompt 下存在三类质量问题：(1) **格式错误**——Qwen2.5-7B 有 20.6% 的输出将 prompt 模板（`cmdLine,tgid,path`）原样输出为 properties，引入无效特征向量；(2) **攻击语义破坏**——GLM-4-9B 有 14.2% 的 extension 策略节点被替换为其他攻击节点的 properties（如将 `pass_mgr` 节点输出为 `./gtcache`），混淆了攻击类型边界；(3) **prompt 泄漏**——GLM-4-9B 有 115 次将 prompt 示例中的 `http://mal.com/pay.sh` 直接复制到变异输出中。增强后攻击语义丢失率从 ~15% 降至 1.7%，策略间区分度达到 100%（三种变异互不相同）。

### 5. 语义变异策略融合：Mix-of-Strategy MoE

**论文描述。** 每个攻击节点由 `_assign_strategy` 基于 H_b 匹配规则分配唯一策略（replacement / rewriting / extension），LLM 输出单一变异结果直接写回图节点 properties。

**问题分析。** 单策略分配在实际数据上暴露出三个局限：

(1) **策略覆盖不均**。策略选择规则要求 rewriting 在命令名和参数同时出现在 H_b 中时触发，但 Theia311 的攻击节点（如 `./gtcache`）的命令名或参数至少有一项不在 H_b 中，导致 rewriting 的触发次数为 0。实际仅 replacement（命令名在 H_b）和 extension（命令名不在 H_b）交替使用，浪费了 rewriting 策略的潜在贡献。

(2) **变异多样性不足**。5 种独立攻击进程节点反复被同一策略处理，replacement 策略 96.3% 的变异为单一替换（如 `/bin/sh → /bin/bash`），extension 策略机械拼接相同前缀（`systemctl status nginx &&`，59 次）。变异后的 word2vec 编码与原始编码的 L2 距离 <0.01，对比学习的负样本在特征空间中缺乏有效偏移。

(3) **策略互补信息丢失**。三种策略对同一攻击节点产生的 word2vec 偏移方向不同：replacement 主要改变命令名相关的 token，extension 引入新的前缀 token，rewriting 重组整个 token 序列。固定选一种策略等于丢弃了其他两种方向的特征扰动信息。

**实际实现。** 借鉴 GAugLLM [KDD'24] 的 Mix-of-Expert-Prompt（MoEP）机制，将单策略输出改为三策略并行生成 + 可学习加权融合。实现分为离线生成和在线融合两个阶段。

**离线阶段：三策略变异生成。** 对每个变异 ego 的攻击进程节点，一次 LLM 调用同时输出三种策略的变异结果（`build_multi_strategy_prompt`）。LLM 返回 JSON 数组，每个元素包含 `replacement`、`rewriting`、`extension` 三个字段。解析后存为 `strategy_variants` 字典挂在 ego 图属性上，**不修改原始 properties**。变异结果的 word2vec 向量在缓存阶段预计算（`variant_vecs = {node_idx: (content_np[3,D], context_np[D])}`），其中 content 为三种变异编码，context 为原始 properties 编码。

**在线阶段：StrategyMoE 可学习融合。** 训练时在 `_encode_single_ego_graph` 的 forward pass 中，对有 `variant_vecs` 的攻击节点实时执行 MoE 融合。

**输入。** 从缓存取出三种变异的 word2vec 向量 content ∈ R^{K×3×D} 和原始 properties 的 word2vec 向量 context ∈ R^{K×D}（K 为该 ego 中有变异的攻击节点数，D=128）。content 的三个分量分别对应 replacement（emb_r）、rewriting（emb_w）、extension（emb_e）的编码；context 是变异前原始攻击节点 properties 的编码，作为"攻击语义锚点"。

**权重计算由两个信号联合决定：**

**信号 1：w_content（MLP 内容评分）——衡量"变异本身对下游任务的价值"。** 三种变异向量各过共享的 Linear(128→64) + LeakyReLU 得到隐层表示 h_r, h_w, h_e ∈ R^{K×64}，拼接为 192 维后经 Linear(192→3) 输出 w_content ∈ R^{K×3}。MLP 同时看到三种变异的表示，学到的是**变异之间的相对质量**：如果某种变异产出了与其他两种差异显著的特征向量（提供了独特的扰动方向），MLP 倾向给它更高分数。这个分数完全数据驱动，通过对比损失的梯度端到端学习。

```
emb_r → Linear(128→64) → LeakyReLU → h_r
emb_w → Linear(128→64) → LeakyReLU → h_w    ← 三者共享同一个 Linear
emb_e → Linear(128→64) → LeakyReLU → h_e
concat(h_r, h_w, h_e) → Linear(192→3) → w_content = [s_r, s_w, s_e]
```

**信号 2：w_sim（内容-上下文点积相似度）——衡量"变异保留了多少原始攻击语义"。** 计算每种变异与原始向量的点积：sim_i = content_i · context ∈ R^{K×1}，拼接为 w_sim ∈ R^{K×3}。点积越高表示变异越保守（与原始攻击特征越接近，攻击语义保留越好），点积越低表示变异越激进（偏离原始攻击特征越远）。该信号不含可学习参数，直接由向量几何关系决定，提供稳定的语义保留度先验。

```
sim_r = emb_r · context    ← 标量，replacement 与原始的相似度
sim_w = emb_w · context    ← rewriting 与原始的相似度
sim_e = emb_e · context    ← extension 与原始的相似度
w_sim = [sim_r, sim_w, sim_e]
```

**两个信号相加的设计动机。** w_content 和 w_sim 衡量的是正交的两个维度：前者衡量"对训练有多大价值"（动态学习），后者衡量"对攻击语义保留有多好"（静态计算）。相加意味着**两者都要好**——一个变异即使 MLP 评分高（对比学习有用）但与原始特征完全不相似（攻击语义丢失），总分也不会高；反过来，与原始几乎一样（语义保留好）但 MLP 评分低（没提供新信息），总分也一般。只有**既对训练有价值、又保留攻击语义的变异**才能获得高权重。

**Softmax 温度与融合行为。** 两个信号相加后经温度缩放 softmax 归一化：weights = softmax((w_content + w_sim) / τ)，τ=0.2。低温度（τ=0.2）使 softmax 输出接近 one-hot 分布——大部分情况下模型会**主选一种最优策略**而非均匀混合三种。这与 GAugLLM 的设计一致：在 4 种 prompt 专家中，低温度让模型对每个节点自适应地选择最合适的专家，而非简单平均。

**最终融合。** fused = Σ_i weights_i × content_i ∈ R^{K×D}。将 fused 替换攻击节点在 x_t 中的特征向量，后续经 GIN 编码 → mean pooling → projection head → 对比损失。梯度从对比损失回传经 GIN → fused（加权和的梯度流向各 content_i 和 weights）→ StrategyMoE 的 MLP 参数（通过 w_content 分支），实现端到端优化。w_sim 分支无可学习参数，但通过 softmax 间接影响 content 的梯度分配。

**梯度流设计。** 早期实现（v1）在缓存构建阶段执行 MoE forward 并通过 `detach().cpu().numpy()` 存储结果，导致 MoE 参数从未收到梯度更新（F1=63.89%）。修正后将 MoE 移至训练 forward pass 内，缓存阶段仅存储原始 numpy 向量，训练时动态转为 torch tensor 执行 MoE（F1 提升至 85.07%）。StrategyMoE 的参数同时加入优化器和梯度裁剪（max_norm=5.0），与 GIN encoder 和 projection head 联合训练。

**自适应 ego 大小与 MoE 的协同。** MoE 融合的有效性依赖攻击节点在 ego 中的占比。当 ego 含 32 个节点但仅 3 个攻击节点时，MoE 修改的 3 个节点特征在 GIN mean pooling 中被 29 个未修改的良性节点（80% 为 MemoryObject）稀释。实验发现 FP 从 63（原版）暴增至 340（MoE+32 节点），误报集中在 MemoryObject 密集的良性 ego（如 sshd、bash、thunderbird 的内存操作）——编码器学到了"MemoryObject 密集结构 = 恶意"的错误模式。引入自适应 ego 大小（`ego_size = max(5, n_attack × 2)`）后，攻击占比提升至 ~50%，FP 降至 34，MoE 的特征修改在 mean pooling 中获得足够权重。

**实验验证。**

| 版本 | ego F1 | Prec | Rec | FP | FN | 关键改动 |
|---|---|---|---|---|---|---|
| 单策略 Qwen14B（对照） | 90.24% | 82.45% | 99.66% | 63 | 1 | 原版实现 |
| v1: MoE 缓存融合, ego=32 | 63.89% | 46.95% | 100% | 330 | 0 | MoE 无梯度 + ego 过大 |
| v2: MoE forward 融合, ego=32 | 85.07% | 74.21% | 99.65% | 98 | 1 | 梯度修复，ego 仍过大 |
| v3: MoE forward, 自适应 ego | 88.38% | 88.07% | 88.69% | 34 | 32 | FP↓ 但 Rec↓（ego=5 过小） |

**当前局限与分析。** MoE 最优配置（v3, F1=88.38%）接近但未超过单策略对照（90.24%）。根因分析表明，word2vec 的 bag-of-tokens 平均编码将 LLM 创造的句子级语义差异（如 `/bin/sh -c ./gtcache` vs `env LANG=C /usr/bin/perl -e 'exec("./gtcache")'`）压缩为 128 维空间中的微小偏移（L2 距离 ~0.05），MoE 缺乏足够的输入区分度进行有效的策略选择。对照组之所以表现更好，恰恰因为 word2vec "抹平"了 LLM 的微小文本变动，攻击节点保留了原始特征，结构-语义一致性未被破坏。

### 6. LLM 并发调用与效率优化

**论文描述。** 未涉及 LLM 调用的并发策略。

**实际实现。** 采用两阶段流水线：阶段 1 串行执行结构变异（WL 匹配 + BFS 对齐 + 子图替换，~3s），阶段 2 使用 ThreadPoolExecutor 以 8 线程并发调用 LLM API 执行语义变异。阶段 1 对每个良性快照限制最多 5 个结构变异成功的 ego 进入阶段 2，避免冗余 LLM 调用。

**效率数据。** 590 个 ego 的语义变异耗时约 600s（0.7 ego/s），较串行方式加速约 4 倍。阶段 1 限制产出后，LLM 调用数从 ~1000 降至 590（减少 41% 冗余调用）。

### 7. JSON 解析容错

**论文描述。** 未涉及 LLM 输出的鲁棒解析。

**实际实现。** 在 `_parse_llm_response` 中加入容错机制：(1) 修复 JSON 对象间缺失逗号（`}\n{` → `},{`）；(2) 合并多段 JSON 数组（`]\n[` → `,`）。

**量化影响。** Qwen2.5-7B 在无容错时 69% 的输出因格式不合规被丢弃，容错后解析成功率从 31% 提升至 99%。GPT-4o 和 Qwen2.5-14B 的原始合规率 >98%，容错对其影响有限。

## 已知问题

- [ ] 结果方差大：同一配置多次运行差异显著（未固定随机种子），需多次运行取平均
- [ ] 攻击变体测试（Table IX-C/D）LLM 模型行数据不完整
- [ ] MoE 融合在 word2vec 特征管道下效果有限，待探索语义级编码器或 LLM 角色重定位
