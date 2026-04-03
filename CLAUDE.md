# Prographer / ATHENA

基于动态溯源图(Provenance Graph)的 APT 检测系统。论文 **"Interpretable Stealthy APT Detection via Contrastive Learning and Semantic Provenance Abstraction"**（投 IEEE Journal，第一作者 Guohua Xin）。

## 论文核心方法

1. **快照构建**：系统审计日志按 1 分钟时间窗口切分为溯源图快照序列
2. **自适应对比学习**：GIN 编码器 + GRU 时序更新，LLM-guided graph mutation 生成难负样本（WL kernel 检索 → 结构变异 → 语义变异 → 验证），加权对比损失
3. **MLP 分类器**：冻结编码器后，两层 MLP + cross-entropy 做快照级异常检测
4. **全局解释**：关键因果路径提取 → 双侧语义增强 → Sentence-BERT 匹配 ATT&CK 技术 → LCS 序列对齐

评估数据集：DARPA E3、E5、OpTC、ATLAS。Baseline：ProGrapher、Unicorn、ATLAS、MAGIC。

## 开发环境

- GitHub: `xinguohua/prographer`，分支 `main`
- 服务器: `nsas2020@labserver.dack.top:8950`
- 项目路径: `/home/nsas2020/fuzz/prographer`
- Conda 环境: `prographer` (Python 3.9)
- **代码直接在远程服务器上开发和运行**（服务器有完整数据集和依赖）。用 Claude Code 在服务器上写代码、运行、分析结果。

## 项目结构

```
process/
├── train_all.py / test_all.py       # 训练/测试入口
├── benchmark_augmentation.py        # Table IV 增强策略基准测试
├── config.py / config.yaml          # 参数 & 多环境路径配置
├── datahandlers/                    # 数据加载（DARPA/ATLAS/OPTC）
├── embedders/                       # 图嵌入
│   └── gcc_embedder_dev.py          # 主力编码器：GIN + GRU + 对比学习
├── classfy/                         # 分类器
│   ├── mlp_classify.py              # 论文方法：两层MLP + cross-entropy（有监督）
│   └── svm_classify.py              # TopK 偏离度（无监督，旧方法）
├── mutation/                        # LLM-guided mutation pipeline（论文 Algorithm 1）
│   ├── wl_kernel.py                 # WL subtree kernel 图相似度
│   ├── structural.py                # BFS 对齐子图搜索 + 子图替换
│   ├── semantic.py                  # 3种语义变异策略 + LLM prompt
│   ├── verification.py              # 4项统一验证
│   └── pipeline.py                  # MutationPipeline 串联三阶段
├── llm_clients/                     # LLM API 客户端
├── technique_semantic_mapper.py     # ATT&CK 技术映射（Chroma + LLM）
└── utils/                           # 工具（性能测量等）
```

## 技术栈

- Python 3, PyTorch, igraph, leidenalg, scikit-learn
- PyKEEN (知识图谱嵌入), pandas, numpy, orjson
- LangChain, ChromaDB, sentence-transformers (RAG/语义映射)
- psutil (性能监控)

## 常用命令

```bash
conda activate prographer && cd /home/nsas2020/fuzz/prographer

# 训练
python -m process.train_all

# 测试
python -m process.test_all

# Table IV 基准测试（单个策略）
python -m process.benchmark_augmentation --strategy no_aug --dataset cadets --scene cadets314

# Table IV 基准测试（全部 5 种策略）
python -m process.benchmark_augmentation --dataset cadets --scene cadets314
```

## 关键配置

训练参数在 `process/config.py` 中配置：
- `SNAPSHOT_SIZE`: 500（快照最大节点数）
- `SEQUENCE_LENGTH_L`: 12（快照序列长度）
- `MALICIOUS_WINDOW_MINUTES`: ±10 分钟（恶意事件窗口）
- `DETECTION_THRESHOLD`: 0.01

`train_all.py` 顶部的变量控制训练行为：
- `DATASET_NAME`: "cadets", "atlas", "theia", "trace", "clearscope", "optcday1"
- `EMBEDDER_NAME`: "gcc_dev", "gcc", "prographer", "unicorn", "roland", "transe", "word2vec"
- `CLASSIFY_NAME`: "topk", "prographer", "unicorn", "mlp"
- `GLOBAL_ID`: 用户标识符（模型文件后缀）

数据集路径在 `process/config.yaml` 中按环境（local/remote）配置。

## 架构模式

- **工厂模式**: `get_handler()`, `get_embedder_by_name()`, `get_classfy()` 创建可插拔组件
- **抽象基类**: `BaseProcessor`, `GraphEmbedderBase`, `BaseClassify` 定义统一接口
- **数据流**: 原始日志 → DataHandler → 图构建 → Partition(快照) → Embedder(嵌入) → Classifier(检测) → 可选 ATT&CK 映射
- **图数据结构**: igraph.Graph，节点属性 `name`/`type`/`label`/`properties`/`frequency`，边属性 `actions`/`timestamp`

## 关键文件路径（相对于 process/ 目录）

脚本通过 `python -m process.xxx` 运行，工作目录是项目根目录，但以下文件都在 `process/` 下：

- `process/snapshot_data_{GLOBAL_ID}.pkl` — 训练产生的快照数据
- `process/data/mitreembed_master_Chroma.csv` — ATT&CK 技术向量库源数据
- `process/chroma_db/` — Chroma 向量库持久化目录
- `process/local_settings.py` — API 密钥（gitignore）
- `process/technique_sequences.txt` — 攻击技术序列库

写新脚本时引用这些文件必须用 `os.path.join(os.path.dirname(__file__), ...)` 拼绝对路径，不能用相对路径。

## 当前实验 TODO（论文 bare_jrnl_new_sample4.pdf）

按顺序实现并跑数据：

| # | 位置 | TODO | 状态 |
|---|------|------|------|
| 1 | RQ2 p11 | **Table IV**: 不同增强策略检测性能 (No aug/GraphCL/GCA/Mimicry/LLM-guided) | 代码已写，待跑 |
| 2 | RQ2 p11 | **Table V**: attack variant usability 分析 | 待做 |
| 3 | RQ4 p12 | **RQ3-TODO-1**: baseline throughput/latency 对比 | 待做 |
| 4 | RQ4 p12 | **RQ3-TODO-2**: 离线训练资源开销分析 | 待做 |
| 5 | RQ4 p12 | **RQ3-TODO-3**: 拆分在线推理 throughput | 待做 |
| 6 | RQ4 p12 | **RQ3-TODO-4**: Table VIII 端到端性能对比 | 待做 |
| 7 | RQ5 p13 | **RQ4-TODO-1**: accuracy gain vs overhead trade-off | 待做 |
| 8 | RQ5 p13 | **RQ4-TODO-1**: Table IX LLM 集成影响分析 | 待做 |
| 9 | RQ5 p13 | **RQ4-TODO-2**: weighted contrastive loss vs cross-entropy 实验 | 待做 |
| 10 | RQ6 p13 | **RQ5-TODO-1**: 参数实验(D/W)用新方法重跑 | 待做 |
| 11 | RQ7 p14 | **RQ6-TODO-1**: LOTL 隐蔽攻击评估实验 | 待做 |
| 12 | Discussion p14 | **DISC-TODO-1**: 长驻留时间 APT 检测讨论 | 待做 |
| 13 | Discussion p14 | **DISC-TODO-2**: concept drift 讨论 | 待做 |

### Table IV 已知问题（待修复）

1. `llm_guided` 策略的 `llm_model=None`：语义变异退化为规则 fallback，需配置 LLM API
2. 无 LLM 时 `skip_verification=True`：应改为即使无 LLM 也做部分验证
3. WL kernel 对大图逐对计算可能很慢，需加采样

## 注意事项

- `local_settings.py` 包含 API 密钥，已被 .gitignore 忽略，模板见 `local_settings_example.py`
- `*.pkl`, `*.pth` 模型文件已被 .gitignore 忽略
- 项目注释使用中文
- 多用户通过 `GLOBAL_ID` 共存（如 train_all_gw.py, train_all_wyb.py 等变体）
