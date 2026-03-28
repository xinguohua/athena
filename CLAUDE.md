# Prographer

基于动态溯源图(Provenance Graph)的异常检测系统，用于网络安全入侵检测。通过分析系统调用日志，构建进程依赖图，使用图嵌入和异常分类检测攻击行为，并映射到 ATT&CK 技术。

## 项目结构

```
process/                      # 主包
├── train_all.py              # 训练入口
├── test_all.py               # 测试/推理入口
├── config.py                 # 参数配置（时间窗口、快照大小等）
├── config.yaml               # 多环境路径配置（local/remote）
├── partition.py              # 图分区 & 社区检测
├── embedding.py              # 嵌入流程
├── technique_semantic_mapper.py  # ATT&CK 技术映射（Chroma + LLM）
├── datahandlers/             # 数据加载（DARPA/ATLAS/OPTC）
├── embedders/                # 图嵌入（GCC/Prographer/ROLAND/TransE/Word2Vec/Unicorn）
├── classfy/                  # 异常检测（TopK/Prographer/Unicorn）
├── llm_clients/              # LLM API 客户端
└── utils/                    # 工具（性能测量等）
```

## 技术栈

- Python 3, PyTorch, igraph, leidenalg, scikit-learn
- PyKEEN (知识图谱嵌入), pandas, numpy, orjson
- LangChain, ChromaDB, sentence-transformers (RAG/语义映射)
- psutil (性能监控)

## 常用命令

### 远程服务器运行（主要方式）

- 服务器: nsas2020@labserver.dack.top:8950
- 项目路径: /home/nsas2020/fuzz/prographer
- Conda 环境: prographer (Python 3.9)

```bash
# 训练
conda activate prographer && cd /home/nsas2020/fuzz/prographer && python -m process.train_all

# 测试
conda activate prographer && cd /home/nsas2020/fuzz/prographer && python -m process.test_all
```

### 开发工作流（本地 ↔ 远程）

代码**不能在本地运行**（缺数据集和依赖），开发流程为：

1. **本地写代码** — Claude 在本地编辑、提交、推送到 GitHub
2. **远程拉取运行** — 用户在远程服务器 `git pull` 后运行
3. **结果回传分析** — 运行产生的输出文件（如 JSON/日志）推到 GitHub，用户本地 `git pull`，Claude 读取分析

Claude 的职责：写代码、分析输出结果、提出改进方案。运行全部由用户在远程服务器操作。

## 关键配置

训练参数在 `process/config.py` 中配置：
- `SNAPSHOT_SIZE`: 500（快照最大节点数）
- `SEQUENCE_LENGTH_L`: 12（快照序列长度）
- `MALICIOUS_WINDOW_MINUTES`: ±10 分钟（恶意事件窗口）
- `DETECTION_THRESHOLD`: 0.01

`train_all.py` 顶部的变量控制训练行为：
- `DATASET_NAME`: "cadets", "atlas", "theia", "trace", "clearscope", "optcday1"
- `EMBEDDER_NAME`: "gcc_dev", "gcc", "prographer", "unicorn", "roland", "transe", "word2vec"
- `CLASSIFY_NAME`: "topk", "prographer", "unicorn"
- `GLOBAL_ID`: 用户标识符（模型文件后缀）

数据集路径在 `process/config.yaml` 中按环境（local/remote）配置。

## 架构模式

- **工厂模式**: `get_handler()`, `get_embedder_by_name()`, `get_classfy()` 创建可插拔组件
- **抽象基类**: `BaseProcessor`, `GraphEmbedderBase`, `BaseClassify` 定义统一接口
- **数据流**: 原始日志 → DataHandler → 图构建 → Partition(快照) → Embedder(嵌入) → Classifier(检测) → 可选 ATT&CK 映射

## 关键文件路径（相对于 process/ 目录）

脚本通过 `python -m process.xxx` 运行，工作目录是项目根目录，但以下文件都在 `process/` 下：

- `process/snapshot_data_{GLOBAL_ID}.pkl` — 训练产生的快照数据
- `process/data/mitreembed_master_Chroma.csv` — ATT&CK 技术向量库源数据
- `process/chroma_db/` — Chroma 向量库持久化目录
- `process/local_settings.py` — API 密钥（gitignore）
- `process/technique_sequences.txt` — 攻击技术序列库

写新脚本时引用这些文件必须用 `os.path.join(os.path.dirname(__file__), ...)` 拼绝对路径，不能用相对路径。

## 注意事项

- `local_settings.py` 包含 API 密钥，已被 .gitignore 忽略，模板见 `local_settings_example.py`
- `*.pkl`, `*.pth` 模型文件已被 .gitignore 忽略
- 项目注释使用中文
- 多用户通过 `GLOBAL_ID` 共存（如 train_all_gw.py, train_all_wyb.py 等变体）