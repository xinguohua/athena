# RQ3: ATT&CK Technique Mapping Accuracy 分析

## 实验设置

- **数据集**: DARPA E3 (cadets314, theia311, trace315, clearscope3.6)
- **评估粒度**: 逐恶意 label（每个 malicious UUID 独立映射）
- **总评估 label 数**: 49
- **Ground truth**: 基于每个 UUID 的实际行为人工标注 ATT&CK parent technique
- **匹配准则**: Hit@K — GT parent technique 出现在 top-K 候选中即为 correct
- **编码模型**: Sentence-BERT (all-MiniLM-L12-v2)
- **置信度阈值**: GAMMA=0.40（低于阈值视为 Unmapped）
- **Full-Enhanced 混合权重**: 操作级三元组(0.5) + 原始描述(0.5)

## 四种方法

| 方法 | 日志侧 | 技术侧 | 说明 |
|------|--------|--------|------|
| Direct | 原始事件字符串 | 原始意图级描述 | 基线 |
| Tech-Enhanced | 原始事件字符串 | 操作级三元组 | 仅技术侧增强 |
| Log-Enhanced | 翻译后自然语言 | 原始意图级描述 | 仅日志侧增强 |
| Full-Enhanced | 翻译后自然语言 | 混合匹配(操作级+原始) | 双侧增强 |

## 主要结果

### Table: Hit@K Accuracy (%) — 49 labels

| Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Unmapped |
|--------|-------|-------|-------|--------|----------|
| Direct | 2.0 | 2.0 | 4.1 | 24.5 | 22.4 |
| Tech-Enhanced | 0.0 | 8.2 | 18.4 | 22.4 | 22.4 |
| Log-Enhanced | **10.2** | **28.6** | **30.6** | **40.8** | 22.4 |
| Full-Enhanced | 2.0 | 20.4 | 26.5 | **40.8** | 22.4 |

*Unmapped: 11/49 labels（2 个空查询 + 9 个置信度 < 0.40），所有方法共享同一 unmapped 集合。*

### 分节点类型

| Node Type | Labels | Method | Hit@5 | Hit@10 |
|-----------|--------|--------|-------|--------|
| NetFlowObject | 13 | Direct | 7.7 | 30.8 |
| | | Tech-Enhanced | 46.2 | 61.5 |
| | | Log-Enhanced | 69.2 | 69.2 |
| | | **Full-Enhanced** | **76.9** | **76.9** |
| SUBJECT_PROCESS | 24 | Direct | 4.2 | 20.8 |
| | | Tech-Enhanced | 8.3 | 8.3 |
| | | Log-Enhanced | 20.8 | **37.5** |
| | | Full-Enhanced | 12.5 | 33.3 |
| FILE_OBJECT | 12 | Direct | 0.0 | 25.0 |
| | | Tech-Enhanced | 8.3 | 8.3 |
| | | Log-Enhanced | 8.3 | **16.7** |
| | | Full-Enhanced | 0.0 | **16.7** |

### 分场景 Hit@10

| Scene | Labels | Direct | Tech-Enh | Log-Enh | Full-Enh | Unmapped |
|-------|--------|--------|----------|---------|----------|----------|
| cadets314 | 8 | 12.5 | 12.5 | 25.0 | 25.0 | 4 |
| theia311 | 23 | 21.7 | 13.0 | 30.4 | **34.8** | 7 |
| trace315 | 13 | 15.4 | 30.8 | **61.5** | 53.8 | 0 |
| clearscope3.6 | 5 | **80.0** | 60.0 | 60.0 | 60.0 | 0 |

## 逐方法分析（从实例出发）

### 1. Direct（无增强基线）——语义鸿沟导致系统性失败

直接将原始审计日志 token（如 `SUBJECT_PROCESS EVENT_SENDTO NetFlowObject`）与 ATT&CK 技术描述做 embedding 匹配。该方法 Hit@1 仅 2.0%，Hit@10 为 24.5%，印证了底层系统调用词汇与意图级 ATT&CK 描述之间存在根本性的**语义鸿沟**。

**失败案例 1（进程行为）。** cadets314 主攻击进程（UUID `4FB0BFEA`，GT: T1059, Execution），原始查询为 `SUBJECT_PROCESS EVENT_EXECUTE FILE_OBJECT_FILE. SUBJECT_PROCESS EVENT_CONNECT NetFlowObject 128.55.12.73,63341,53.158.101.118,80...`，充斥系统调用名和原始 IP 地址。top-1 预测为 T1049（System Network Connections Discovery），T1059 未进入 top-10。原始查询中的 `EVENT_EXECUTE`、`EVENT_CONNECT` 等 token 在 ATT&CK 描述空间中没有对应语义锚点。

**失败案例 2（网络行为）。** theia311 C2 连接 `128.55.12.110→146.153.68.151:80`（GT: T1071），原始查询为 `SUBJECT_PROCESS EVENT_SENDTO NetFlowObject 128.55.12.110,49721,146.153.68.151,80`，预测为 T1568.002（Domain Generation）。原始 IP、端口号等数值 token 对 embedding 模型是纯噪声，与 T1071 描述 "communicate using application layer protocols" 毫无语义交集。

**失败案例 3（文件行为）。** cadets314 数据文件（UUID `0C773AFD`，GT: T1005, Collection），原始查询为 `SUBJECT_PROCESS EVENT_OPEN FILE_OBJECT_FILE. SUBJECT_PROCESS EVENT_CLOSE FILE_OBJECT_FILE. SUBJECT_PROCESS EVENT_LSEEK FILE_OBJECT_FILE...`，top-1 预测为 T1569.001（System Services: Launchctl），纯粹的系统调用序列（OPEN/CLOSE/LSEEK）无法传递 "采集本地数据" 的攻击意图。

### 2. Tech-Enhanced（仅技术侧增强）——三元组去噪显著提升网络行为匹配

将 ATT&CK 技术描述转化为操作级三元组（如 T1071 → "process sends network connection"），去除原始描述中的高层意图词汇（如 "adversaries may", "to evade detection" 等），使技术侧表示向系统层语义靠拢。Hit@5 从 4.1% 提升至 18.4%，**网络行为 NetFlowObject Hit@10 从 30.8% 跃升至 61.5%**。

**成功案例 1（网络流）。** trace315 C2 连接 `128.55.12.118→146.153.68.151:80`（GT: T1071），Direct 预测为 T1049（top-5 全为发现类技术），完全未命中。Tech-Enhanced 下，技术库中 T1090 的三元组为 "process sends network connection. process executes network connection"，与原始查询中 `EVENT_CONNECT NetFlowObject` 的 embedding 距离大幅缩短，rank=3 命中 T1071 parent。同一场景的另外 3 个 C2 连接（→162.66.239.75、→61.130.69.232、→17.146.0.252）也全部从 Direct 失败翻转为 Tech-Enhanced 命中（rank=3~5），说明**三元组格式有效消除了技术描述中的高层语义噪声**。

**成功案例 2（进程行为）。** theia311 `/var/log/mail` 伪装后门进程（UUID `283847BC`，GT: T1071），发起 6862 次 C2 连接。Direct 下原始查询包含大量 `EVENT_MMAP MemoryObject`、`EVENT_MPROTECT` 等内存操作 token，top-1 被拉向 T1070.001（Indicator Removal）。Tech-Enhanced 下，技术库中 T1534 的三元组 "process sends email. process sends network connection. process sends file" 与查询中的网络操作 token 匹配，rank=4 命中 T1071。**三元组将技术描述从意图层（"communicate using protocols"）拉到操作层（"process sends network connection"），即使日志侧未翻译，也缩短了语义距离。**

**局限：描述坍缩。** Hit@1 从 2.0% 降至 0.0%，因为多个技术的三元组完全相同（如 T1071/T1669/T1008 均为 "process sends network connection"），top-1 无法区分。

### 3. Log-Enhanced（仅日志侧增强）——翻译弥合词汇鸿沟

将审计事件翻译为自然语言（如 `EVENT_SENDTO` → "sends network connection"，`EVENT_EXECUTE` → "executes"，`EVENT_UNLINK` → "deletes file"），产生了最大的单侧提升：**Hit@1 达 10.2%，Hit@3 达 28.6%**，较 Direct 提升超 14 倍（Hit@3: 2.0% → 28.6%）。核心在于翻译将日志侧从系统调用词汇转换为自然语言，直接对齐 ATT&CK 的意图级描述空间。

**成功案例 1（翻译消除噪声 token）。** cadets314 主攻击进程（UUID `4FB0BFEA`，GT: T1059），Direct 下原始查询充满 `EVENT_EXECUTE`、`EVENT_CONNECT`、IP 地址等噪声 token。Log-Enhanced 翻译后查询变为 "process executes executable. process sends network connection. process receives network connection. process writes process memory"，去除了全部原始 token 噪声，与 T1059 描述（"execute commands, scripts, or binaries"）语义对齐，rank=2 命中。**对比 Direct 预测的 T1049，翻译后的 "executes executable" 直接锚定了 Execution 类技术。**

**成功案例 2（翻译激活操作语义）。** trace315 子进程（UUID `66C31C14`，GT: T1059），Direct 下原始查询 `SUBJECT_PROCESS EVENT_MMAP MemoryObject. SUBJECT_PROCESS EVENT_MPROTECT MemoryObject. SUBJECT_PROCESS EVENT_WRITE UnnamedPipe...` 被预测为 T1055.011（进程注入），因为 `EVENT_MMAP`/`EVENT_MPROTECT` 在 embedding 空间中与内存注入类技术接近。Log-Enhanced 翻译后查询为 "process writes pipe. process executes executable"，`EVENT_MMAP`/`EVENT_MPROTECT` 等低信息量事件被过滤，保留的 "executes executable" 直接匹配 T1059.003，rank=1 精准命中。

**成功案例 3（翻译恢复删除语义）。** theia311 痕迹清除行为（UUID `CB37BFBA`，GT: T1070），Direct 下原始查询 `SUBJECT_PROCESS EVENT_UNLINK FILE_OBJECT_BLOCK /tmp/memtrace.so...` 被预测为 T1055.008（进程注入），因为 `EVENT_UNLINK` 在原始 embedding 空间中无语义。Log-Enhanced 翻译后查询为 "process deletes file. process sends network connection. command shell executes process"，其中 "deletes file" 与 T1070 描述中的 "remove indicators" 语义对齐，rank=1 命中 T1070.007。

### 4. Full-Enhanced（双侧增强）——解决单侧增强均失败的案例

结合日志侧翻译与混合技术匹配策略（操作级三元组 0.5 + 原始描述 0.5），**Hit@10 与 Log-Enhanced 持平达 40.8%**，但独有 4 个仅 Full-Enhanced 能命中的 label（其他三种方法均失败）。混合匹配的价值在于：日志侧翻译提供自然语言查询，技术侧混合同时保留操作级对齐和意图级消歧。

**独有成功案例 1（C2 连接，三种方法均失败）。** theia311 C2 连接 `128.55.12.110→146.153.68.151:80`（UUID `80370C6E`，GT: T1071）。Direct 预测 T1070.007，Tech-Enhanced 预测 T1090（虽是网络类但 parent 不匹配），Log-Enhanced 预测 T1059.003（执行类，完全偏离）。三者失败的原因各不相同：Direct 受原始 token 噪声干扰；Tech-Enhanced 的三元组坍缩无法区分 T1071 和 T1090；Log-Enhanced 翻译后的 "command shell sends network connection" 在原始技术描述空间中被拉向执行类技术。Full-Enhanced 下，翻译后查询 "command shell sends network connection. process sends network connection" 匹配到混合描述库中 T1011 的操作级三元组 "process sends network connection" + 原始描述 "exfiltrate data over a different protocol"，rank=4 命中 T1071 parent。**操作级三元组提供了网络操作的语义锚定，原始描述提供了 T1071（"application layer protocols"）相对于 T1090 的区分度，两者缺一不可。**

**独有成功案例 2（混合行为进程）。** theia311 `./gtcache` 进程（UUID `F335D6B7`，GT: T1071），行为为 C2 通信 + 访问 /var/log/wdev。Direct/Tech-Enhanced/Log-Enhanced 分别预测为 T1055.012、T1074.001、T1059.003，均聚焦于进程/执行类技术而遗漏 C2 语义。Full-Enhanced 翻译后查询 "process sends network connection. command shell executes process" 在混合描述库中，T1071 的操作级三元组 "process sends network connection" 与原始描述 "communicate using application layer protocols" 叠加得分超过纯执行类技术，rank=8 命中。

**独有成功案例 3（复杂多行为进程）。** trace315 恶意主进程（UUID `1F52B45B`，GT: T1059），行为涵盖加载库、网络通信、大量子进程创建。Full-Enhanced 翻译后查询 "process loads shared library. process sends network connection. process executes process. process executes executable" 在混合库中，T1059 的原始描述 "execute command and script interpreters" 与操作级三元组 "process executes executable" 叠加，在 rank=7 命中（通过 T1559 的 parent T1059 匹配）。Direct/Tech-Enhanced/Log-Enhanced 均失败，因为多行为混合的查询在单侧增强下被不同类型的噪声主导。

### 5. 瓶颈分析：FILE_OBJECT 的意图歧义

在所有方法中，FILE_OBJECT 类 label 始终表现最差（最高仅 16.7% Hit@10）。根因是**意图歧义**：同一翻译查询 "process reads file" 可对应 T1005（Data from Local System, Collection）、T1036（Masquerading, Defense Evasion）或 T1059（Execution），ATT&CK 技术的攻击意图无法从单条审计事件的操作语义中推断。

**典型失败。** cadets314 中 GT=T1005 的文件 label（UUID `0C773AFD`），Direct 下原始查询为 `SUBJECT_PROCESS EVENT_OPEN FILE_OBJECT_FILE. SUBJECT_PROCESS EVENT_CLOSE FILE_OBJECT_FILE...`，Log-Enhanced 翻译后为 "process reads file"，但无论哪种方法，T1005 描述中的 "search local system for files of interest and sensitive data" 与 "process reads file" 的语义距离始终大于 T1055.013 等进程注入类描述。cadets314 的另一个文件 label（UUID `2A03F3BA`，GT: T1074, Data Staged），翻译后查询为 "process writes file"，同样无法与 T1074 描述中的 "stage collected data in a central location" 建立语义关联。

这一局限性正是因果路径级解释模块（§IV-D）的设计动机：通过上下文路径（如"进程先建立 C2 连接，再读取文件并网络外传"）提供单条事件所缺失的意图消歧信号。

### 6. Unmapped 分析：置信度过滤的效果

49 个 label 中有 11 个（22.4%）被标记为 Unmapped，所有方法共享同一 unmapped 集合（置信度阈值 GAMMA=0.40 基于 Log-Enhanced 的 raw mapper 统一计算）。Unmapped 分两类：

**空查询（2 个）。** cadets314 的两个 NetFlowObject label（UUID `937BA111`、`937BE863`），在恶意事件表中无对应边记录，查询文本为空，无法映射。

**低置信度（9 个）。** 翻译后查询的最高匹配分数低于 0.40，说明该行为与所有 ATT&CK 技术均无强语义关联。例如 theia311 中 `/home/admin/profile` 伪装进程（UUID `223838BC`、`ED35A9B7`，GT: T1036, Masquerading），翻译后查询的最高 raw 置信度仅 0.3626。其行为是"写入 /var/log/ 下的文件"，操作语义（"process writes log file"）确实难以匹配到 T1036 描述中的 "masquerade" 意图，低置信度正确反映了语义匹配的不确定性。另一组例子是 theia311 的两个痕迹清除进程（UUID `0836A1B8`、`55387DBE`，GT: T1070），置信度 0.3991——接近阈值但未通过，因为删除单个文件的语义强度不足以超过阈值。

**Unmapped 的分布特征。** 11 个 unmapped 全部集中在 cadets314（4 个）和 theia311（7 个），trace315 和 clearscope3.6 无 unmapped。这与两个场景的攻击特征相关：cadets 和 theia 的部分恶意节点行为较为隐蔽（如伪装进程仅做日志写入），单事件语义弱；而 trace 和 clearscope 的攻击链更显式（大量 C2 通信 + 文件操作组合），查询置信度普遍较高。

## 论文填表建议

| Method | Correct(%) | Incorrect(%) | Unmapped(%) |
|--------|-----------|-------------|-------------|
| Direct | 24.5 | 53.1 | 22.4 |
| Tech-Enhanced | 22.4 | 55.1 | 22.4 |
| Log-Enhanced | **40.8** | 36.7 | 22.4 |
| Full-Enhanced | **40.8** | 36.7 | 22.4 |

*评估标准: Hit@10, 49 malicious labels, DARPA E3, GAMMA=0.40*
