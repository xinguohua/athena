# Snapshot-level Semantic Abstraction（v2）

对每个异常快照，我们执行三步语义抽象，包括关键路径提取、路径语义提升、技术匹配，将其映射到标准化的 ATT&CK 技术标签。核心思路：日志侧通过可配置的语义提升规则表（configurable semantic lifting mapping）将系统事件翻译为安全领域自然语言，拼接为路径级行为描述；ATT&CK 侧直接使用技术描述原文。两边通过 Sentence-BERT 映射到统一语义空间进行匹配。路径级上下文是消解单事件歧义、实现准确匹配的关键。

## Key Path Extraction（不变）

在检测到异常快照后，我们从对应的溯源有向图中提取因果路径以表征快照级行为。\tool 枚举图中所有由入度为零节点通向出度为零节点的因果路径，并经过去重与合并得到快照级路径集合。随后，我们依据下述指标对路径进行优先级排序。

**Path Bridging.** 在隐蔽攻击中，攻击进程常混入大量良性操作以规避检测，导致其在溯源图中的度显著增大。经过此类高度数节点的路径更可能经过攻击相关实体。我们引入路径桥接性指标，量化路径经过高度数节点的程度。具体地，路径桥接性定义为路径上节点度的均值：

$$f_{\text{bridge}}(P) = \frac{1}{|P|} \sum_{v_i \in P} \deg(v_i)$$

**Path Rarity.** 异常快照中的因果路径分为两类：在周期性系统任务中反复出现的常规路径，以及仅在特定上下文中出现的稀有路径。后者更可能携带与异常事件直接相关的行为。路径稀有性定义为路径在历史良性行为语料中出现频率的倒数：

$$f_{\text{rarity}}(P) = \frac{1}{\text{Freq}(P)}$$

基于路径桥接性和路径稀有性，我们以相同权重计算每条路径的优先级得分，并据此排序，选择排名前 $m$ 的路径作为关键路径。

---

## Path Semantic Lifting and Behavior Description Generation

关键路径由一串因果有序的系统事件三元组构成：$P = [(s_1, op_1, o_1), (s_2, op_2, o_2), \ldots, (s_n, op_n, o_n)]$，其中 $s$ 是主体进程，$op$ 是系统调用，$o$ 是目标实体（文件、网络地址或进程）。这些三元组使用系统特定标识符（进程名、文件路径、系统调用名），而 ATT&CK 技术描述使用安全领域的自然语言。两者之间存在词汇鸿沟，无法直接匹配。

为弥合这一鸿沟，我们对路径中的每个三元组执行语义提升（Semantic Lifting），通过可配置的映射规则表（configurable mapping table）将系统标识符转换为安全领域的自然语言表示，然后将所有提升后的三元组拼接为一段路径级行为描述。

我们提供一套默认映射规则，涵盖 N 条条目，基于操作系统命令文档和安全知识库构建。用户可根据其部署环境扩展或定制该映射表，无需修改系统其他部分。映射规则表以 JSON 配置文件形式管理，支持增量更新。

### Subject Lifting（主体进程 → 自然语言工具描述）

基于进程名和命令行参数，将主体进程映射为可读的工具描述。默认映射规则参考操作系统命令文档构建，以下为示例条目：

| 系统标识符 | 提升后 |
|---|---|
| `bash`, `sh`, `cmd.exe` | shell |
| `python3`, `perl`, `ruby` | script interpreter |
| `curl`, `wget` | download tool |
| `scp`, `sftp`, `ftp` | file transfer tool |
| `cat`, `less`, `more` | file reader |
| `gcc`, `make` | compiler |
| ... | ... |

对于同一进程名可能对应不同功能的情况（如 `python3` 既可以是下载器也可以是扫描器），我们结合命令行参数进行区分。

### Operation Lifting（系统调用 → 语义动词）

将系统调用映射为语义层面的操作动词。同一语义动作在不同操作系统上可能对应不同的系统调用，统一提升后可实现跨平台匹配。示例条目：

| 系统调用 | 提升后 |
|---|---|
| `read`, `pread`, `readv` | read |
| `write`, `pwrite`, `writev` | write |
| `execve`, `execveat` | execute |
| `connect` | connect |
| `fork`, `clone`, `vfork` | create process |
| `unlink`, `rmdir` | delete |
| `chmod`, `fchmod` | change permission |
| `sendto`, `sendmsg` | send |
| `recvfrom`, `recvmsg` | receive |
| ... | ... |

### Object Lifting（目标实体 → 安全语义描述）

将文件路径、网络地址等系统标识符映射为包含安全语义的自然语言描述。映射规则基于路径模式匹配（正则表达式），示例条目：

**文件路径提升：**

| 路径模式 | 提升后 |
|---|---|
| `/etc/shadow`, `/etc/passwd` | credential file |
| `/etc/crontab`, `HKLM\...\Run` | autostart configuration |
| `/proc/[PID]/mem` | process memory |
| `/tmp/*`, `%TEMP%\*` | temporary file |
| `/var/log/*` | log file |
| `/bin/*`, `/usr/bin/*` | system binary |
| `*.so`, `*.dll` | shared library |
| `/home/*/.*` | user configuration file |
| ... | ... |

**网络地址提升：**

| 地址模式 | 提升后 |
|---|---|
| `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` | internal network address |
| 其他 IP | external network address |
| 可反向 DNS 解析的 IP | 对应域名 |

### Lifted 三元组序列生成

对路径中的每个事件三元组分别完成三个槽位的语义提升后，保持三元组结构不变，按因果顺序排列为 lifted 三元组序列 $T_P = [(s'_1, op'_1, o'_1), \ldots, (s'_n, op'_n, o'_n)]$。

**示例：**

原始路径：
```
(bash, fork, python3)
(python3, read, /etc/shadow)
(python3, write, /tmp/creds.txt)
(python3, connect, 10.0.0.5:4444)
(python3, sendto, socket)
```

Lifted 三元组序列：
```
(shell, create process, script interpreter)
(script interpreter, read, credential file)
(script interpreter, write, temporary file)
(script interpreter, connect, external network address)
(script interpreter, send, external network address)
```

与 LLM 生成自由文本行为描述相比，规则表提升具有确定性（同一路径始终产生相同结果）、可审计（每一步提升可追溯到具体规则）、无幻觉风险的优势。对于规则表未覆盖的系统标识符，我们采用 fallback 策略：保留原始标识符作为提升结果（例如未识别的进程名直接保留），由下游 Sentence-BERT 的语义泛化能力处理。

---

## Semantic Matching

语义提升将日志侧的系统事件翻译为安全领域自然语言后，匹配的核心问题是：如何将路径级行为描述与 216 个 ATT&CK 技术进行语义比对。

### 为什么需要路径级上下文：单事件匹配的固有歧义

将单个 lifted 事件直接匹配 ATT&CK 技术存在固有歧义。一条孤立的事件 $(script\ interpreter, read, credential\ file)$ 可能对应多个完全不同的技术：

- **T1003 OS Credential Dumping**：读取凭据文件以提取密码哈希
- **T1552 Unsecured Credentials**：搜索未加密存储的凭据
- **正常系统管理操作**：例如用户认证服务的常规读取

仅凭单事件无法区分上述情况。这一歧义是操作层到意图层映射的根本困难——同一操作在不同上下文中承载不同意图。

### 路径级序列上下文消解歧义

因果路径天然保留了事件之间的时序和因果关系，为每个事件提供了上下文。同一个 $(*, read, credential\ file)$ 事件：

- 当其后跟随 $(*, write, temporary\ file) \to (*, connect, external\ address) \to (*, send, external\ address)$ 时，完整序列编码了 **"读取凭据 → 暂存 → 外传"** 的步骤逻辑，明确指向 T1003（OS Credential Dumping）；
- 当其后跟随 $(*, write, autostart\ configuration)$ 时，序列指向 **持久化** 相关技术；
- 当其前后均为常规系统操作时，更可能是 **良性行为**。

路径级序列上下文提供三方面优势：

**（1）歧义消解。** 如上所述，序列上下文将单事件的多义性收窄为确定的攻击意图。这是路径级匹配相比单事件匹配（如 KnowHow 的逐事件 lifting + 逐事件匹配）最核心的优势。

**（2）容错能力。** 即使路径中某个事件因映射规则未覆盖而未能完全提升（fallback 为原始标识符），路径中其他已成功提升的事件仍可提供足够的上下文支撑正确匹配。单事件匹配则无此容错机制——一旦单事件提升失败，匹配即失败。

**（3）丰富的语义信号。** 路径级行为描述包含多个事件的语义信息，为 Sentence-BERT 提供了更丰富的输入，使其能够捕捉到单事件无法表达的组合语义模式。

**示例：**

Lifted 路径级行为描述（线性化后）：
```
"shell create process script interpreter. script interpreter read credential file.
 script interpreter write temporary file. script interpreter connect external network address.
 script interpreter send external network address."
```

这段描述编码了完整的攻击步骤链，与 ATT&CK T1003 的技术描述（"Adversaries may attempt to dump credentials... Tools such as Mimikatz access LSASS process memory to extract plaintext passwords..."）在语义空间中具有高相似度。

### 日志侧与 ATT&CK 侧的语义对齐

经过语义提升后，日志侧的路径级行为描述已使用安全领域的自然语言词汇（如 "credential file"、"external network address"、"create process"）。ATT&CK 技术描述本身也使用安全领域的自然语言。两者处于同一词汇空间，可直接通过语义嵌入模型进行比对。

**日志侧：** 将 lifted 三元组序列按因果顺序线性化为行为描述文本 $d$。每个三元组的三个槽位以自然语言拼接，三元组之间以句号分隔，保留因果顺序。

**ATT&CK 侧：** 直接使用每个技术 $i$ 的官方描述原文 $s_i$（含子技术描述）。ATT&CK 描述本身已是结构化的安全领域自然语言，包含攻击动作、工具、目标等关键语义元素，无需额外处理即可作为匹配目标。这避免了中间 NLP 提取步骤引入的信息损失和噪声。

两边通过 Sentence-BERT 编码为向量，以余弦相似度衡量匹配程度：

$$t_e = \arg\max_{i \in I} S(\mathcal{V}(d), \mathcal{V}(s_i))$$

其中 $I$ 为 216 个父技术集合；$d$ 为日志侧 lifted 路径行为描述；$s_i$ 为技术 $i$ 的 ATT&CK 官方描述原文；$\mathcal{V}$ 表示 Sentence-BERT 嵌入模型；$S(\cdot,\cdot)$ 计算余弦相似度。

当 $\max_{i \in I} S(\mathcal{V}(d), \mathcal{V}(s_i)) < \gamma$ 时，该快照标记为 unmatched。

---

## Attack Sequence Alignment（不变）

（同原文，此处省略）

---

