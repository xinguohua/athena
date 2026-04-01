# 快照级语义抽象

在识别出异常快照后，我们通过关键路径提取、双侧语义提升和语义匹配三个阶段，将快照映射到 ATT&CK 技术标签。

## 关键路径提取

在检测到异常快照后，我们从对应的溯源有向图中提取因果路径以表征快照级行为。我们枚举图中所有从入度为零的源节点到出度为零的汇节点的因果路径，经去重与合并得到快照级路径集合，然后按照两个互补指标对路径进行优先级排序。

**路径桥接性。** 在隐蔽攻击中，攻击进程频繁地与大量良性操作交织，导致其在溯源图中的度显著增大。经过此类高度节点的路径更可能穿越攻击相关实体。我们通过路径上节点度的均值来量化路径穿越高度节点的倾向：

$$f_{\text{bridge}}(P) = \frac{1}{|P|} \sum_{v_i \in P} \deg(v_i)$$

**路径稀有性。** 异常快照中的因果路径分为两类：在周期性系统任务中反复出现的常规路径，以及仅在特定上下文中出现的稀有路径。后者更可能携带与异常事件直接相关的行为。路径稀有性定义为路径在历史良性行为语料中出现频率的倒数：

$$f_{\text{rarity}}(P) = \frac{1}{\text{Freq}(P)}$$

我们以等权重组合桥接性与稀有性计算每条路径的优先级得分，选取排名前 $m$ 的路径作为关键路径。

---

## 语义提升

将关键路径映射到 ATT&CK 技术需要弥合两个语义层次之间的鸿沟。日志侧记录的是操作级标识符（如 `EVENT_WRITE memhelp.so`），ATT&CK 技术描述的是意图级自然语言（如 "Adversaries may inject malicious code into processes"）。二者之间不存在词汇交集，嵌入模型无法直接关联。我们提出*双侧语义提升*：日志侧将系统调用向上翻译为自然语言描述，技术侧将攻击意图向下分解为结构化三元组，两侧在句子嵌入空间中汇合。

### 日志侧翻译词表

溯源图中的原始标识符（进程名 `bash`、文件路径 `/tmp/memhelp.so`、套接字 `192.168.1.1:80`）对嵌入模型而言信息密度低且缺乏语义。我们构建一套翻译词表，将这些标识符翻译为嵌入模型可理解的自然语言类型标签。

类型标签的划分通过数据驱动推导：我们对 691 项 ATT&CK 技术的 1,737 个三元组宾语用 Sentence-BERT 编码为语义向量，然后进行 Ward 层次聚类。聚类结果显示，ATT&CK 宾语自然分离为若干语义簇：凭据类（user credentials, passwords, hashes）、动态库类（DLLs, modules）、证书类（SSL/TLS certificates）、配置类（Registry, config）、命令/执行类（arbitrary commands, scripts）、远程访问类（remote IPC, remote systems）、网络类（network traffic, DoS）、邮件类（phishing, spam）等。每个簇对应一种类型标签。据此，进程名按功能角色翻译为 3 类，文件路径按内容类型翻译为 6 类，网络地址翻译为 2 类，共 12 类（含兜底类型）。该推导过程完全可复现（`derive_type_labels.py`）。

### 日志侧提升

日志侧提升将溯源图中的系统内部标识符翻译为 $\mathcal{Y}$ 中的系统级类型，同时**保留原始标识符**以提供区分信号。一条溯源边 $\langle$主体进程, 事件类型, 客体实体$\rangle$ 的翻译分三步：

**主体翻译。** 将进程名映射为进程类型。通过对 1,765 个三元组的主语进行分类，识别出四类在日志中可通过进程名区分的角色：命令行解释器（bash、cmd）、脚本引擎（PowerShell、Python）、远程访问服务（sshd）、系统代理执行工具（rundll32、regsvr32、msiexec 等 LOLBins）。其余进程名保持原样——进程名本身已是嵌入模型可理解的自然语言标识。

**客体翻译。** 将文件路径和网络地址映射为文件或网络类型，**同时保留原始路径/文件名**。映射依据操作系统标准化约定：文件扩展名（.dll → shared library），目录路径（/etc/shadow → credential file），端口号统一映射为 network connection。扩展名匹配优先于路径匹配。例如，`/tmp/memhelp.so` 翻译为 `memhelp.so shared library` 而非仅 `shared library`——保留文件名使嵌入模型能捕获名称中蕴含的语义线索。

| 类别 | 日志原始标识符 | 翻译结果 |
|------|-------------|---------|
| **进程** | bash / sh / zsh / cmd | command shell |
| | python / perl / powershell | scripting interpreter |
| | sshd / telnetd | remote access service |
| | rundll32 / regsvr32 / msiexec / cmstp 等 | proxy executor |
| | 其他进程名 | 保留原名 |
| **文件** | .so / .dll / .dylib | 原名 shared library |
| | /etc/shadow / /etc/passwd / SAM | 原名 credential file |
| | .conf / .cfg / .ini / .plist | 原名 configuration file |
| | .pem / .key / authorized_keys | 原名 authentication key file |
| | .exe / ELF binary | 原名 executable |
| | 其他文件 | 保留原名 |
| **网络** | 任意 IP:端口 | network connection |
| | :25 / :587 | email |

**事件翻译。** 事件类型翻译为操作动词：EVENT_READ → reads，EVENT_WRITE → writes，EVENT_EXECUTE → executes，EVENT_SENDTO → sends，EVENT_RECVFROM → receives。低信息量事件（opens, closes, seeks 等）直接过滤。

三步翻译后，溯源边 $\langle$`bash`, EVENT_WRITE, `/tmp/memhelp.so`$\rangle$ 被翻译为 `command shell writes memhelp.so shared library`。路径上的全部翻译后事件按因果顺序拼接，形成路径级行为描述。

### 技术侧提升

ATT&CK 技术描述为意图级自然语言段落，需要分解为结构化三元组。与日志侧不同，技术侧**不做系统级类型压缩**——保留 ATT&CK 原文中的具体工具名、API 名和攻击细节，使不同技术之间保持区分度。

设技术 $i$ 的描述为 $D_i$，流水线 $\mathcal{T}$ 将其转换为三元组集合。

**结构化解析。** 使用 spaCy 依存分析从 $D_i$ 的每个句子中抽取候选 $\langle$主语, 谓语, 宾语$\rangle$ 三元组。对无主句或系动词句直接丢弃，对代词主语回溯替换为前句中的名词性先行词。自动提取后进行人工校验，检查三项：(1) 攻击行为性——三元组须描述攻击者的主动行为；(2) 主语具体性——主语须为明确实体；(3) 工具覆盖性——原文提及的具体工具、命令、API 须在三元组中保留。

**谓语规范化。** 通用包装动词（use、abuse、leverage 等）不携带操作语义，在结构化解析阶段即被替换为谓语短语中内嵌的真正动作动词（如 "use tools to dump" → dump，"abuse BITS to establish" → establish）。

最终，每个技术 $i$ 的三元组集合拼接为描述 $s_i$。例如，T1055.001（DLL Injection）的描述为：

> Adversaries inject dynamic-link libraries (DLLs) into processes. Adversaries write native Windows API calls such as VirtualAllocEx and WriteProcessMemory, then invoke with CreateRemoteThread.

描述保留了 "DLLs"、"VirtualAllocEx"、"WriteProcessMemory" 等具体细节，使 T1055.001 与其他涉及共享库的技术（如 T1574 DLL Hijacking）在嵌入空间中保持可区分。我们对全部 691 项 ATT&CK Enterprise 技术离线执行 $\mathcal{T}$，预构建描述库 $\{s_i\}_{i \in I}$，检测时直接加载。

---

## 语义匹配

经过双侧提升后，日志侧描述使用系统级类型词汇（如 "command shell writes memhelp.so shared library"），技术侧描述使用 ATT&CK 自然语言（如 "Adversaries inject DLLs into processes"）。二者虽然词汇不完全相同，但描述的是同一类系统行为，句子嵌入模型能够捕获这种跨词汇的语义关联——"shared library" 与 "DLLs"、"writes" 与 "inject" 在嵌入空间中语义相近。

我们使用 Sentence-BERT 对日志侧描述 $d$ 和技术库中每个描述 $s_i$ 进行编码，通过余弦相似度检索最匹配的技术：

$$t_e = \arg\max_{i \in I} \; S\bigl(\mathcal{V}(d),\; \mathcal{V}(s_i)\bigr)$$

其中 $\mathcal{V}(\cdot)$ 为 Sentence-BERT 编码器，$S(\cdot, \cdot)$ 为余弦相似度。当 $\max_{i} S < \gamma$ 时，该快照标记为*未匹配*。

**日志侧保留原始标识符的作用。** 日志侧提升并非仅输出系统级类型（如 "process writes shared library"），而是同时保留原始文件名（如 "process writes memhelp.so shared library"）。实验表明，这一设计对匹配准确率至关重要。纯类型描述将大量不同攻击压缩为相同的词汇组合（如 "process writes file" 同时对应文件删除、数据篡改、载荷投递等多种技术），导致区分度丧失。保留原始文件名为嵌入模型提供了额外的语义线索——文件名中蕴含的语义（如 "memhelp" 暗示 memory helper，"injectLog" 暗示 injection logging）使模型能在嵌入空间中将查询推向正确的技术簇。

**路径级上下文的作用。** 将单个事件直接匹配 ATT&CK 技术存在固有歧义。孤立事件 "process reads credential file" 可能对应凭据转储（T1003）、不安全凭据存储（T1552）或常规身份认证。路径级序列上下文通过三种机制解决这一问题：(1) *歧义消解*——当该事件后跟随 "process writes file → process sends network connection"，完整序列编码了"读取凭据→暂存→外传"的行为链，明确指向凭据转储；(2) *容错能力*——即使路径中某个事件未能完全提升，其余事件仍可提供足够上下文支撑正确匹配；(3) *语义信号聚合*——路径级描述聚合多个事件的语义信息，使嵌入模型能够捕捉单事件无法表达的组合行为模式。

---

## 攻击序列对齐

（与前一版本相同，此处省略。）

---
