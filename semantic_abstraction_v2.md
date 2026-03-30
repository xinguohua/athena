# 快照级语义抽象

在识别出异常快照后，我们通过关键路径提取、双侧语义提升和语义匹配三个阶段，将快照映射到 ATT&CK 技术标签。

## 关键路径提取（不变）

在检测到异常快照后，我们从对应的溯源有向图中提取因果路径以表征快照级行为。我们枚举图中所有从入度为零的源节点到出度为零的汇节点的因果路径，经去重与合并得到快照级路径集合，然后按照两个互补指标对路径进行优先级排序。

**路径桥接性。** 在隐蔽攻击中，攻击进程频繁地与大量良性操作交织，导致其在溯源图中的度显著增大。经过此类高度节点的路径更可能穿越攻击相关实体。我们通过路径上节点度的均值来量化路径穿越高度节点的倾向：

$$f_{\text{bridge}}(P) = \frac{1}{|P|} \sum_{v_i \in P} \deg(v_i)$$

**路径稀有性。** 异常快照中的因果路径分为两类：在周期性系统任务中反复出现的常规路径，以及仅在特定上下文中出现的稀有路径。后者更可能携带与异常事件直接相关的行为。路径稀有性定义为路径在历史良性行为语料中出现频率的倒数：

$$f_{\text{rarity}}(P) = \frac{1}{\text{Freq}(P)}$$

我们以等权重组合桥接性与稀有性计算每条路径的优先级得分，选取排名前 $m$ 的路径作为关键路径。

---

## 语义提升

将关键路径映射到 ATT&CK 技术需要弥合两个语义层次之间的鸿沟。以进程注入攻击为例，日志侧记录的是 `EVENT_WRITE memhelp.so`、`EVENT_SENDTO` 等系统调用，而 ATT&CK 对应技术 T1055 的描述为 "Adversaries may inject malicious code into processes"。前者是操作级标识符，后者是意图级自然语言，二者之间不存在词汇交集，嵌入模型无法建立有效关联。然而，系统调用和攻击意图都可以用"做了什么操作、操作了什么对象"这一统一形式来表达。例如，`EVENT_WRITE memhelp.so` 可以描述为 "process writes shared library"；而 "inject code into processes" 的实现过程同样可以分解为 "process writes shared library, process modifies another process memory"。我们将这种以动作和对象为核心的自然语言表述称为*系统事件描述*，并据此提出*双侧语义提升*：日志侧将系统调用向上提升为系统事件描述，技术侧将攻击意图向下分解为系统事件描述，两侧通过统一的词汇表在同一语义空间汇合。

**日志侧语义提升。** 溯源图中一条边记录的是三元组 $\langle$主体进程, 事件类型, 客体实体$\rangle$，其中主体和客体均以系统内部标识符表示（进程名 `bash`、文件路径 `/tmp/memhelp.so`、套接字 `192.168.1.1:80`），事件类型为粗粒度的系统调用类别（EVENT_READ、EVENT_WRITE、EVENT_EXECUTE 等）。日志侧提升的目标是将主体和客体标识符翻译为 ATT&CK 共享词汇，同时保留事件类型的原始操作语义（reads、writes、executes），使提升后的描述忠实反映日志中可直接观察到的系统行为，而不引入日志无法支撑的语义推断。

我们对 691 项 ATT&CK Enterprise 技术描述进行词频统计，将全部词汇按语义角色归入四类（进程、文件、网络/套接字、动作），并进一步区分*系统级*词汇（可从日志直接观察或推导，如 "process"、"shared library"、"credential file"）与*意图级*词汇（仅描述攻击目的，如 "adversary"、"evade"、"persistence"）。统计显示，ATT&CK 描述中约 44% 为系统级词汇，涵盖 2 424 个进程角色词、1 227 个文件类型词和 1 695 个网络实体词。这一系统级子集构成双侧对齐的共享词表：日志侧将实体标识符向上翻译至该词表，技术侧将攻击意图向下分解至该词表。

我们构建主客体映射规则表，将主体进程名翻译为功能角色，将客体文件路径和网络地址翻译为系统级类型。映射的合理性基于操作系统的标准化约定：文件扩展名与类型之间存在确定性对应，目录结构遵循 FHS 等规范，知名端口号由 IANA 统一分配，IP 地址依据 RFC 1918 划分内外网。这些映射均基于公开标准，具有确定性和可验证性。未命中规则表的进程名保持原样，因其本身已是嵌入模型可理解的自然语言标识。

| 类别 | 日志原始标识符 | 系统级类型 |
|------|--------------|-----------|
| **进程** | bash / sh / zsh | command shell |
| | python / perl / ruby | scripting interpreter |
| | sshd / telnetd | SSH service |
| | crond / at | task scheduler |
| | apache / nginx | web server |
| | mysqld / postgres | database service |
| | curl / wget | network utility |
| | systemd / init | system service manager |
| **文件** | .so / .dll / .dylib | shared library |
| | .conf / .cfg / .ini | configuration file |
| | /etc/shadow / /etc/passwd / SAM | credential file |
| | /var/log/* / *.evtx / *.log | log file |
| | /proc/* | process information |
| | crontab / systemd unit / /etc/init.d/* | scheduled task configuration |
| | .pem / .key / authorized_keys | authentication key file |
| | .exe / ELF binary | executable |
| **网络/套接字** | :80 / :443 | HTTP / HTTPS |
| | :22 | SSH |
| | :53 | DNS |
| | :25 / :587 | SMTP |
| | RFC 1918 内网IP | internal connection |
| | 其他IP | external connection |

经过主体和客体提升后，事件类型直接翻译为对应的基本操作动词（EVENT_READ $\to$ reads，EVENT_WRITE $\to$ writes，EVENT_EXECUTE $\to$ executes，EVENT_SENDTO $\to$ sends，EVENT_RECVFROM $\to$ receives），不做进一步的语义推断。例如，溯源图中 $\langle$`bash`, EVENT_WRITE, `/tmp/memhelp.so`$\rangle$ 被翻译为 "command shell writes shared library"。日志侧的动作词汇保持在操作层面（reads、writes、executes），与 ATT&CK 意图层面的动词（inject、load、exfiltrate）之间的鸿沟留给技术侧提升来弥合。提升后的事件按因果顺序拼接，形成路径级行为描述。

**技术侧语义提升。** 日志侧提升后的描述使用操作动词（writes、reads、sends），而 ATT&CK 使用意图动词（inject、exfiltrate、persist）。二者描述同一行为却处于不同抽象层：日志中不存在 "inject" 事件，实际发生的是一系列 writes 和 reads。技术侧提升的目标是将意图动词展开为操作动词序列，使技术描述降落至与日志侧相同的操作语义层。

设技术 $i$ 的原始描述为 $D_i$，翻译后的操作级描述为 $s_i = \mathcal{T}(D_i)$。流水线 $\mathcal{T}$ 依次完成结构化解析、语义层转换和操作级实例化。

**结构化解析。** 对 $D_i$ 断句后，使用依赖句法分析从每个句子中抽取（主语, 谓语动词, 宾语）三元组。例如 "Adversaries may inject malicious code into processes" 产生 (adversaries, inject, code into processes)。

**语义层转换。** 对每个三元组的主谓宾分别执行转换，转换规则由以下映射表统一定义：

| 类别 | 意图级（ATT&CK原始） | 系统级（转换后） |
|------|---------------------|----------------|
| **进程** | adversaries / threat actors | process |
| | legitimate users | user |
| | victim | process |
| **动作** | inject | writes, reads |
| | exfiltrate | reads, sends |
| | persist / establish | writes |
| | dump | reads, writes |
| | escalate | reads, executes |
| | steal | reads, sends |
| | enumerate / discover | reads |
| | encrypt | reads, writes |
| **文件** | malicious code / arbitrary code | shared library / executable |
| | credentials / passwords | credential file |
| | data / collected data | file |
| | registry | configuration file |
| | scheduled task | scheduled task configuration |
| **网络/套接字** | C2 channel / command and control | network connection |
| | exfiltration channel | network data |
| | remote services | remote connection |
| | phishing / spearphishing | email |

主语命中进程类映射时执行替换；谓语若命中动作类映射则保留并展开，未命中则丢弃整个三元组——这自然过滤掉背景句和检测建议句；宾语剥离意图修饰词（malicious, stolen）后命中文件类或网络类映射时执行替换。

**操作级实例化。** 语义层转换中一个意图动词映射为多个操作动词，因此一个三元组展开为多个操作级三元组。例如 (adversaries, inject, code into processes) 经转换与展开后生成：

- (process, writes, shared library)
- (process, writes, process memory)
- (process, reads, process information)

全部三元组去重拼接为技术 $i$ 的操作级描述 $s_i$。我们对 691 项 ATT&CK Enterprise 技术离线执行 $\mathcal{T}$，预构建描述库 $\{s_i\}_{i \in I}$，检测时直接加载。对于子技术，将父技术的翻译合并以确保行为覆盖完整。

---

## 语义匹配

经过双侧提升后，日志侧行为描述与技术侧描述均处于共享的系统事件自然语言空间，可通过句子嵌入模型直接比较。设 $d$ 为日志侧提升后的行为描述，$s_i$ 为技术 $i$ 的翻译后描述。我们使用 Sentence-BERT 对两者进行编码，并计算余弦相似度：

$$t_e = \arg\max_{i \in I} \; S\bigl(\mathcal{V}(d),\; \mathcal{V}(s_i)\bigr)$$

其中 $I$ 为 ATT&CK 技术集合，$\mathcal{V}(\cdot)$ 表示 Sentence-BERT 编码器，$S(\cdot, \cdot)$ 计算余弦相似度。当 $\max_{i} S(\mathcal{V}(d), \mathcal{V}(s_i)) < \gamma$ 时，该快照标记为*未匹配*。

**路径级上下文的作用。** 将单个提升后的事件直接匹配 ATT&CK 技术存在固有歧义。孤立事件 *(process, reads, credential file)* 可能对应 T1003（凭据转储）、T1552（不安全的凭据存储）或常规身份认证操作，仅凭单事件无法消歧。路径级序列上下文通过三种机制解决这一问题。其一，歧义消解：当该事件后跟随 *(process, writes, temporary file) $\to$ (process, sends network data)*，完整序列编码了"读取凭据→暂存→外传"的行为链，明确指向凭据转储。其二，容错能力：即使路径中某个事件未能完全提升，其余事件仍可提供足够上下文支撑正确匹配。其三，更丰富的语义信号：路径级描述聚合多个事件的语义信息，使嵌入模型能够捕捉单事件无法表达的组合行为模式。

---

## 攻击序列对齐

（与前一版本相同，此处省略。）

---
