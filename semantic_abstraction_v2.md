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

**系统级类型空间。** 为实现日志侧与技术侧的统一对齐，我们定义系统级类型空间 $\mathcal{Y}$ 作为中间抽象层。其设计遵循三条原则：(1) *双侧可映射*——每个类型必须能被日志侧（通过进程名、文件扩展名、目录路径等）稳定识别，同时能被技术侧（通过 ATT&CK 描述中的对象语义）稳定表达；若某类对象只能被一侧区分，则不单独保留。(2) *攻击语义可区分*——只有当某类型能区分不同攻击行为时才单独保留；例如 credential file 与普通 file 必须区分，因为读取前者对应凭据访问（T1003），读取后者仅为一般文件操作；同理 command shell 与 task scheduler 必须区分，因为前者执行对应命令行执行（T1059），后者执行对应计划任务持久化（T1053）。但进一步区分 /etc/shadow 与 SAM 则不增加语义收益，因为二者对应同一类攻击行为。(3) *最小互斥*——类型不过细（否则两侧无法对齐），也不过粗（否则丢失攻击语义），每个对象只映射到一个类型。

溯源图包含三种节点——进程、文件、网络套接字，我们对每种节点按上述原则细分。进程节点按功能角色细分为 8 类（含兜底类型 process）：command shell（命令行执行，T1059.003/004）、scripting interpreter（脚本执行，T1059.001/005/006）、remote access service（远程登录，T1021）、task scheduler（计划任务，T1053）、web server（Web 应用利用，T1190）、database service（数据库访问，T1213）、system service（服务执行，T1543/T1569）。文件节点按内容类型细分为 7 类：executable、shared library、credential file、configuration file、log file、authentication key file，以及兜底类型 file。网络节点不做协议级细分——技术侧大多以 network、C2、traffic 等粗粒度描述，端口号本身不携带攻击语义（同样是 :80 连接，是 C2 还是正常浏览取决于行为模式而非端口）——仅区分 network connection 与 email，后者对应独立的钓鱼投递战术。

综合三类节点，$\mathcal{Y}$ 共 17 类：

$$\mathcal{Y} = \underbrace{\text{process, command shell, scripting interpreter, remote access service, task scheduler, web server, database service, system service}}_{\text{进程 8 类}} \cup \underbrace{\text{executable, shared library, credential file, configuration file, log file, authentication key file, file}}_{\text{文件 7 类}} \cup \underbrace{\text{network connection, email}}_{\text{网络 2 类}}$$

其中 process 和 file 分别为进程类和文件类的兜底类型，仅当对象无法归入更具体的子类型时使用。

**日志侧语义提升。** 溯源图中一条边记录的是三元组 $\langle$主体进程, 事件类型, 客体实体$\rangle$，其中主体和客体均以系统内部标识符表示（进程名 `bash`、文件路径 `/tmp/memhelp.so`、套接字 `192.168.1.1:80`），事件类型为粗粒度的系统调用类别。日志侧提升的目标是将主体和客体标识符映射到 $\mathcal{Y}$ 中的系统级类型。映射基于操作系统的标准化约定：文件扩展名与类型之间存在确定性对应（.dll → shared library, .conf → configuration file），目录结构遵循 FHS 等规范（/etc/shadow → credential file, /var/log/ → log file），知名端口号由 IANA 统一分配（:80 → network connection），IP 地址依据 RFC 1918 划分内外网。

| 类别 | 日志侧（原始标识符） | 系统级类型 | 技术侧（ATT&CK 词汇） |
|------|-------------------|-----------|---------------------|
| **进程** | 未命中以下角色的进程名 | process | process（兜底） |
| | bash / sh / zsh / cmd | command shell | command shell, cmd |
| | python / perl / powershell | scripting interpreter | PowerShell, script, VBA |
| | sshd / telnetd | remote access service | SSH, remote service, RDP |
| | crond / at / schtasks | task scheduler | scheduled task, cron |
| | apache / nginx / httpd | web server | web server, web application |
| | mysqld / postgres | database service | database |
| | systemd / init / launchd | system service | service, daemon |
| **文件** | .exe / ELF binary | executable | code, payload, malware, binary, exploit |
| | .so / .dll / .dylib | shared library | DLL, module, dylib, extension |
| | /etc/shadow / /etc/passwd / SAM | credential file | credential, password, hash, token, secret |
| | .conf / .cfg / .ini / .plist | configuration file | registry, config, policy, settings |
| | /var/log/* / *.evtx / *.log | log file | event log, audit, history |
| | .pem / .key / authorized_keys | authentication key file | certificate, key |
| | 其他文件 | file | data, document, file |
| **网络** | 任意 IP:端口 | network connection | C2, command and control, network, traffic |
| | :25 / :587 + 邮件内容 | email | email, phishing, spearphishing |

经过主体和客体提升后，事件类型直接翻译为对应的基本操作动词（EVENT_READ $\to$ reads，EVENT_WRITE $\to$ writes，EVENT_EXECUTE $\to$ executes，EVENT_SENDTO $\to$ sends，EVENT_RECVFROM $\to$ receives），不做进一步的语义推断。例如，溯源图中 $\langle$`bash`, EVENT_WRITE, `/tmp/memhelp.so`$\rangle$ 被翻译为 "command shell writes shared library"。日志侧的动作词汇保持在操作层面（reads、writes、executes），与 ATT&CK 意图层面的动词（inject、load、exfiltrate）之间的鸿沟留给技术侧提升来弥合。提升后的事件按因果顺序拼接，形成路径级行为描述。

**技术侧语义提升。** 日志侧提升后的描述使用操作动词（writes、reads、sends），而 ATT&CK 使用意图动词（inject、exfiltrate、persist）。二者描述同一行为却处于不同抽象层：日志中不存在 "inject" 事件，实际发生的是一系列 writes 和 reads。技术侧提升的目标是将意图动词展开为操作动词序列，使技术描述降落至与日志侧相同的操作语义层。

设技术 $i$ 的原始描述为 $D_i$，翻译后的操作级描述为 $s_i = \mathcal{T}(D_i)$。流水线 $\mathcal{T}$ 依次完成结构化解析、语义层转换和操作级实例化。

**结构化解析。** 对 $D_i$ 完成分句后，使用 spaCy 依存分析抽取候选 $\langle$主语, 谓语, 宾语$\rangle$ 三元组：遍历句中非助动词的动词节点，沿依存弧收集名词性主语（nsubj/nsubjpass）与直接宾语、介宾及属性补语，组合为候选三元组；对无主句或系动词句（通常为背景定义）直接丢弃，对代词主语（"This"、"They"）回溯替换为前句中的名词性先行词。自动提取后进行人工校验，检查三项：(1) 攻击行为性——三元组须描述攻击者的主动行为，而非背景定义或检测建议；(2) 主语具体性——主语须为明确实体（如 "Adversaries"、工具名）；(3) 工具覆盖性——原文提及的具体工具、命令、API 和文件路径须在三元组中保留。不合格者修正后重新纳入，最终的句级三元组集合构成该技术的结构化攻击表示。

**语义层转换。** 对每个三元组的主谓宾分别执行转换，映射规则如下。

*主语映射。* 若主语中包含关键词 adversary、threat actor、threat group、phisher，则映射为 `process`——这些是意图层面的攻击者角色标签，日志中不存在。其余主语（如工具名 Mimikatz、PowerShell，恶意软件 malware、rootkit，系统组件 Registry、DCOM 等）本身已是系统级词汇，直接保留。

*谓语映射。* 将意图动词映射为操作级动词，映射依据是该动词在系统日志中触发什么系统调用：

| 操作级动词 | 意图级动词 |
|-----------|-----------|
| executes | execute, run, invoke, launch, trigger, load, call, interact, bypass, target |
| writes | modify, create, add, install, set, place, register, configure, overwrite, replace, hijack, delete, clear, disable, establish, persist, forge, spoof, deploy, manipulate, alter, hide, obfuscate, patch, poison, embed, implant, remove, rename, craft, tamper, impersonate, conceal, infect, downgrade, store, generate, attach |
| reads | gather, collect, search, enumerate, access, obtain, acquire, query, discover, scan, check, monitor, identify, seek, analyze, find |
| sends | send, communicate, upload, stage, transfer, deliver, redirect, tunnel, flood |
| receives | download, receive |
| reads + writes + executes | inject, exploit, compromise, escalate, force |
| reads + writes | dump, extract, compress, encrypt, encode, copy |
| reads + sends | steal, exfiltrate, harvest |

多映射动词展开为多个操作级三元组，例如 (adversaries, inject, code into processes) 展开为 (process, writes, executable) 和 (process, reads, process)。未命中映射表的攻击准备阶段动词（purchase、develop、build 等）丢弃。

*宾语映射。* 先剥离意图修饰词（malicious、stolen、arbitrary、compromised 等），再将宾语映射为日志侧提升中定义的同一套系统级类型（executable、shared library、credential file、configuration file、network connection 等）。由于同一关键词在不同技术上下文中可能指代不同实体（如 "certificate" 在数据混淆技术中属于网络流量，在凭据访问技术中属于凭据文件），映射需结合技术上下文判断。

全部三元组去重拼接为技术 $i$ 的操作级描述 $s_i$。我们对全部 ATT&CK Enterprise 技术离线执行 $\mathcal{T}$，预构建描述库 $\{s_i\}_{i \in I}$，检测时直接加载。

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
