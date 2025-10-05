import os.path
import pandas as pd
import igraph as ig
import re
from igraph import Graph
from .base import BaseProcessor
from .common import merge_properties, collect_dot_paths, collect_atlas_label_paths
from .type_enum import ObjectType

class ATLASHandler(BaseProcessor):
    def __init__(self, base_path=None, train=True ):
        super().__init__(base_path, train)

        # 用于按图（场景）分开存储其对应的恶意标签
        self.graph_to_label = {}
        self.all_netobj2pro = {}
        self.all_subject2pro = {}
        self.all_file2pro = {}
        self.total_loaded_bytes = 0


    def load(self):
        """加载 ATLAS 数据集。
        - 训练模式：去掉恶意标签部分
        - 测试模式：保留所有数据
        """
        print("处理 ATLAS 数据集...")
        graph_files = collect_dot_paths(self.base_path)  # 所有 .dot 文件路径
        label_map = collect_atlas_label_paths(self.base_path)  # 标签映射

        # 清空缓存
        self.all_labels.clear()
        self.graph_to_label.clear()

        def filter_bad_edges(df, labels):
            """过滤恶意边"""
            if not labels:
                return df, len(df), len(df)
            bad = set(labels)
            before = len(df)
            mask_bad = df.isin(bad).any(axis=1)
            df_clean = df.loc[~mask_bad]
            after = len(df_clean)
            return df_clean, before, after

        malicious_name = "M1-CVE-2015-5122_windows_h1"
        benign_name = "M1-CVE-2015-5122_windows_h2"

        for dot_file in graph_files:
            dot_name = os.path.splitext(os.path.basename(dot_file))[0]
            if dot_name not in [malicious_name, benign_name]:
                continue  # 跳过无关文件
            print(f"正在加载场景: {dot_name}")

            # 加载标签
            if dot_name in label_map:
                with open(label_map[dot_name], 'r', encoding='utf-8') as label_file:
                    graph_labels = [line.strip() for line in label_file if line.strip()]
                    self.graph_to_label[dot_name] = graph_labels
            else:
                if not self.train:
                    print(f"  - 警告: 未找到场景 '{dot_name}' 的标签文件。")


            # 解析 .dot 文件 -> DataFrame
            netobj2pro, subject2pro, file2pro, dns, ips, conns, sess, webs = collect_nodes_from_log(dot_file)
            dot_df= collect_edges_from_log(dot_file, dns, ips, conns, sess, webs, subject2pro, file2pro)

            if dot_name == benign_name:
                df_begin, before, after = filter_bad_edges(dot_df, self.graph_to_label[dot_name])
                print(f"  - 良性图全部: {len(df_begin)} 条边")
                self.begin = df_begin
            elif dot_name == malicious_name:
                print(f"  - 恶意图全部: {len(dot_df)} 条边")
                self.malicious = dot_df
                self.all_labels.extend(self.graph_to_label[dot_name])

            merge_properties(netobj2pro, self.all_netobj2pro)
            merge_properties(subject2pro, self.all_subject2pro)
            merge_properties(file2pro, self.all_file2pro)

        # 去重标签
        self.all_labels = list(set(self.all_labels))
        if not self.train:
            print(f"共找到 {len(self.all_labels)} 个唯一恶意标签: {self.all_labels}")

    def create_snapshots_from_graph(self, df, is_malicious):
        snapshots = []
        if df is None or len(df) == 0:
            return []

        node_frequency = {}
        node_timestamps = {}
        cache_graph = ig.Graph(directed=True)
        first_flag = True
        snapshot_size = 300
        forgetting_rate = 0.3
        # --- 排序 ---
        sorted_df = df.sort_values(by='timestamp') if 'timestamp' in df.columns else df

        # --- 遍历每条边 ---
        for _, row in sorted_df.iterrows():
            actor_id, object_id = row["actorID"], row["objectID"]
            action = row["action"]
            timestamp = row.get("timestamp", 0)

            # 频率统计
            node_frequency[actor_id] = node_frequency.get(actor_id, 0) + 1
            node_frequency[object_id] = node_frequency.get(object_id, 0) + 1

            # === 加点 ===
            try:
                v_actor = cache_graph.vs.find(name=actor_id)
                v_actor["frequency"] = node_frequency[actor_id]
            except ValueError:
                actor_type_enum = ObjectType[row['actor_type']]
                cache_graph.add_vertex(
                    name=actor_id, type= actor_type_enum.name,
                    properties=extract_properties(actor_id, self.all_netobj2pro, self.all_subject2pro,
                                                  self.all_file2pro),
                    label = int(any(lbl in actor_id for lbl in self.all_labels)),
                    frequency= node_frequency[actor_id]
                )
            node_timestamps[actor_id] = timestamp

            try:
                v_object = cache_graph.vs.find(name=object_id)
                v_object["frequency"] = node_frequency[object_id]
            except ValueError:
                object_type_enum = ObjectType[row['object']]
                cache_graph.add_vertex(
                    name=object_id, type= object_type_enum.name,
                    properties=extract_properties(object_id, self.all_netobj2pro, self.all_subject2pro,
                                                  self.all_file2pro),
                    label = int(any(lbl in object_id for lbl in self.all_labels)),
                    frequency= node_frequency[object_id]
                )
            node_timestamps[object_id] = timestamp

            # === 加边 ===
            a_idx = cache_graph.vs.find(name=actor_id).index
            o_idx = cache_graph.vs.find(name=object_id).index
            if not cache_graph.are_connected(a_idx, o_idx):
                cache_graph.add_edge(a_idx, o_idx, actions=action, timestamp=timestamp)

            # --- 快照生成逻辑 ---
            n_nodes = len(cache_graph.vs)
            if first_flag and n_nodes >= snapshot_size:
                self._generate_snapshot(cache_graph, snapshots)
                first_flag = False
            elif not first_flag and n_nodes >= snapshot_size * (1 + forgetting_rate):
                self._retire_old_nodes(snapshot_size, forgetting_rate, node_timestamps, cache_graph)
                self._generate_snapshot(cache_graph, snapshots)

        # 收尾
        if len(cache_graph.vs) > 0:
            self._generate_snapshot(cache_graph, snapshots)
        return snapshots



    def _retire_old_nodes(self, snapshot_size: int, forgetting_rate: float, node_timestamps: dict, cache_graph: Graph) -> None:
        """这个函数保持不变"""
        n_nodes_to_remove = int(snapshot_size * forgetting_rate)
        if n_nodes_to_remove <= 0: return
        sorted_nodes = sorted(node_timestamps.items(), key=lambda item: item[1])
        nodes_to_remove = [node_id for node_id, _ in sorted_nodes[:n_nodes_to_remove]]
        try:
            indices_to_remove = [cache_graph.vs.find(name=name).index for name in nodes_to_remove]
            cache_graph.delete_vertices(indices_to_remove)
        except ValueError:
            pass # 节点可能已被删除
        for node_id in nodes_to_remove:
            if node_id in node_timestamps:
                del node_timestamps[node_id]

    def _generate_snapshot(self, cache_graph, snapshots) -> None:
        snapshot = cache_graph.copy()
        snapshots.append(snapshot)






def collect_nodes_from_log(paths):  # dot文件的路径
    # 创建字典
    netobj2pro = {}
    subject2pro = {}
    file2pro = {}
    domain_name_set = {}
    ip_set = {}
    connection_set = {}
    session_set = {}
    web_object_set = {}
    nodes = []

    # 读取整个文件
    with open(paths, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按分号分隔，处理每个段落
    statements = content.split(';')

    # 正则表达式匹配节点定义
    node_pattern = re.compile(r'^\s*"?(.+?)"?\s*\[.*?type="?([^",\]]+)"?', re.IGNORECASE)

    for stmt in statements:
        if 'capacity=' in stmt:
            continue  # 跳过包含 capacity 字段的段落
        match = node_pattern.search(stmt)
        if match:
            node_name = match.group(1)
            node_typen = match.group(2)
            nodes.append((node_name, node_typen))
    for node_name, node_typen in nodes:  # 遍历所有的节点
        node_id = node_name  # 节点id赋值
        node_type = node_typen  # 赋值type属性
        # -- 网络流节点 --
        if node_type == 'domain_name':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            domain_name_set[node_id] = nodeproperty
        if node_type == 'IP_Address':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            ip_set[node_id] = nodeproperty
        if node_type == 'connection':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            connection_set[node_id] = nodeproperty
        if node_type == 'session':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            session_set[node_id] = nodeproperty
        if node_type == 'web_object':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            web_object_set[node_id] = nodeproperty
        # -- 进程节点 --
        elif node_type == 'process':
            nodeproperty = node_id
            subject2pro[node_id] = nodeproperty
        # -- 文件节点 --
        elif node_type == 'file':
            nodeproperty = node_id
            file2pro[node_id] = nodeproperty

    return netobj2pro, subject2pro, file2pro, domain_name_set, ip_set, connection_set, session_set, web_object_set


def collect_edges_from_log(paths, domain_name_set, ip_set, connection_set, session_set, web_object_set, subject2pro,
                           file2pro) -> pd.DataFrame:
    """
    从 DOT-like 日志文件中提取含 capacity 的边，并识别 source/target 属于哪个节点集合。
    返回一个包含 source、target、type、timestamp、source_type、target_type 的 DataFrame。
    """
    # 预定义的节点集合

    edges = []

    with open(paths, "r", encoding="utf-8") as f:
        content = f.read()

    statements = content.split(";")

    edge_pattern = re.compile(
        r'"?([^"]+)"?\s*->\s*"?(.*?)"?\s*\['
        r'.*?capacity=.*?'
        r'type="?([^",\]]+)"?.*?'
        r'timestamp=(\d+)',
        re.IGNORECASE | re.DOTALL
    )

    for stmt in statements:
        if "capacity=" not in stmt:
            continue
        m = edge_pattern.search(stmt)
        if m:
            source, target, edge_type, ts = (x.strip() for x in m.groups())

            # 判断 source/target 所属集合
            if source in domain_name_set:
                source_type = "NETFLOW_OBJECT"
            elif source in ip_set:
                source_type = "NETFLOW_OBJECT"
            elif source in connection_set:
                source_type = "NETFLOW_OBJECT"
            elif source in session_set:
                source_type = "NETFLOW_OBJECT"
            elif source in web_object_set:
                source_type = "NetFlowObject"
            elif source in subject2pro:
                source_type = "SUBJECT_PROCESS"
            elif source in file2pro:
                source_type = "FILE_OBJECT_BLOCK"
            else:
                source_type = "PRINCIPAL_LOCAL"

            if target in domain_name_set:
                target_type = "NETFLOW_OBJECT"
            elif target in ip_set:
                target_type = "NETFLOW_OBJECT"
            elif target in connection_set:
                target_type = "NETFLOW_OBJECT"
            elif target in session_set:
                target_type = "NETFLOW_OBJECT"
            elif target in web_object_set:
                target_type = "NetFlowObject"
            elif target in subject2pro:
                target_type = "SUBJECT_PROCESS"
            elif target in file2pro:
                target_type = "FILE_OBJECT_BLOCK"
            else:
                target_type = "PRINCIPAL_LOCAL"

            edges.append((source, source_type, target, target_type, edge_type, int(ts)))

    return pd.DataFrame(edges, columns=["actorID", "actor_type", "objectID", "object", "action", "timestamp"])



def extract_properties(node_id, netobj2pro, subject2pro, file2pro):
    if node_id in netobj2pro:
        return netobj2pro[node_id]
    elif node_id in file2pro:
        return file2pro[node_id]
    elif node_id in subject2pro:
        return subject2pro[node_id]
    else:
        return node_id