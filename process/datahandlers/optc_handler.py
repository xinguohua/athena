import json
import os
import re
import time
import igraph as ig
import pandas as pd
from .base import BaseProcessor
from .common import collect_json_paths, collect_label_paths
from .common import merge_properties, add_node_properties
from process.partition import detect_communities_with_max
from process.optc_type_enum import optcObjectType
from typing import Optional


class OptcHandler(BaseProcessor):
    """
    OPTC 数据集处理器
    基于 DARPAHandler 完善，支持场景过滤和快照生成
    """
    def __init__(self, base_path, train, *, scene_name: Optional[str] = None):
        """
        参数:
        - base_path: 数据根路径
        - train: 是否训练模式
        - scene_name: 仅加载指定场景 (例如 "0402"), 为 None 则加载全部可用场景
        """
        super().__init__(base_path, train)
        self.scene_name = scene_name
        
        # 用于按图（场景）分开存储其对应的恶意标签
        self.graph_to_label = {}
        self.all_netobj2pro = {}
        self.all_subject2pro = {}
        self.all_file2pro = {}
        self.total_loaded_bytes = 0
        self.all_dfs = []

    def load(self):
        """
        加载 OPTC 数据集
        按 benign/malicious 文件夹分开处理
        """
        # 初始化数据属性
        self.begin = None
        self.malicious = None
        
        json_map = collect_json_paths(self.base_path)
        label_map = collect_label_paths(self.base_path)
        
        # 清空缓存
        self.all_labels.clear()
        
        for scene, category_data in json_map.items():
            # 若配置了 scene_name，则只保留该场景
            if self.scene_name and scene != self.scene_name:
                continue
            # 兼容之前硬编码的逻辑：未显式指定时仍可过滤
            # 如果希望加载全部，请在调用 get_handler 时显式传 scene_name=None
                
            # 处理标签（训练模式）
            if self.train:
                if scene in label_map:
                    label_file = open(label_map[scene])
                    print(f"正在处理: 场景={scene}, label={label_map[scene]}")
                    self.all_labels.extend([
                        line.strip() for line in label_file.read().splitlines() if line.strip()
                    ])
                    
            for category, json_files in category_data.items():
                # 如果是编码器训练模式，加载全部数据
                print(f"正在处理: 场景={scene}, 类别={category}, 文件={json_files}")
                
                # OPTC 特有：需要遍历 JSON 文件来构建 TXT 路径
                category_dfs = []  # 收集当前 category 的所有 df
                for jf in json_files:
                    abs_json_path = os.path.abspath(jf)
                    
                    if not os.path.isfile(abs_json_path):
                        print(f"[WARN] JSON 文件不存在: {abs_json_path}, 跳过")
                        continue
                    
                    self.total_loaded_bytes += os.path.getsize(abs_json_path)
                    
                    # 构建对应的 TXT 文件路径
                    dir_name = os.path.dirname(jf)
                    base_name = os.path.basename(jf)
                    name, _ext = os.path.splitext(base_name)
                    parent_dir = os.path.dirname(os.path.dirname(dir_name))
                    last1 = os.path.basename(os.path.dirname(dir_name))
                    last2 = os.path.basename(dir_name)
                    prefix = f"{last1}_{last2}"
                    txt_path = os.path.join(parent_dir, f"{prefix}_{name}.txt")
                    
                    if not os.path.isfile(txt_path):
                        print(f"[WARN] 找不到对应 TXT 文件: {txt_path}, 跳过")
                        continue
                    
                    # 训练分隔
                    df = _read_optc_txt_as_df(txt_path)
                    df = df.dropna()
                    df.sort_values(by="timestamp", ascending=True, inplace=True)
                    
                    print("==========collect_nodes_from_log_optc=======start")
                    t0 = time.time()
                    netobj2pro, subject2pro, file2pro = collect_nodes_from_log_optc([abs_json_path])
                    t1 = time.time()
                    total_nodes = len(netobj2pro) + len(subject2pro) + len(file2pro)
                    print(f"收集到了 {total_nodes} 个节点")
                    print("==========collect_nodes_from_log_optc=======end")
                    print(f"耗时: {t1 - t0:.2f} 秒")

                    # 形成一个更完整的视图
                    #按 benign/malicious 分开存储
                    if category == "benign":

                        print("==========collect_edges_from_log_optc=======start")
                        t0 = time.time()
                        df = collect_edges_from_log_optc(df, [abs_json_path], True)
                        t1 = time.time()
                        print("==========collect_edges_from_log_optc=======end")
                        print(f"耗时: {t1 - t0:.2f} 秒")
                    elif category == "malicious":
                        print("==========collect_edges_from_log_optc=======start")
                        t0 = time.time()
                        df = collect_edges_from_log_optc(df, [abs_json_path], False)
                        t1 = time.time()
                        print("==========collect_edges_from_log_optc=======end")
                        print(f"耗时: {t1 - t0:.2f} 秒")
                    
                    # 收集到 category_dfs
                    category_dfs.append(df)
                    
                    # 合并到总数据集（用于 use_df）
                    self.all_dfs.append(df)
                    
                    merge_properties(netobj2pro, self.all_netobj2pro)
                    merge_properties(subject2pro, self.all_subject2pro)
                    merge_properties(file2pro, self.all_file2pro)
                
                # 形成一个更完整的视图
                #按 benign/malicious 分开存储
                if category_dfs:
                    merged_df = pd.concat(category_dfs, ignore_index=True).drop_duplicates()
                    if category == "benign":
                        self.begin = merged_df  # 存储到 base.py 定义的属性
                        print(f"  - 良性数据: {len(merged_df)} 条边")
                    elif category == "malicious":
                        self.malicious = merged_df  # 存储到 base.py 定义的属性
                        print(f"  - 恶意数据: {len(merged_df)} 条边")
                
        # 训练用的数据集
        use_df = pd.concat(self.all_dfs, ignore_index=True)
        self.use_df = use_df.drop_duplicates()

    def create_snapshots_from_graph(self, df, is_malicious=False, mode="time"):
        """
        通用快照生成函数
        - mode: "community" 或 "time"
        - is_malicious: 是否恶意数据
        """
        if df is None or len(df) == 0:
            return []

        snapshots = []

        if mode == "community":
            # === 一次性构建全局图 ===
            features, edges, mapp, relations, G = self._build_graph_from_df(df)

            communities = detect_communities_with_max(G)
            name_to_idx = {v["name"]: v.index for v in G.vs}

            for community_id, node_names in communities.items():
                try:
                    node_indices = [name_to_idx[name] for name in node_names if name in name_to_idx]
                    if not node_indices:
                        continue

                    subgraph = G.subgraph(node_indices)
                    self._process_subgraph(subgraph, is_malicious, community_id)
                    snapshots.append(subgraph)
                except Exception as e:
                    print(f"警告：创建快照时出错: {e}")

        elif mode == "time":
            window = pd.Timedelta(minutes=5)
            # window = pd.Timedelta(minutes=2)
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")  # OPTC使用ISO格式字符串，直接转换
            t_min, t_max = df["timestamp_dt"].min(), df["timestamp_dt"].max()
            if pd.isna(t_min) or pd.isna(t_max):
                return []  # 没有有效时间戳，直接返回空
            bins = pd.date_range(start=t_min, end=t_max + window, freq=window)
            
            for i in range(len(bins) - 1):
                part = df[(df["timestamp_dt"] >= bins[i]) & (df["timestamp_dt"] < bins[i + 1])]

                if part.empty:
                    continue

                features, edges, mapp, relations, G = self._build_graph_from_df(part)

                if G.vcount() == 0 or G.ecount() == 0:
                    continue

                self._process_subgraph(G, is_malicious, i)

                snapshots.append(G)

        return snapshots

    def _build_graph_from_df(self, df):
        """给定 DataFrame 构建 igraph.Graph，返回 (features, edges, node_ids, relations, G)"""
        all_labels = set(self.all_labels)
        
        _otype_cache = {}
        
        def _otype(v):
            if v not in _otype_cache:
                _otype_cache[v] = optcObjectType[v].value
            return _otype_cache[v]
        
        nodes_props, nodes_type, edges_map, node_frequency, node_last_ts = {}, {}, {}, {}, {}

        for r in df.itertuples(index=False):
            action = getattr(r, "action")
            actor_id = getattr(r, "actorID")
            object_id = getattr(r, "objectID")
            raw_ts = getattr(r, "timestamp")
            # OPTC使用ISO格式字符串，需要先转换为datetime再转为timestamp
            if hasattr(r, "timestamp_dt") and pd.notna(r.timestamp_dt):
                timestamp = r.timestamp_dt.timestamp()
            else:
                try:
                    timestamp = pd.to_datetime(raw_ts).timestamp()
                except:
                    timestamp = 0.0

            # 频率统计
            node_frequency[actor_id] = node_frequency.get(actor_id, 0) + 1
            node_frequency[object_id] = node_frequency.get(object_id, 0) + 1

            # === 更新时间戳 ===
            node_last_ts[actor_id] = max(timestamp, node_last_ts.get(actor_id, 0))
            node_last_ts[object_id] = max(timestamp, node_last_ts.get(object_id, 0))

            # actor 节点
            props_actor = extract_properties_optc(actor_id, r, action,
                                             self.all_netobj2pro, self.all_subject2pro, self.all_file2pro)
            add_node_properties(nodes_props, actor_id, props_actor)
            if actor_id not in nodes_type:
                nodes_type[actor_id] = _otype(getattr(r, "actor_type"))

            # object 节点
            props_obj = extract_properties_optc(object_id, r, action,
                                           self.all_netobj2pro, self.all_subject2pro, self.all_file2pro)
            add_node_properties(nodes_props, object_id, props_obj)
            if object_id not in nodes_type:
                nodes_type[object_id] = getattr(r, "object")

            # === 累加动作和时间 ===
            edges_map.setdefault((actor_id, object_id), {"actions": set(), "timestamp": []})
            edges_map[(actor_id, object_id)]["actions"].add(action)
            edges_map[(actor_id, object_id)]["timestamp"].append(timestamp)

        # === 创建图节点 ===
        node_ids = list(nodes_props.keys())
        index_map = {nid: i for i, nid in enumerate(node_ids)}

        G = ig.Graph(directed=True)
        G.add_vertices(len(node_ids))
        G.vs["name"] = node_ids
        G.vs["type"] = [nodes_type.get(nid) for nid in node_ids]
        G.vs["properties"] = [str(nodes_props[nid]) for nid in node_ids]
        G.vs["label"] = [1 if nid in all_labels else 0 for nid in node_ids]
        G.vs["frequency"] = [node_frequency.get(nid, 0) for nid in node_ids]
        G.vs["timestamp"] = [node_last_ts.get(nid, 0) for nid in node_ids]

        # === 创建图边 ===
        unique_edges = list(edges_map.keys())
        if unique_edges:
            edge_idx = [(index_map[a], index_map[b]) for (a, b) in unique_edges]
            G.add_edges(edge_idx)
            G.es["actions"] = [
                ",".join(sorted(edges_map[(a, b)]["actions"]))
                if not isinstance(edges_map[(a, b)]["actions"], str)
                else edges_map[(a, b)]["actions"]
                for (a, b) in unique_edges
            ]
            G.es["timestamp"] = [
                max(edges_map[(a, b)]["timestamp"])
                for (a, b) in unique_edges
            ]
        # === 下游需要的结构 ===
        features = [nodes_props[nid] for nid in node_ids]
        edge_index = [[], []]
        relations_index = {}
        for a, b in unique_edges:
            s, d = index_map[a], index_map[b]
            edge_index[0].append(s)
            edge_index[1].append(d)
            relations_index[(s, d)] = list(edges_map[(a, b)])

        return features, edge_index, node_ids, relations_index, G

    def _process_subgraph(self, subgraph, is_malicious=False, cid=None):
        pass
        # if is_malicious:
        #     labels = subgraph.vs["label"] if "label" in subgraph.vs.attributes() else []
        #     mal_nodes = sum(lbl == 1 for lbl in labels)
        #     if mal_nodes > 0:
        #         print(f"社区 {cid} 是恶意社区 (恶意节点数={mal_nodes})")
        #         for v in subgraph.vs:
        #             for attr, old_val in v.attributes().items():
        #                 new_val = _replace_event_in_value(old_val)
        #                 if new_val != old_val:
        #                     print(f"malicious val ===== change {old_val} -> {new_val}")
        #                     v[attr] = new_val


def _read_optc_txt_as_df(txt_path):
    """读取 OPTC TXT 文件为 DataFrame"""
    df = pd.read_csv(txt_path, sep=r"\t| {2,}|\s{1}", engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {
        "Source_ID": "actorID",
        "Source_Type": "actor_type",
        "Destination_ID": "objectID",
        "Destination_Type": "object",
        "Edge_Type": "action",
        "Timestamp": "timestamp"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    for c in ["actorID", "actor_type", "objectID", "object", "action", "timestamp"]:
        df[c] = df[c].astype(str)
    return df[["actorID", "actor_type", "objectID", "object", "action", "timestamp"]]


def iter_json_records(json_path):
    """迭代读取 JSON 记录"""
    with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read().strip()
    if not data:
        return
    try:
        arr = json.loads(data)
        if isinstance(arr, list):
            for obj in arr:
                if isinstance(obj, dict):
                    yield obj
            return
    except:
        pass
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        chunks = re.split(r"}\s*{\s*", line)
        if len(chunks) > 1:
            chunks[0] += "}"
            chunks[-1] = "{" + chunks[-1]
            for c in chunks:
                try:
                    obj = json.loads(c)
                    if isinstance(obj, dict):
                        yield obj
                except:
                    continue
        else:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except:
                continue


def collect_nodes_from_log_optc(paths):
    """收集 OPTC 节点信息"""
    netobj2pro, subject2pro, file2pro = {}, {}, {}
    for p in paths:
        for rec in iter_json_records(p):
            obj_type = str(rec.get("object", "")).upper()
            obj_id = str(rec.get("objectID", ""))
            props = rec.get("properties", {}) or {}

            if obj_type == "FILE":
                file2pro[obj_id] = props.get("file_path", "")
            elif obj_type == "PROCESS":
                node_property = ",".join([
                    props.get("command_line", ""),
                    str(rec.get("tgid", "")),
                    props.get("image_path", "")
                ])
                subject2pro[obj_id] = node_property
            elif obj_type in ["FLOW", "NETFLOW"]:
                node_property = ",".join([
                    props.get("src_ip", ""),
                    props.get("src_port", ""),
                    props.get("dest_ip", ""),
                    props.get("dest_port", "")
                ])
                netobj2pro[obj_id] = node_property
    return netobj2pro, subject2pro, file2pro


def collect_edges_from_log_optc(d, paths, benigin, max_lines=600000):
    """收集 OPTC 边信息"""
    info = []
    for p in paths:
        line_count = 0
        for x in iter_json_records(p):
            if benigin and line_count >= max_lines:
                break
            line_count += 1
            
            action = str(x.get("action", ""))
            actor = str(x.get("actorID", ""))
            obj = str(x.get("objectID", ""))
            ts = str(x.get("timestamp", ""))
            props = x.get("properties", {}) or {}
            cmd = str(props.get("command_line", "") or "")
            path = str(props.get("image_path", "") or "")
            info.append({
                'actorID': actor,
                'objectID': obj,
                'action': action,
                'timestamp': ts,
                'exec': cmd,
                'path': path
            })
    rdf = pd.DataFrame.from_records(info).astype(str)
    return d.merge(rdf, how='inner', on=['actorID', 'objectID', 'action', 'timestamp']).drop_duplicates()


def extract_properties_optc(node_id, row, action, netobj2pro, subject2pro, file2pro):
    """提取 OPTC 节点属性"""
    if node_id in netobj2pro:
        return netobj2pro[node_id]
    elif node_id in file2pro:
        return file2pro[node_id]
    elif node_id in subject2pro:
        return subject2pro[node_id]
    else:
        exec_cmd = getattr(row, "exec", "")
        path_val = getattr(row, "path", "")
        return " ".join([exec_cmd, action] + ([path_val] if path_val else []))


_EVENT_TOKEN = re.compile(r'(?<!\w)EVENT[^\s]*')


def _replace_event_in_value(val):
    """替换事件标记（保留 bug，与 darpa_handler.py 一致）"""
    if isinstance(val, str):
        return _EVENT_TOKEN.sub("chentuoyu", val)
    elif isinstance(val, list):
        return [_replace_event_in_value(x) for x in val]
    elif isinstance(val, tuple):
        return tuple(_replace_event_in_value(x) for x in val)
    elif isinstance(val, dict):
        return {k: _replace_event_in_value(v) for k, v in val.items()}
    elif isinstance(val, set):
        return {_replace_event_in_value(x) for x in val}
    else:
        return val  # 非字符串/容器类型原样返回
