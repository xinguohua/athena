import os
import re
import time
import numpy as np
import igraph as ig
import orjson
import pandas as pd

from .base import BaseProcessor
from .common import collect_json_paths, collect_label_paths
from .common import merge_properties, add_node_properties
from process.partition import detect_communities_with_max


from typing import Optional

class DARPAHandler(BaseProcessor):
    def __init__(self, base_path, train, *, scene_name: Optional[str] = None):
        """DARPA 处理器。

        参数:
        - base_path: 数据根路径
        - train: 是否训练模式
        - scene_name: 仅加载指定场景 (例如 "cadets314"), 为 None 则加载全部可用场景
        """
        super().__init__(base_path, train)
        # 可选场景过滤
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
        加载 DARPA 数据集 - 仿照原逻辑，但按 benign/malicious 文件夹分开处理
        
        参数:
        - load_all_for_encoder: 如果为True，加载全部数据用于编码器训练；如果为False，按原逻辑加载
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
            # 兼容之前硬编码 cadets314 的逻辑：未显式指定时仍可过滤到 cadets314
            # 如果希望加载全部，请在调用 get_handler 时显式传 scene_name=None
                
            # 处理标签（测试模式）
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
                scene_category = f"/{scene}_{category}.txt"
                f = open(self.base_path + scene_category)
                self.total_loaded_bytes += os.path.getsize(self.base_path + scene_category)
                
                # 训练分隔
                data = f.read().split('\n')
                data = [line.split('\t') for line in data]
                df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
                df = df.dropna()
                df.sort_values(by='timestamp', ascending=True, inplace=True)
                netobj2pro, subject2pro, file2pro = collect_nodes_from_log(json_files)

                # 形成一个更完整的视图
                #按 benign/malicious 分开存储
                if category == "benign":

                    print("==========collect_edges_from_log=======start")
                    t0 = time.time()
                    df = collect_edges_from_log(df, json_files, True)
                    t1 = time.time()
                    print("==========collect_edges_from_log=======end")
                    print(f"耗时: {t1 - t0:.2f} 秒")

                    self.begin = df  # 存储到 base.py 定义的属性
                    print(f"  - 良性数据: {len(df)} 条边")
                elif category == "malicious":
                    print("==========collect_edges_from_log=======start")
                    t0 = time.time()
                    df = collect_edges_from_log(df, json_files, False )
                    t1 = time.time()
                    print("==========collect_edges_from_log=======end")
                    print(f"耗时: {t1 - t0:.2f} 秒")
                    self.malicious = df  # 存储到 base.py 定义的属性
                    print(f"  - 恶意数据: {len(df)} 条边")
                
                # 合并到总数据集（用于 use_df）
                self.all_dfs.append(df)
                
                merge_properties(netobj2pro, self.all_netobj2pro)
                merge_properties(subject2pro, self.all_subject2pro)
                merge_properties(file2pro, self.all_file2pro)
                
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
            df["timestamp_dt"] = pd.to_numeric(df["timestamp"], errors="coerce")  # 转为数值
            df["timestamp_dt"] = df["timestamp_dt"] // 1000  # 可能是微秒，转换为毫秒
            df["timestamp_dt"] = pd.to_datetime(df["timestamp_dt"], unit="us", errors="coerce")  # 转换为 datetime
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
        nodes_props, nodes_type, edges_map, node_frequency,node_last_ts =  {}, {}, {}, {},{}

        for r in df.itertuples(index=False):
            action = getattr(r, "action")
            actor_id = getattr(r, "actorID")
            object_id = getattr(r, "objectID")
            raw_ts = getattr(r, "timestamp")
            timestamp = float(raw_ts) if raw_ts is not None else 0.0

            # 频率统计
            node_frequency[actor_id] = node_frequency.get(actor_id, 0) + 1
            node_frequency[object_id] = node_frequency.get(object_id, 0) + 1

            # === 更新时间戳 ===
            node_last_ts[actor_id] = max(timestamp, node_last_ts.get(actor_id, 0))
            node_last_ts[object_id] = max(timestamp, node_last_ts.get(object_id, 0))

            # actor 节点
            props_actor = extract_properties(actor_id, r, action,
                                             self.all_netobj2pro, self.all_subject2pro, self.all_file2pro)
            add_node_properties(nodes_props, actor_id, props_actor)
            if actor_id not in nodes_type:
                nodes_type[actor_id] = getattr(r, "actor_type")

            # object 节点
            props_obj = extract_properties(object_id, r, action,
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




def collect_nodes_from_log(paths):
    netobj2pro = {}
    subject2pro = {}
    file2pro = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                # --- NetFlowObject ---
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.NetFlowObject"' in line:
                    try:
                        res = re.findall(
                            'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":"(.*?)","localPort":(.*?),"remoteAddress":"(.*?)","remotePort":(.*?),',
                            line
                        )[0]
                        nodeid = res[0]
                        srcaddr = res[2]
                        srcport = res[3]
                        dstaddr = res[4]
                        dstport = res[5]
                        nodeproperty = f"{srcaddr},{srcport},{dstaddr},{dstport}"
                        netobj2pro[nodeid] = nodeproperty
                    except Exception:
                        pass

                # --- Subject ---
                elif '{"datum":{"com.bbn.tc.schema.avro.cdm18.Subject"' in line:
                    try:
                        res = re.findall(
                            'Subject":{"uuid":"(.*?)"(.*?)"cmdLine":{"string":"(.*?)"}(.*?)"properties":{"map":{"tgid":"(.*?)"',
                            line
                        )[0]
                        nodeid = res[0]
                        cmdLine = res[2]
                        tgid = res[4]
                        try:
                            path_str = re.findall('"path":"(.*?)"', line)[0]
                            path = path_str
                        except Exception:
                            path = "null"
                        nodeProperty = f"{cmdLine},{tgid},{path}"
                        subject2pro[nodeid] = nodeProperty
                    except Exception:
                        pass

                # --- FileObject ---
                elif '{"datum":{"com.bbn.tc.schema.avro.cdm18.FileObject"' in line:
                    try:
                        res = re.findall(
                            'FileObject":{"uuid":"(.*?)"(.*?)"filename":"(.*?)"',
                            line
                        )[0]
                        nodeid = res[0]
                        filepath = res[2]
                        nodeproperty = filepath
                        file2pro[nodeid] = nodeproperty
                    except Exception:
                        pass

    return netobj2pro, subject2pro, file2pro


def collect_edges_from_log(d, paths, benigin, max_lines= 600000):
    info = []
    for p in paths:
        with open(p, "rb") as f:
            for i, line in enumerate(f):
                if benigin and i >= max_lines:
                    break
                if b"EVENT" not in line:
                    continue
                try:
                    x = orjson.loads(line)
                except Exception:
                    continue

                try:
                    ev = x["datum"]["com.bbn.tc.schema.avro.cdm18.Event"]
                except Exception:
                    continue

                action = ev.get("type", "")
                actor = (ev.get("subject") or {}).get("com.bbn.tc.schema.avro.cdm18.UUID", "")
                obj = (ev.get("predicateObject") or {}).get("com.bbn.tc.schema.avro.cdm18.UUID", "")
                timestamp = ev.get("timestampNanos", "")
                cmd = ((ev.get("properties") or {}).get("map") or {}).get("cmdLine", "")
                path = (ev.get("predicateObjectPath") or {}).get("string", "")
                path2 = (ev.get("predicateObject2Path") or {}).get("string", "")

                obj2 = (ev.get("predicateObject2") or {}).get("com.bbn.tc.schema.avro.cdm18.UUID")
                if obj2:
                    info.append({
                        "actorID": actor, "objectID": obj2, "action": action,
                        "timestamp": timestamp, "exec": cmd, "path": path2
                    })

                info.append({
                    "actorID": actor, "objectID": obj, "action": action,
                    "timestamp": timestamp, "exec": cmd, "path": path
                })

    rdf = pd.DataFrame.from_records(info).astype(str)
    d = d.astype(str)

    return d.merge(rdf, how="inner",
                   on=["actorID", "objectID", "action", "timestamp"]) \
        .drop_duplicates()


def extract_properties(node_id, row, action, netobj2pro, subject2pro, file2pro):
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