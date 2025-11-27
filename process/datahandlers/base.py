from abc import ABC, abstractmethod
import pickle

class BaseProcessor(ABC):
    def __init__(self, base_path, train):
        self.base_path = base_path
        self.train = train
        self.all_dfs = []                 # 多个数据帧
        self.all_netobj2pro = {}  # 网络对象 UUID → 属性字符串
        self.all_subject2pro = {}  # 进程 UUID → 属性字符串
        self.all_file2pro = {}  # 文件 UUID → 属性字符串
        self.all_labels = []
        self.total_loaded_bytes = 0
        ###########
        self.begin  = []
        self.malicious = []
        self.snapshots = []
        self.benign_idx_start = 0
        self.benign_idx_end = 0
        self.malicious_idx_start = 0
        self.malicious_idx_end = 0


    @abstractmethod
    def load(self):
        """加载原始数据，返回预处理好的 DataFrame 以及属性映射字典"""
        pass

    @abstractmethod
    def create_snapshots_from_graph(self, df, is_malicious):
        pass


    def build_graph(self, gid=None):
        self.snapshots = []
        # 良性
        print("===============构建良性图并检测社区=============")
        self.benign_idx_start = len(self.snapshots)
        benign_snaps = self.create_snapshots_from_graph(self.begin, is_malicious=False)
        self.snapshots.extend(benign_snaps)
        self.benign_idx_end = len(self.snapshots) - 1 if benign_snaps else -1

        # 恶意
        print("===============构建恶意图并检测社区=============")
        self.malicious_idx_start = len(self.snapshots)
        mal_snaps = self.create_snapshots_from_graph(self.malicious, is_malicious=True)
        self.snapshots.extend(mal_snaps)
        self.malicious_idx_end = len(self.snapshots) - 1 if mal_snaps else -1

        print(f"总共生成了 {len(self.snapshots)} 个快照")
        print(f"良性快照索引范围: {self.benign_idx_start} 到 {self.benign_idx_end}")
        print(f"恶意快照索引范围: {self.malicious_idx_start} 到 {self.malicious_idx_end}")

        # 输出报告文件名：若提供身份 gid，则拼接到文件名
        report_file = f"all_snapshots_{gid}.txt" if gid else "all_snapshots.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            for i, g in enumerate(self.snapshots):
                f.write(f"Community {i}:\n")
                for v in g.vs:
                    attrs = v.attributes()
                    attr_str = ", ".join([f"{k}={v[k]}" for k in attrs])
                    f.write(f"  Vertex {v.index}: {attr_str}\n")
                f.write("\n")
            print(f"{report_file} write completed ")  # 打印进度

        snapshot_data = {
            'all_snapshots': self.snapshots,
            'benign_idx_start': self.benign_idx_start,
            'benign_idx_end': self.benign_idx_end,
            'malicious_idx_start': self.malicious_idx_start,
            'malicious_idx_end': self.malicious_idx_end,
        }
        # 快照数据文件名：追加身份后缀（若有）
        snapshot_file = f"snapshot_data_{gid}.pkl" if gid else "snapshot_data.pkl"
        with open(snapshot_file, 'wb') as f:
            pickle.dump(snapshot_data, f)
        print(f"快照数据已保存到: {snapshot_file}")
