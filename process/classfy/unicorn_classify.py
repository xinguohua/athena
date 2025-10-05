#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNICORN Detector (Prographer-style class)
----------------------------------------
基于 UNICORN 论文的聚类阈值检测算法
    - 使用 dataclass Config
    - 统一训练、保存、预测接口
    - 打印检测得分表 + 输出 diff_vectors
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import numpy as np
from scipy.spatial.distance import pdist, squareform

from process.classfy import BaseClassify


# ======================================================
# ========== 配置类 =====================================
# ======================================================
@dataclass
class UnicornConfig:
    max_k: int = 6
    n_trials: int = 20
    max_iter: int = 500
    num_stds: float = 1.0
    metric: str = "both"
    model_save_path: str = "unicorn_model.json"
    threshold: float = 0.016


# ======================================================
# ========== 工具函数 ===================================
# ======================================================
def pairwise_hamming(arr: np.ndarray) -> np.ndarray:
    return squareform(pdist(arr, metric="hamming"))

def hamming(a, b):
    return np.mean(a != b)


class KMedoids:
    """简化版 K-Medoids"""
    def __init__(self, dists: np.ndarray, k: int, max_iter: int = 200, n_trials: int = 20, seed: int = 42):
        self.dists = dists
        self.k = k
        self.max_iter = max_iter
        self.n_trials = n_trials
        self.seed = seed

    def run(self):
        N = self.dists.shape[0]
        rng = random.Random(self.seed)
        best_cost = float("inf")
        best = None

        def assign_and_cost(meds):
            labels = np.zeros(N, dtype=int)
            cost = 0.0
            for i in range(N):
                best_j, best_d = min(((j, self.dists[i, m]) for j, m in enumerate(meds)), key=lambda t: t[1])
                labels[i] = best_j
                cost += best_d
            return labels, cost

        for _ in range(self.n_trials):
            meds = rng.sample(range(N), min(self.k, N))
            labels, cost = assign_and_cost(meds)
            improved = True
            it = 0
            while improved and it < self.max_iter:
                improved = False
                it += 1
                for mi, m in enumerate(list(meds)):
                    for h in range(N):
                        if h in meds:
                            continue
                        trial = list(meds)
                        trial[mi] = h
                        t_labels, t_cost = assign_and_cost(trial)
                        if t_cost + 1e-12 < cost:
                            meds, labels, cost = trial, t_labels, t_cost
                            improved = True
            if cost < best_cost:
                best_cost = cost
                best = (meds, labels)
        return best


# ======================================================
# ========== 检测器实现 =================================
# ======================================================
class UnicornClassify(BaseClassify):
    def __init__(self, cfg: Optional[UnicornConfig] = None, **kwargs):
        super().__init__()
        self.cfg = cfg or UnicornConfig()
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

    # ---------- 模型结构 ----------
    def _build_model(self):
        """这里模型本质是一个存放多个子模型的字典"""
        return {}

    def save(self, path=None):
        """覆盖 BaseClassify.save，用 JSON 保存聚类模型"""
        path = path or self.cfg.model_save_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model, f)
        print(f"[Save] model saved to {path}")

    def load(self, path=None):
        """覆盖 BaseClassify.load，用 JSON 读取"""
        path = path or self.cfg.model_save_path
        with open(path, "r") as f:
            self.model = json.load(f)
        print(f"[Load] model loaded from {path} ({len(self.model)} submodels)")

    # ---------- 训练逻辑 ----------
    def _train_loop(self, embeddings, **kwargs):
        """
        embeddings: dict 或 ndarray
            - 如果是 dict: {name: np.ndarray(sketch_vectors)}
            - 如果是 ndarray: 单一数据集
        """
        cfg = self.cfg
        if isinstance(embeddings, np.ndarray):
            train_data = {"default": embeddings}
        elif isinstance(embeddings, dict):
            train_data = embeddings
        else:
            raise TypeError("embeddings 必须是 ndarray 或 dict[str, ndarray]")

        print(f"[UNICORN] Training {len(train_data)} benign submodels...")

        for name, sketches in train_data.items():
            print(f"  -> {name}: {sketches.shape}")
            dists = pairwise_hamming(sketches)
            best_cost, best_res = float("inf"), None

            # K-Medoids 搜索最佳聚类
            for k in range(1, cfg.max_k + 1):
                km = KMedoids(dists, k, cfg.max_iter, cfg.n_trials)
                meds, labels = km.run()
                cost = sum(dists[i, meds[labels[i]]] for i in range(len(labels)))
                if cost < best_cost:
                    best_cost, best_res = cost, (meds, labels)

            meds, labels = best_res
            clusters = []
            for j, m_idx in enumerate(meds):
                members = np.where(labels == j)[0]
                if len(members) == 0:
                    continue
                medoid = sketches[m_idx]
                dists_j = [hamming(sketches[i], medoid) for i in members]
                clusters.append({
                    "medoid": medoid.tolist(),
                    "mean": float(np.mean(dists_j)),
                    "max": float(np.max(dists_j)),
                    "std": float(np.std(dists_j) + 1e-12),
                })
            self.model[name] = {"clusters": clusters}

        # 保存
        self.save(cfg.model_save_path)

    # ---------- 检测 ----------
    def predict(self, embeddings: np.ndarray, threshold: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
        """
        返回：
            labels: 0 正常 / 1 异常
            diff_vectors: 异常详情
        """
        if self.model is None or not self.model:
            raise RuntimeError("model 未训练或未加载")

        cfg = self.cfg
        num_stds = cfg.num_stds
        metric = cfg.metric
        threshold = threshold or cfg.threshold

        pred_labels = np.zeros(len(embeddings), dtype=int)
        diff_vectors, scores = {}, {}

        for i, sk in enumerate(embeddings):
            distances = []
            for submodel in self.model.values():
                for c in submodel["clusters"]:
                    d = hamming(sk, np.array(c["medoid"]))
                    distances.append({
                        "d": d,
                        "mean": c["mean"],
                        "max": c["max"],
                        "std": c["std"],
                    })

            # 判定
            abnormal = True
            for dd in distances:
                mean_ok = dd["d"] <= dd["mean"] + num_stds * dd["std"]
                max_ok = dd["d"] <= dd["max"] + num_stds * dd["std"]
                ok = (mean_ok and max_ok) if metric == "both" else (mean_ok if metric == "mean" else max_ok)
                if ok:
                    abnormal = False
                    break

            pred_labels[i] = 1 if abnormal else 0
            scores[i] = np.mean([d["d"] for d in distances])
            if abnormal:
                diff_vectors[i] = {
                    "position": i,
                    "score": scores[i],
                    "distances": distances,
                }

        # ------- 打印结果 -------
        print("\n--- UNICORN 检测结果 ---")
        print("索引 | 平均距离 | 状态")
        print("-" * 35)
        for i, sc in scores.items():
            status = "🔴 异常" if pred_labels[i] == 1 else "🟢 正常"
            print(f"{i:3d}  | {sc:.6f} | {status}")
        print("-" * 35 + "\n")

        return pred_labels, diff_vectors