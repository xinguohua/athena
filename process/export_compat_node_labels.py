"""
Export PIDSMaker-style node label CSV from ORIGINAL ProGrapher pipeline.

Output format:
  node_id,y_true,pred,score

Notes:
  - This is a compatibility export for downstream tooling expecting
    PIDSMaker-like CSV shape.
  - ProGrapher is snapshot-level detection. We project snapshot prediction
    to nodes in that snapshot, then aggregate by stable node_id.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import platform
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
import torch.nn.functional as F

from process.datahandlers import get_handler
from process.embedders import get_embedder_by_name
from process.classfy import get_classfy


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    system = platform.system().lower()
    return config["local"] if "windows" in system else config["remote"]


def stable_node_id(node_name: str) -> int:
    digest = hashlib.blake2b(node_name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def predict_with_scores(classify, embeddings: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    classify.model.eval()
    x = torch.as_tensor(embeddings, dtype=torch.float32, device=classify.device)
    seq_len = classify.cfg.sequence_length_L

    pred_labels = np.zeros(len(x), dtype=int)
    scores = np.zeros(len(x), dtype=float)

    with torch.no_grad():
        for i in range(len(x)):
            if i < seq_len:
                pad = torch.zeros(seq_len - i, x.size(1), device=classify.device)
                seq = torch.cat([pad, x[:i]], dim=0) if i > 0 else x[0].repeat(seq_len, 1)
            else:
                seq = x[i - seq_len : i]

            seq = seq.unsqueeze(0)
            target = x[i]
            pred = classify.model(seq).squeeze(0)
            error = F.mse_loss(pred, target).item()
            scores[i] = error
            if error > threshold:
                pred_labels[i] = 1

    return pred_labels, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run original ProGrapher and export compatible node labels CSV")
    parser.add_argument("--dataset", required=True, help="theia|cadets|trace|clearscope|theia5|cadets5|optcday1")
    parser.add_argument("--scene", default="", help="e.g. theia311")
    parser.add_argument("--dataset_key", default="THEIA_E3", help="Folder key in output path, e.g. THEIA_E3")
    parser.add_argument("--gid", default="prographer_compat", help="Run identity suffix")
    parser.add_argument("--threshold", type=float, default=0.0048, help="ProGrapher anomaly threshold")
    parser.add_argument(
        "--output_root",
        default="/workspace/prographer/artifacts/compat_training",
        help="Root directory for compatibility artifact layout",
    )
    args = parser.parse_args()

    cfg = load_config()
    path_map = cfg["path_map"]

    dataset_name = args.dataset
    scene_name = args.scene or None
    gid = args.gid

    print(f"[INFO] dataset={dataset_name} scene={scene_name or 'all'} gid={gid}")
    handler = get_handler(dataset_name, True, path_map, scene_name=scene_name)
    handler.load()
    handler.build_graph(gid)

    embedder_cls = get_embedder_by_name("prographer")
    embedder_model_path = f"prographer_encoder_{gid}.pth"
    embedder = embedder_cls(handler.snapshots, model_path=embedder_model_path)
    embedder.train()
    snapshot_embeddings = embedder.get_snapshot_embeddings()

    benign_embeddings = snapshot_embeddings[handler.benign_idx_start : handler.benign_idx_end + 1]
    detector_model_path = f"prographer_detector_{gid}.pth"
    classify = get_classfy("prographer", model_save_path=detector_model_path)
    classify.train(benign_embeddings)

    pred_labels, scores = predict_with_scores(classify, snapshot_embeddings, threshold=args.threshold)

    agg: Dict[int, Dict[str, float]] = {}
    for sidx, snap in enumerate(handler.snapshots):
        spred = int(pred_labels[sidx])
        sscore = float(scores[sidx])
        for v in snap.vs:
            name = str(v.attributes().get("name", f"s{sidx}_v{v.index}"))
            nid = stable_node_id(name)
            y_true = int(v.attributes().get("label", 0))

            prev = agg.get(nid)
            if prev is None:
                agg[nid] = {
                    "y_true": y_true,
                    "pred": spred,
                    "score": sscore,
                }
            else:
                prev["y_true"] = max(int(prev["y_true"]), y_true)
                prev["pred"] = max(int(prev["pred"]), spred)
                prev["score"] = max(float(prev["score"]), sscore)

    out_base = Path(args.output_root) / args.dataset_key
    csv_path = out_base / "training_labels" / "model_epoch_11" / "train_node_labels.csv"
    done_path = out_base / "done.txt"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "y_true", "pred", "score"])
        for nid in sorted(agg):
            row = agg[nid]
            writer.writerow([nid, int(row["y_true"]), int(row["pred"]), float(row["score"])])

    done_path.write_text(
        f"status=done\n"
        f"dataset={dataset_name}\n"
        f"scene={scene_name or 'all'}\n"
        f"gid={gid}\n"
        f"rows={len(agg)}\n"
        f"csv={csv_path}\n",
        encoding="utf-8",
    )

    print(f"[OK] wrote {csv_path} ({len(agg)} rows)")
    print(f"[OK] wrote {done_path}")


if __name__ == "__main__":
    main()
