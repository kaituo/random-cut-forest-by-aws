#!/usr/bin/env python3
"""
Supervised LogHub benchmark helpers for comparing against Drain + RF style
results. HDFS uses the full block/session event occurrence matrix. BGL uses
full raw logs aggregated into 10-minute Drain-template bucket count vectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEEDS = [42, 123, 456, 789, 1024]


class SimpleDrainParser:
    def __init__(self, similarity: float = 0.5):
        self.similarity = similarity
        self.clusters_by_length: dict[int, list[list[str]]] = defaultdict(list)
        self.cluster_ids_by_length: dict[int, list[int]] = defaultdict(list)
        self.next_id = 0

    def parse(self, message: str) -> int:
        tokens = self._tokenize(message)
        candidates = self.clusters_by_length[len(tokens)]
        candidate_ids = self.cluster_ids_by_length[len(tokens)]
        best_index = -1
        best_similarity = -1.0
        for i, template in enumerate(candidates):
            sim = self._similarity(tokens, template)
            if sim > best_similarity:
                best_similarity = sim
                best_index = i
        if best_index >= 0 and best_similarity >= self.similarity:
            template = candidates[best_index]
            for i, token in enumerate(tokens):
                if template[i] != token:
                    template[i] = "<*>"
            return candidate_ids[best_index]

        cluster_id = self.next_id
        self.next_id += 1
        candidates.append(tokens)
        candidate_ids.append(cluster_id)
        return cluster_id

    @staticmethod
    def _tokenize(message: str) -> list[str]:
        tokens = []
        for raw in message.lower().split():
            if any(ch.isdigit() for ch in raw):
                tokens.append("<*>")
                continue
            token = raw.strip(" \t\r\n.,:;()[]{}\"'")
            if token:
                tokens.append(token)
        return tokens or ["<empty>"]

    @staticmethod
    def _similarity(tokens: list[str], template: list[str]) -> float:
        comparable = 0
        exact = 0
        for token, templated in zip(tokens, template):
            if templated != "<*>":
                comparable += 1
                exact += int(token == templated)
        return 1.0 if comparable == 0 else exact / comparable


def run_classifier(name: str, x_train, y_train, x_test, y_test, seed: int) -> dict[str, float]:
    if name == "RF":
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed, class_weight="balanced_subsample")
    elif name == "LR":
        clf = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"),
        )
    elif name == "IF":
        clf = IsolationForest(contamination=float(np.mean(y_train)), n_jobs=-1, random_state=seed)
    else:
        raise ValueError(name)

    start = time.perf_counter()
    clf.fit(x_train, y_train)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    if name == "IF":
        y_pred = (clf.predict(x_test) == -1).astype(int)
    else:
        y_pred = clf.predict(x_test)
    inference_ms = (time.perf_counter() - start) / max(1, len(y_test)) * 1000

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "train_seconds": train_seconds,
        "inference_ms": inference_ms,
    }
    try:
        if name == "IF":
            scores = -clf.score_samples(x_test)
        else:
            scores = clf.predict_proba(x_test)[:, 1]
        metrics["auc"] = roc_auc_score(y_test, scores)
    except Exception:
        metrics["auc"] = math.nan
    return {key: float(value) for key, value in metrics.items()}


def summarize(seed_metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = seed_metrics[0].keys()
    answer = {}
    for key in keys:
        values = np.array([m[key] for m in seed_metrics], dtype=float)
        answer[key] = float(np.nanmean(values))
        answer[f"{key}_std"] = float(np.nanstd(values))
    return answer


def hdfs_session_matrix(path: Path):
    df = pd.read_csv(path)
    feature_columns = [col for col in df.columns if re.fullmatch(r"E\d+", col)]
    x = df[feature_columns].to_numpy(dtype=np.float32)
    y = (df["Label"].to_numpy() == "Fail").astype(np.int32)
    return x, y, {"sessions": int(len(y)), "features": len(feature_columns), "positives": int(y.sum())}


def bgl_bucket_matrix(path: Path, bucket_seconds: int):
    parser = SimpleDrainParser()
    bucket_counts: dict[int, Counter[int]] = defaultdict(Counter)
    bucket_labels: dict[int, int] = defaultdict(int)
    line_count = 0
    anomalous_lines = 0

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = line.strip().split(None, 9)
            if len(parts) < 9:
                continue
            try:
                bucket = int(int(parts[1]) // bucket_seconds)
            except ValueError:
                continue
            content = parts[9] if len(parts) >= 10 else ""
            message = f"{parts[8]} {parts[7]} {content}"
            template_id = parser.parse(message)
            bucket_counts[bucket][template_id] += 1
            anomaly = parts[0] != "-"
            if anomaly:
                bucket_labels[bucket] = 1
                anomalous_lines += 1
            line_count += 1

    buckets = sorted(bucket_counts)
    x = np.zeros((len(buckets), parser.next_id), dtype=np.float32)
    y = np.zeros(len(buckets), dtype=np.int32)
    for row, bucket in enumerate(buckets):
        y[row] = bucket_labels[bucket]
        for template_id, count in bucket_counts[bucket].items():
            x[row, template_id] = count
    meta = {
        "buckets": len(buckets),
        "features": int(parser.next_id),
        "positives": int(y.sum()),
        "lines": line_count,
        "anomalous_lines": anomalous_lines,
        "bucket_seconds": bucket_seconds,
    }
    return x, y, meta


def evaluate_dataset(name: str, x, y, methods: list[str]) -> dict:
    result = {
        "samples": int(len(y)),
        "features": int(x.shape[1]),
        "positives": int(y.sum()),
        "positive_rate": float(np.mean(y)),
        "methods": {},
    }
    for method in methods:
        seed_results = []
        for seed in SEEDS:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=seed, stratify=y
            )
            seed_results.append(run_classifier(method, x_train, y_train, x_test, y_test, seed))
        result["methods"][method] = summarize(seed_results)
    return result


def write_summary_csv(results: dict, output: Path):
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "dataset",
                "method",
                "samples",
                "features",
                "positive_rate",
                "precision",
                "recall",
                "f1",
                "auc",
                "train_seconds",
                "inference_ms",
            ]
        )
        for dataset, result in results.items():
            for method, metrics in result["methods"].items():
                writer.writerow(
                    [
                        dataset,
                        method,
                        result["samples"],
                        result["features"],
                        result["positive_rate"],
                        metrics["precision"],
                        metrics["recall"],
                        metrics["f1"],
                        metrics["auc"],
                        metrics["train_seconds"],
                        metrics["inference_ms"],
                    ]
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs-occurrence", type=Path, required=True)
    parser.add_argument("--bgl-log", type=Path, required=True)
    parser.add_argument("--bucket-seconds", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    x_hdfs, y_hdfs, hdfs_meta = hdfs_session_matrix(args.hdfs_occurrence)
    results["HDFS_session"] = evaluate_dataset("HDFS_session", x_hdfs, y_hdfs, ["RF", "LR", "IF"])
    results["HDFS_session"]["meta"] = hdfs_meta

    x_bgl, y_bgl, bgl_meta = bgl_bucket_matrix(args.bgl_log, args.bucket_seconds)
    results["BGL_10m_bucket"] = evaluate_dataset("BGL_10m_bucket", x_bgl, y_bgl, ["RF", "LR", "IF"])
    results["BGL_10m_bucket"]["meta"] = bgl_meta

    json_path = args.output_dir / "supervised_loghub_results.json"
    csv_path = args.output_dir / "supervised_loghub_results.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary_csv(results, csv_path)
    print(json.dumps(results, indent=2))
    print(f"saved {json_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
