#!/usr/bin/env python3
"""Isolation Forest benchmark over precomputed Drain bucket feature CSVs."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.ensemble import IsolationForest


REPRESENTATIONS = {
    "a": "Representation A rarity-only",
    "b": "Representation B top-50 counts + rarity",
    "c": "Representation C hash buckets",
}


@dataclass
class Metrics:
    dataset: str
    model: str
    records: int
    train_records: int
    test_records: int
    positives: int
    predicted: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    k: int
    precision_at_k: float
    recall_at_k: float

    @staticmethod
    def header() -> list[str]:
        return [
            "dataset",
            "model",
            "records",
            "train_records",
            "test_records",
            "positives",
            "predicted",
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
            "f1",
            "k",
            "precision_at_k",
            "recall_at_k",
        ]

    def row(self) -> list[str]:
        return [
            self.dataset,
            self.model,
            str(self.records),
            str(self.train_records),
            str(self.test_records),
            str(self.positives),
            str(self.predicted),
            str(self.tp),
            str(self.fp),
            str(self.fn),
            str(self.tn),
            f"{self.precision:.6f}",
            f"{self.recall:.6f}",
            f"{self.f1:.6f}",
            str(self.k),
            f"{self.precision_at_k:.6f}",
            f"{self.recall_at_k:.6f}",
        ]


def load_feature_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_flags: list[bool] = []
    labels: list[bool] = []
    values: list[list[float]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        feature_start = 3
        if len(header) <= feature_start:
            raise ValueError(f"{path} has no feature columns")
        for row in reader:
            train_flags.append(row[1] == "1")
            labels.append(row[2] == "1")
            values.append([float(value) for value in row[feature_start:]])
    return (
        np.asarray(train_flags, dtype=bool),
        np.asarray(labels, dtype=bool),
        np.asarray(values, dtype=np.float64),
    )


def top_k_predictions(scores: np.ndarray, k: int) -> np.ndarray:
    predictions = np.zeros(scores.shape[0], dtype=bool)
    if k <= 0:
        return predictions
    order = np.argsort(-scores, kind="mergesort")
    predictions[order[: min(k, scores.shape[0])]] = True
    return predictions


def compute_metrics(
    dataset: str,
    model: str,
    labels: np.ndarray,
    train_mask: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> Metrics:
    test_mask = ~train_mask
    y_test = labels[test_mask]
    positives = int(y_test.sum())
    k = positives
    tp = int(np.logical_and(y_test, predictions).sum())
    fp = int(np.logical_and(~y_test, predictions).sum())
    fn = int(np.logical_and(y_test, ~predictions).sum())
    tn = int(np.logical_and(~y_test, ~predictions).sum())
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    top_k = top_k_predictions(scores, k)
    top_k_hits = int(np.logical_and(y_test, top_k).sum())
    precision_at_k = 0.0 if k == 0 else top_k_hits / k
    recall_at_k = 0.0 if positives == 0 else top_k_hits / positives
    return Metrics(
        dataset=dataset,
        model=model,
        records=int(labels.shape[0]),
        train_records=int(train_mask.sum()),
        test_records=int(test_mask.sum()),
        positives=positives,
        predicted=int(predictions.sum()),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        k=k,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
    )


def run_one(
    path: str,
    dataset: str,
    representation: str,
    seed: int,
    n_estimators: int,
    contamination: float | str,
) -> Metrics:
    train_mask, labels, values = load_feature_csv(path)
    train_values = values[train_mask]
    test_values = values[~train_mask]
    if train_values.shape[0] == 0 or test_values.shape[0] == 0:
        raise ValueError(f"{path} must contain both train and test rows")
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_values)
    scores = -model.score_samples(test_values)
    predictions = model.predict(test_values) == -1
    contamination_label = contamination if isinstance(contamination, str) else f"{contamination:g}"
    return compute_metrics(
        dataset,
        "Isolation Forest deployable contamination="
        + str(contamination_label)
        + " + "
        + REPRESENTATIONS[representation],
        labels,
        train_mask,
        scores,
        predictions,
    )


def iter_jobs(prefix: str, datasets: Iterable[str], representations: Iterable[str]) -> Iterable[tuple[str, str, str]]:
    for dataset in datasets:
        for representation in representations:
            path = f"{prefix}_{dataset}_{representation}.csv"
            if os.path.exists(path):
                yield path, dataset, representation
            else:
                print(f"missing feature file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-prefix", default="benchmark-results/log-ad/bucket_features")
    parser.add_argument("--datasets", default="bgl,thunderbird")
    parser.add_argument("--representations", default="a,b,c")
    parser.add_argument("--output", default="benchmark-results/log-ad/if_bucket_metrics.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--contamination", default="0.02")
    args = parser.parse_args()

    datasets = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    representations = [item.strip().lower() for item in args.representations.split(",") if item.strip()]
    contamination: float | str
    contamination = "auto" if args.contamination == "auto" else float(args.contamination)
    results = []
    for path, dataset, representation in iter_jobs(args.feature_prefix, datasets, representations):
        print(f"running IF dataset={dataset} representation={representation} path={path}")
        results.append(run_one(path, dataset, representation, args.seed, args.trees, contamination))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(Metrics.header())
        for result in results:
            writer.writerow(result.row())

    print(",".join(Metrics.header()))
    for result in results:
        print(",".join(result.row()))


if __name__ == "__main__":
    main()
