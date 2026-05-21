#!/usr/bin/env python3
"""Compare sklearn IsolationForest and AWS RCF on the same Drain event vectors.

This intentionally mirrors the replication package for arXiv:2312.01934 for
the unfiltered + parsed-events setting:

* BGL: 10% sampled event-level rows, 95% test split
* Thunderbird: 0.3% sampled event-level rows, 95% test split
* Hadoop: all application-level sequences, 50% test split
* HDFS: 10% block/session-level sequences, 95% test split
* representation: Drain event IDs -> TfidfVectorizer
* IsolationForest: sklearn defaults, n_estimators=100, max_samples='auto'
* RCF: same feature matrix, 100 trees, sampleSize=min(256, train_size)

RCF is trained on the train split and scores the test split without updating on
the test rows, so the comparison is offline train/test like IsolationForest.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import jpype
import jpype.imports  # noqa: F401
import numpy as np
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from scipy import sparse
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
RCF_JAR = ROOT / "python_rcf_wrapper/lib/randomcutforest-core-4.0.0-SNAPSHOT.jar"
DEFAULT_DRAIN_CONFIG = Path("/tmp/LL-mod-unsupervised/parsers/drain3/drain3_no_masking.ini")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}")
BLOCK_RE = re.compile(r"\bblk_-?\d+\b")


@dataclass
class Dataset:
    name: str
    docs: list[str | list[str]]
    labels: np.ndarray
    test_fraction: float
    sequence: bool


@dataclass
class Result:
    dataset: str
    model: str
    rows: int
    train_rows: int
    test_rows: int
    positives: int
    negatives: int
    features: int
    auc: float
    seconds: float


def normalize(message: str) -> str:
    lowered = message.lower()
    digit_replaced = re.sub(r"\d", "0", lowered)
    return re.sub(r"0+", "0", digit_replaced)


def make_drain(config_path: Path) -> TemplateMiner:
    config = TemplateMinerConfig()
    config.load(str(config_path))
    return TemplateMiner(config=config)


def drain_event_id(miner: TemplateMiner, message: str) -> str:
    result = miner.add_log_message(normalize(message))
    return "e" + str(result["cluster_id"])


def sample_keep(rng: random.Random, fraction: float) -> bool:
    return fraction >= 1.0 or rng.random() < fraction


def load_bgl(path: Path, fraction: float, seed: int, drain_config: Path) -> Dataset:
    rng = random.Random(seed)
    miner = make_drain(drain_config)
    docs: list[str] = []
    labels: list[bool] = []
    with path.open("r", encoding="latin-1", errors="ignore") as handle:
        for line_number, line in enumerate(handle, 1):
            if not sample_keep(rng, fraction):
                continue
            parts = line.rstrip("\n").split(None, 9)
            if len(parts) < 9:
                continue
            message = parts[8] if len(parts) == 9 else parts[8] + " " + parts[9]
            if not message:
                continue
            docs.append(drain_event_id(miner, message))
            labels.append(not parts[0].startswith("-"))
            if line_number % 1_000_000 == 0:
                print(f"  BGL line={line_number} sampled={len(docs)}", flush=True)
    return Dataset("BGL", docs, np.asarray(labels, dtype=bool), 0.95, False)


def load_thunderbird(path: Path, fraction: float, seed: int, drain_config: Path) -> Dataset:
    rng = random.Random(seed)
    miner = make_drain(drain_config)
    docs: list[str] = []
    labels: list[bool] = []
    with path.open("r", encoding="latin-1", errors="ignore") as handle:
        for line_number, line in enumerate(handle, 1):
            if not sample_keep(rng, fraction):
                continue
            parts = line.rstrip("\n").split(None, 9)
            if len(parts) < 9:
                continue
            message = parts[8] if len(parts) == 9 else parts[8] + " " + parts[9]
            if not message:
                continue
            docs.append(drain_event_id(miner, message))
            labels.append(not parts[0].startswith("-"))
            if len(docs) and len(docs) % 100_000 == 0:
                print(f"  Thunderbird line={line_number} sampled={len(docs)}", flush=True)
    return Dataset("Thunderbird", docs, np.asarray(labels, dtype=bool), 0.95, False)


def load_hadoop(root: Path, drain_config: Path) -> Dataset:
    labels = parse_hadoop_labels(root / "abnormal_label.txt")
    miner = make_drain(drain_config)
    docs: list[list[str]] = []
    y: list[bool] = []
    for app_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("application_")):
        events: list[str] = []
        for file_path in sorted(app_dir.rglob("*.log")):
            for entry in hadoop_entries(file_path):
                events.append(drain_event_id(miner, entry))
        docs.append(events or ["e_empty"])
        y.append(labels.get(app_dir.name, True))
    return Dataset("Hadoop", docs, np.asarray(y, dtype=bool), 0.50, True)


def parse_hadoop_labels(path: Path) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    current_abnormal = False
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "Normal:":
                current_abnormal = False
            elif line.endswith(":"):
                current_abnormal = True
            elif line.startswith("+"):
                labels[line[1:].strip()] = current_abnormal
    return labels


def hadoop_entries(path: Path) -> Iterator[str]:
    current: list[str] = []
    with path.open("r", encoding="latin-1", errors="ignore") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if TIMESTAMP_RE.match(line):
                if current:
                    yield parse_hadoop_entry("\n".join(current))
                current = [line]
            elif current:
                current.append(line)
        if current:
            yield parse_hadoop_entry("\n".join(current))


def parse_hadoop_entry(entry: str) -> str:
    first_line = entry.split("\n", 1)[0]
    without_process = re.sub(r"\s*\[.*?\]\s*", " ", first_line)
    without_process = re.sub(r"\s+", " ", without_process).strip()
    parts = without_process.split(None, 4)
    return parts[4] if len(parts) >= 5 else without_process


def load_hdfs(matrix_path: Path, fraction: float, seed: int) -> Dataset:
    rng = random.Random(seed)
    docs: list[list[str]] = []
    labels: list[bool] = []
    with matrix_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        event_names = header[3:]
        for row in reader:
            if not sample_keep(rng, fraction):
                continue
            tokens: list[str] = []
            for event_name, value in zip(event_names, row[3:]):
                count = int(value)
                if count:
                    tokens.extend([event_name.lower()] * count)
            docs.append(tokens or ["e_empty"])
            labels.append(row[1].lower() == "fail")
    return Dataset("HDFS", docs, np.asarray(labels, dtype=bool), 0.95, True)


def split_dataset(dataset: Dataset, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset.labels))
    test_size = int(dataset.test_fraction * len(order))
    test_idx = order[:test_size]
    train_idx = order[test_size:]
    docs = np.asarray(dataset.docs, dtype=object)
    return docs[train_idx].tolist(), dataset.labels[train_idx], docs[test_idx].tolist(), dataset.labels[test_idx]


def vectorize(train_docs, test_docs, sequence: bool):
    if sequence:
        vectorizer = TfidfVectorizer(analyzer=lambda x: x)
    else:
        vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(train_docs)
    x_test = vectorizer.transform(test_docs)
    return x_train.astype(np.float32), x_test.astype(np.float32)


def isolation_forest_auc(x_train, x_test, y_test, seed: int) -> tuple[float, float]:
    started = time.time()
    model = IsolationForest(n_estimators=100, max_samples="auto", contamination="auto", random_state=seed)
    model.fit(x_train)
    scores = 1.0 - model.score_samples(x_test)
    return roc_auc_score(y_test, scores), time.time() - started


def ensure_jvm() -> None:
    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=[str(RCF_JAR)])


def rcf_auc(
    x_train,
    x_test,
    y_test,
    seed: int,
    update_test: bool = False,
    number_of_trees: int = 100,
    requested_sample_size: int | None = None,
) -> tuple[float, float]:
    ensure_jvm()
    from com.amazon.randomcutforest import RandomCutForest

    started = time.time()
    dimensions = x_train.shape[1]
    sample_size = min(requested_sample_size or 256, max(1, x_train.shape[0]))
    forest = (
        RandomCutForest.builder()
        .compact(True)
        .dimensions(dimensions)
        .numberOfTrees(number_of_trees)
        .sampleSize(sample_size)
        .randomSeed(seed)
        .parallelExecutionEnabled(False)
        .build()
    )
    point = jpype.JArray(jpype.JFloat)(dimensions)
    for row in sparse_rows(x_train):
        fill_point(point, row)
        forest.update(point)
        clear_point(point, row)

    scores = score_rcf_rows(forest, point, x_test, update_test)
    return roc_auc_score(y_test, scores), time.time() - started


def score_rcf_rows(forest, point, x_test, update_test: bool) -> np.ndarray:
    rows = list(sparse_rows(x_test))
    scores = np.empty(len(rows), dtype=np.float64)
    if update_test:
        for i, row in enumerate(rows):
            fill_point(point, row)
            scores[i] = forest.getAnomalyScore(point)
            forest.update(point)
            clear_point(point, row)
            if i and i % 250_000 == 0:
                print(f"    RCF scored={i}", flush=True)
        return scores

    cache: dict[tuple[tuple[int, ...], tuple[float, ...]], float] = {}
    for i, row in enumerate(rows):
        key = row_key(row)
        score = cache.get(key)
        if score is None:
            fill_point(point, row)
            score = float(forest.getAnomalyScore(point))
            clear_point(point, row)
            cache[key] = score
        scores[i] = score
    print(f"    RCF unique_test_vectors={len(cache)}", flush=True)
    return scores


def row_key(row: tuple[np.ndarray, np.ndarray]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    indices, values = row
    return tuple(int(index) for index in indices), tuple(round(float(value), 8) for value in values)


def sparse_rows(matrix) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    csr = matrix.tocsr()
    for i in range(csr.shape[0]):
        start, end = csr.indptr[i], csr.indptr[i + 1]
        yield csr.indices[start:end], csr.data[start:end]


def fill_point(point, row: tuple[np.ndarray, np.ndarray]) -> None:
    indices, values = row
    for index, value in zip(indices, values):
        point[int(index)] = float(value)


def clear_point(point, row: tuple[np.ndarray, np.ndarray]) -> None:
    indices, _ = row
    for index in indices:
        point[int(index)] = 0.0


def run_dataset(dataset: Dataset, seed: int, run_online_rcf: bool, rcf_trees: int, rcf_sample_size: int) -> list[Result]:
    train_docs, y_train, test_docs, y_test = split_dataset(dataset, seed)
    print(
        f"{dataset.name}: rows={len(dataset.labels)} train={len(y_train)} "
        f"test={len(y_test)} positives={int(y_test.sum())}",
        flush=True,
    )
    x_train, x_test = vectorize(train_docs, test_docs, dataset.sequence)
    print(f"{dataset.name}: features={x_train.shape[1]} nnz_train={x_train.nnz} nnz_test={x_test.nnz}", flush=True)

    results: list[Result] = []
    auc, seconds = isolation_forest_auc(x_train, x_test, y_test, seed)
    results.append(make_result(dataset.name, "Isolation Forest + parsed events", x_train, x_test, y_test, auc, seconds))
    print(f"{dataset.name}: IF AUC={auc:.6f} seconds={seconds:.2f}", flush=True)

    auc, seconds = rcf_auc(
        x_train,
        x_test,
        y_test,
        seed,
        update_test=False,
        number_of_trees=rcf_trees,
        requested_sample_size=rcf_sample_size,
    )
    results.append(
        make_result(
            dataset.name,
            f"RCF offline + parsed events ({rcf_trees}x{min(rcf_sample_size, max(1, x_train.shape[0]))})",
            x_train,
            x_test,
            y_test,
            auc,
            seconds,
        )
    )
    print(f"{dataset.name}: RCF offline AUC={auc:.6f} seconds={seconds:.2f}", flush=True)

    if run_online_rcf:
        auc, seconds = rcf_auc(
            x_train,
            x_test,
            y_test,
            seed,
            update_test=True,
            number_of_trees=rcf_trees,
            requested_sample_size=rcf_sample_size,
        )
        results.append(
            make_result(
                dataset.name,
                f"RCF score-update + parsed events ({rcf_trees}x{min(rcf_sample_size, max(1, x_train.shape[0]))})",
                x_train,
                x_test,
                y_test,
                auc,
                seconds,
            )
        )
        print(f"{dataset.name}: RCF score-update AUC={auc:.6f} seconds={seconds:.2f}", flush=True)

    return results


def make_result(dataset, model, x_train, x_test, y_test, auc, seconds) -> Result:
    return Result(
        dataset=dataset,
        model=model,
        rows=x_train.shape[0] + x_test.shape[0],
        train_rows=x_train.shape[0],
        test_rows=x_test.shape[0],
        positives=int(y_test.sum()),
        negatives=int((~y_test).sum()),
        features=x_train.shape[1],
        auc=float(auc),
        seconds=float(seconds),
    )


def write_results(path: Path, results: Sequence[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "model", "rows", "train_rows", "test_rows", "positives", "negatives", "features", "auc", "seconds"])
        for result in results:
            writer.writerow([
                result.dataset,
                result.model,
                result.rows,
                result.train_rows,
                result.test_rows,
                result.positives,
                result.negatives,
                result.features,
                f"{result.auc:.6f}",
                f"{result.seconds:.3f}",
            ])
        writer.writerow([])
        writer.writerow(["model", "average_auc"])
        for model in dict.fromkeys(result.model for result in results):
            values = [result.auc for result in results if result.model == model]
            writer.writerow([model, f"{sum(values) / len(values):.6f}"])


def cache_path(cache_dir: Path, dataset_name: str, seed: int) -> Path:
    return cache_dir / f"{dataset_name}_seed{seed}.pkl"


def load_or_build_dataset(dataset_name: str, args, data_root: Path, drain_config: Path) -> Dataset:
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        path = cache_path(cache_dir, dataset_name, args.seed)
        if path.exists():
            with path.open("rb") as handle:
                dataset = pickle.load(handle)
            print(f"{dataset.name}: loaded cache {path}", flush=True)
            return dataset

    if dataset_name == "bgl":
        dataset = load_bgl(data_root / "BGL/BGL.log", 0.1, args.seed, drain_config)
    elif dataset_name == "thunderbird":
        dataset = load_thunderbird(data_root / "Thunderbird/Thunderbird.log", 0.003, args.seed, drain_config)
    elif dataset_name == "hadoop":
        dataset = load_hadoop(data_root / "Hadoop", drain_config)
    elif dataset_name == "hdfs":
        dataset = load_hdfs(data_root / "HDFS/preprocessed/Event_occurrence_matrix.csv", 0.1, args.seed)
    else:
        raise ValueError(f"unsupported dataset: {dataset_name}")

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_path(cache_dir, dataset_name, args.seed)
        with path.open("wb") as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{dataset.name}: wrote cache {path}", flush=True)
    return dataset


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/tmp/loghub_data")
    parser.add_argument("--datasets", default="bgl,thunderbird,hadoop,hdfs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="benchmark-results/log-ad/if_rcf_parsed_events_results.csv")
    parser.add_argument("--drain-config", default=str(DEFAULT_DRAIN_CONFIG))
    parser.add_argument("--online-rcf", action="store_true")
    parser.add_argument("--rcf-trees", type=int, default=100)
    parser.add_argument("--rcf-sample-size", type=int, default=256)
    parser.add_argument("--cache-dir", default="benchmark-results/log-ad/cache")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    drain_config = Path(args.drain_config)
    results: list[Result] = []
    for dataset_name in [item.strip().lower() for item in args.datasets.split(",") if item.strip()]:
        started = time.time()
        dataset = load_or_build_dataset(dataset_name, args, data_root, drain_config)
        print(f"{dataset.name}: loaded in {time.time() - started:.2f}s", flush=True)
        results.extend(run_dataset(dataset, args.seed, args.online_rcf, args.rcf_trees, args.rcf_sample_size))

    output = Path(args.output)
    write_results(output, results)
    print(output)
    for result in results:
        print(f"{result.dataset},{result.model},{result.auc:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
