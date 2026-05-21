#!/usr/bin/env python3
"""
Unsupervised online log anomaly benchmark using:
  * online parsing when raw text needs it
  * per-key EWMA thresholds for rates/counts/latency
  * Count-Min and SpaceSaving sketches for high-cardinality keys
  * Bloom filters for novelty
  * Page-Hinkley drift detection to reset/retune baselines
  * n-gram surprisal only for ordered HDFS block sessions

This is intentionally dependency-light; it uses Python stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


class EWMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.count = 0
        self.mean = 0.0
        self.var = 1.0

    def z(self, value: float) -> float:
        if self.count < 2:
            return 0.0
        return (value - self.mean) / math.sqrt(max(self.var, 1e-6))

    def abs_z(self, value: float) -> float:
        return abs(self.z(value))

    def update(self, value: float):
        self.count += 1
        if self.count == 1:
            self.mean = value
            self.var = 1.0
            return
        delta = value - self.mean
        self.mean += self.alpha * delta
        self.var = (1.0 - self.alpha) * (self.var + self.alpha * delta * delta)


class BloomFilter:
    def __init__(self, bits: int = 1 << 22, hashes: int = 4):
        self.bits = bits
        self.hashes = hashes
        self.mask = bytearray(bits // 8)

    def _positions(self, key: str):
        first = hash(key)
        second = hash(("salt", key)) | 1
        for i in range(self.hashes):
            yield (first + i * second) % self.bits

    def contains(self, key: str) -> bool:
        for pos in self._positions(key):
            if not (self.mask[pos >> 3] & (1 << (pos & 7))):
                return False
        return True

    def add(self, key: str):
        for pos in self._positions(key):
            self.mask[pos >> 3] |= 1 << (pos & 7)


class CountMinSketch:
    def __init__(self, width: int = 1 << 15, depth: int = 4):
        self.width = width
        self.depth = depth
        self.rows = [[0] * width for _ in range(depth)]

    def _index(self, key: str, row: int) -> int:
        return hash((row, key)) % self.width

    def add(self, key: str, count: int = 1):
        for row in range(self.depth):
            self.rows[row][self._index(key, row)] += count

    def estimate(self, key: str) -> int:
        return min(self.rows[row][self._index(key, row)] for row in range(self.depth))


class SpaceSaving:
    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.counts: dict[str, int] = {}

    def add(self, key: str, count: int = 1):
        if key in self.counts:
            self.counts[key] += count
        elif len(self.counts) < self.capacity:
            self.counts[key] = count
        else:
            victim = min(self.counts, key=self.counts.get)
            value = self.counts.pop(victim)
            self.counts[key] = value + count

    def contains(self, key: str) -> bool:
        return key in self.counts


class PageHinkley:
    def __init__(self, delta: float = 0.02, threshold: float = 25.0, alpha: float = 0.999):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0
        self.count = 0

    def update(self, value: float) -> bool:
        self.count += 1
        if self.count == 1:
            self.mean = value
            self.cumulative = 0.0
            self.minimum = 0.0
            return False
        self.mean = self.alpha * self.mean + (1.0 - self.alpha) * value
        self.cumulative += value - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        drift = self.cumulative - self.minimum > self.threshold
        if drift:
            self.reset(value)
        return drift

    def reset(self, value: float = 0.0):
        self.mean = value
        self.cumulative = 0.0
        self.minimum = 0.0
        self.count = 1


class SimpleDrainParser:
    def __init__(self, similarity: float = 0.5):
        self.similarity = similarity
        self.clusters_by_length: dict[int, list[list[str]]] = defaultdict(list)
        self.ids_by_length: dict[int, list[int]] = defaultdict(list)
        self.next_id = 0

    def parse(self, message: str) -> int:
        tokens = self.tokenize(message)
        candidates = self.clusters_by_length[len(tokens)]
        ids = self.ids_by_length[len(tokens)]
        best = -1
        best_similarity = -1.0
        for i, template in enumerate(candidates):
            similarity = self._similarity(tokens, template)
            if similarity > best_similarity:
                best_similarity = similarity
                best = i
        if best >= 0 and best_similarity >= self.similarity:
            template = candidates[best]
            for i, token in enumerate(tokens):
                if token != template[i]:
                    template[i] = "<*>"
            return ids[best]
        template_id = self.next_id
        self.next_id += 1
        candidates.append(tokens)
        ids.append(template_id)
        return template_id

    @staticmethod
    def tokenize(message: str) -> list[str]:
        answer = []
        for raw in message.lower().split():
            if any(ch.isdigit() for ch in raw):
                answer.append("<*>")
            else:
                token = raw.strip(" \t\r\n.,:;()[]{}\"'")
                if token:
                    answer.append(token)
        return answer or ["<empty>"]

    @staticmethod
    def _similarity(tokens: list[str], template: list[str]) -> float:
        comparable = 0
        same = 0
        for token, template_token in zip(tokens, template):
            if template_token != "<*>":
                comparable += 1
                same += token == template_token
        return 1.0 if comparable == 0 else same / comparable


@dataclass
class Metrics:
    name: str
    evaluated: int = 0
    positives: int = 0
    predicted: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    elapsed_s: float = 0.0
    resets: int = 0
    state: str = ""

    def record(self, label: bool, prediction: bool):
        self.evaluated += 1
        self.positives += int(label)
        self.predicted += int(prediction)
        if label and prediction:
            self.tp += 1
        elif not label and prediction:
            self.fp += 1
        elif label:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        return 2 * self.precision * self.recall / (self.precision + self.recall) if self.precision + self.recall else 0.0

    def as_row(self) -> list:
        return [
            self.name,
            self.evaluated,
            self.positives,
            self.predicted,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            self.precision,
            self.recall,
            self.f1,
            self.elapsed_s,
            self.resets,
            self.state,
        ]


class HdfsSketchDetector:
    def __init__(self, alpha: float, z: float, warmup: int, update_on_anomaly: bool):
        self.alpha = alpha
        self.z = z
        self.warmup = warmup
        self.update_on_anomaly = update_on_anomaly
        self.index = 0
        self.total = EWMA(alpha)
        self.latency = EWMA(alpha)
        self.rate = EWMA(alpha)
        self.template_stats = [EWMA(alpha) for _ in range(29)]
        self.bigram_surprisal = EWMA(alpha)
        self.bloom = BloomFilter(bits=1 << 21)
        self.cms = CountMinSketch(width=1 << 14)
        self.frequent = SpaceSaving(1024)
        self.bigram_counts: Counter[str] = Counter()
        self.seen_bigrams: set[str] = set()
        self.total_bigrams = 0
        self.ph = PageHinkley(threshold=50.0)
        self.resets = 0

    def process(self, counts: list[int], latency: float, sequence: list[str]) -> bool:
        total = sum(counts)
        rate = total / max(1.0, latency)
        template_max_z = max((self.template_stats[i].z(counts[i]) for i in range(len(counts))), default=0.0)
        missing_max_z = max(
            ((self.template_stats[i].mean - counts[i]) / math.sqrt(max(self.template_stats[i].var, 1e-6))
             for i in range(len(counts)) if self.template_stats[i].mean >= 3.0),
            default=0.0,
        )
        novelty = 0
        for i, count in enumerate(counts):
            if count > 0:
                key = f"E{i+1}"
                if not self.bloom.contains(key):
                    novelty += 1
        bigram_novel = 0
        surprisal_sum = 0.0
        bigrams = [f"{a}>{b}" for a, b in zip(sequence, sequence[1:])]
        for bigram in bigrams:
            if bigram not in self.seen_bigrams:
                bigram_novel += 1
            estimate = self.bigram_counts[bigram]
            probability = (estimate + 1.0) / (self.total_bigrams + 1024.0)
            surprisal_sum += -math.log(probability)
        surprisal = surprisal_sum / max(1, len(bigrams))
        score = max(
            self.total.abs_z(total),
            self.latency.z(math.log1p(latency)),
            self.rate.abs_z(rate),
            template_max_z,
            missing_max_z,
            self.bigram_surprisal.z(surprisal),
        )
        enough = self.index >= self.warmup
        anomaly = enough and (
            score >= self.z
            or novelty >= 2
            or (bigram_novel >= 3 and bigram_novel / max(1, len(bigrams)) >= 0.05)
        )
        if enough and self.ph.update(score):
            self.resets += 1
            self._reset_numeric()
            anomaly = False
        if not anomaly or self.update_on_anomaly:
            self._update(counts, latency, rate, sequence, bigrams, surprisal)
        self.index += 1
        return anomaly

    def _reset_numeric(self):
        self.total = EWMA(self.alpha)
        self.latency = EWMA(self.alpha)
        self.rate = EWMA(self.alpha)
        self.template_stats = [EWMA(self.alpha) for _ in range(29)]
        self.bigram_surprisal = EWMA(self.alpha)

    def _update(self, counts, latency, rate, sequence, bigrams, surprisal):
        self.total.update(sum(counts))
        self.latency.update(math.log1p(latency))
        self.rate.update(rate)
        for i, count in enumerate(counts):
            self.template_stats[i].update(count)
            if count > 0:
                key = f"E{i+1}"
                self.bloom.add(key)
                self.cms.add(key, count)
                self.frequent.add(key, count)
        self.bigram_surprisal.update(surprisal)
        for bigram in bigrams:
            self.seen_bigrams.add(bigram)
            self.bigram_counts[bigram] += 1
        self.total_bigrams += len(bigrams)


class BglSketchDetector:
    def __init__(self, alpha: float, z: float, warmup_buckets: int, update_on_anomaly: bool):
        self.alpha = alpha
        self.z = z
        self.warmup_buckets = warmup_buckets
        self.update_on_anomaly = update_on_anomaly
        self.index = 0
        self.total = EWMA(alpha)
        self.error = EWMA(alpha)
        self.warn = EWMA(alpha)
        self.fatal = EWMA(alpha)
        self.failure = EWMA(alpha)
        self.template_stats: dict[int, EWMA] = defaultdict(lambda: EWMA(alpha))
        self.bloom = BloomFilter(bits=1 << 23)
        self.cms = CountMinSketch(width=1 << 16)
        self.frequent = SpaceSaving(4096)
        self.ph = PageHinkley(threshold=35.0)
        self.resets = 0

    def process(self, bucket: dict) -> bool:
        counts: Counter[int] = bucket["templates"]
        total = bucket["total"]
        error = bucket["levels"].get("ERROR", 0)
        warn = bucket["levels"].get("WARNING", 0)
        fatal = bucket["levels"].get("FATAL", 0)
        failure = bucket["levels"].get("FAILURE", 0)
        template_max_z = 0.0
        new_template_count = 0
        repeated_new_template = 0
        for template, count in counts.items():
            template_max_z = max(template_max_z, self.template_stats[template].z(count))
            key = f"T:{template}"
            if not self.bloom.contains(key):
                new_template_count += 1
                repeated_new_template = max(repeated_new_template, count)
        new_node_template = 0
        repeated_new_node_template = 0
        for key, count in bucket["node_templates"].items():
            full = f"NT:{key}"
            if not self.bloom.contains(full):
                new_node_template += 1
                repeated_new_node_template = max(repeated_new_node_template, count)
        score = max(
            self.total.abs_z(total),
            self.error.z(error),
            self.warn.z(warn),
            self.fatal.z(fatal),
            self.failure.z(failure),
            template_max_z,
        )
        enough = self.index >= self.warmup_buckets
        anomaly = enough and (
            score >= self.z
            or error >= self.error.mean + max(8.0, self.z * math.sqrt(max(self.error.var, 1e-6)))
            or fatal >= self.fatal.mean + max(5.0, self.z * math.sqrt(max(self.fatal.var, 1e-6)))
            or failure >= self.failure.mean + max(3.0, self.z * math.sqrt(max(self.failure.var, 1e-6)))
            or repeated_new_template >= max(8, int(0.05 * max(1, total)))
            or repeated_new_node_template >= 4
            or (new_node_template >= 25 and fatal + failure + error >= 3)
        )
        if enough and self.ph.update(score):
            self.resets += 1
            self._reset_numeric()
            anomaly = False
        if not anomaly or self.update_on_anomaly:
            self._update(bucket)
        self.index += 1
        return anomaly

    def _reset_numeric(self):
        self.total = EWMA(self.alpha)
        self.error = EWMA(self.alpha)
        self.warn = EWMA(self.alpha)
        self.fatal = EWMA(self.alpha)
        self.failure = EWMA(self.alpha)
        self.template_stats = defaultdict(lambda: EWMA(self.alpha))

    def _update(self, bucket):
        counts: Counter[int] = bucket["templates"]
        levels = bucket["levels"]
        self.total.update(bucket["total"])
        self.error.update(levels.get("ERROR", 0))
        self.warn.update(levels.get("WARNING", 0))
        self.fatal.update(levels.get("FATAL", 0))
        self.failure.update(levels.get("FAILURE", 0))
        for template, count in counts.items():
            self.template_stats[template].update(count)
            key = f"T:{template}"
            self.bloom.add(key)
            self.cms.add(key, count)
            self.frequent.add(key, count)
        for key, count in bucket["node_templates"].items():
            full = f"NT:{key}"
            self.bloom.add(full)
            self.cms.add(full, count)
            self.frequent.add(full, count)


def parse_hdfs_sequence(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    return [part.strip() for part in raw.split(",") if part.strip()]


def run_hdfs(occurrence: Path, traces: Path, alpha: float, z: float, update_on_anomaly: bool, output_name: str) -> Metrics:
    detector = HdfsSketchDetector(alpha=alpha, z=z, warmup=1000, update_on_anomaly=update_on_anomaly)
    metrics = Metrics(output_name)
    started = time.perf_counter()
    with occurrence.open(newline="", encoding="utf-8") as occ_stream, traces.open(newline="", encoding="utf-8") as trace_stream:
        occ_reader = csv.reader(occ_stream)
        trace_reader = csv.reader(trace_stream)
        next(occ_reader)
        next(trace_reader)
        for index, (occ, trace) in enumerate(zip(occ_reader, trace_reader)):
            counts = [int(value) for value in occ[3:32]]
            label = occ[1] == "Fail"
            sequence = parse_hdfs_sequence(trace[3])
            latency = float(trace[5]) if len(trace) > 5 and trace[5] else 0.0
            prediction = detector.process(counts, latency, sequence)
            if index >= detector.warmup:
                metrics.record(label, prediction)
    metrics.elapsed_s = time.perf_counter() - started
    metrics.resets = detector.resets
    metrics.state = f"alpha={alpha},z={z},update_on_anomaly={update_on_anomaly},cms,bloom,ph,bigram"
    return metrics


def build_bgl_bucket(lines, parser: SimpleDrainParser):
    templates: Counter[int] = Counter()
    levels: Counter[str] = Counter()
    node_templates: Counter[str] = Counter()
    total = 0
    label = False
    for parts in lines:
        content = parts[9] if len(parts) >= 10 else ""
        message = f"{parts[8]} {parts[7]} {content}"
        template = parser.parse(message)
        node = parts[3]
        level = parts[8]
        templates[template] += 1
        levels[level] += 1
        node_templates[f"{node}:{template}"] += 1
        total += 1
        label = label or parts[0] != "-"
    return {"templates": templates, "levels": levels, "node_templates": node_templates, "total": total, "label": label}


def iter_bgl_buckets(path: Path, bucket_seconds: int, parser: SimpleDrainParser):
    current_key = None
    lines = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = line.strip().split(None, 9)
            if len(parts) < 9:
                continue
            try:
                key = int(parts[1]) // bucket_seconds
            except ValueError:
                continue
            if current_key is None:
                current_key = key
            elif key != current_key:
                yield build_bgl_bucket(lines, parser)
                lines = []
                current_key = key
            lines.append(parts)
    if lines:
        yield build_bgl_bucket(lines, parser)


def load_bgl_buckets(path: Path) -> tuple[list[dict], int, float]:
    parser = SimpleDrainParser()
    started = time.perf_counter()
    buckets = list(iter_bgl_buckets(path, 600, parser))
    return buckets, parser.next_id, time.perf_counter() - started


def run_bgl(buckets: list[dict], template_count: int, parse_elapsed: float, alpha: float, z: float,
            update_on_anomaly: bool, output_name: str) -> Metrics:
    detector = BglSketchDetector(alpha=alpha, z=z, warmup_buckets=80, update_on_anomaly=update_on_anomaly)
    metrics = Metrics(output_name)
    started = time.perf_counter()
    for index, bucket in enumerate(buckets):
        prediction = detector.process(bucket)
        if index >= detector.warmup_buckets:
            metrics.record(bucket["label"], prediction)
    metrics.elapsed_s = parse_elapsed + time.perf_counter() - started
    metrics.resets = detector.resets
    metrics.state = f"alpha={alpha},z={z},update_on_anomaly={update_on_anomaly},templates={template_count},cms,bloom,ph"
    return metrics


def write_csv(path: Path, metrics: list[Metrics]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "name",
            "evaluated",
            "positive",
            "predicted",
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
            "f1",
            "elapsed_s",
            "resets",
            "state",
        ])
        for metric in metrics:
            writer.writerow(metric.as_row())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs-occurrence", type=Path, required=True)
    parser.add_argument("--hdfs-traces", type=Path, required=True)
    parser.add_argument("--bgl-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = []
    configs = [(0.03, 3.0, True)]
    for alpha, z, update in configs:
        suffix = f"a{alpha}_z{z}_{'upd' if update else 'hold'}"
        metrics.append(run_hdfs(args.hdfs_occurrence, args.hdfs_traces, alpha, z, update, f"HDFS_session_sketch_{suffix}"))
    bgl_buckets, template_count, parse_elapsed = load_bgl_buckets(args.bgl_log)
    for alpha, z, update in configs:
        suffix = f"a{alpha}_z{z}_{'upd' if update else 'hold'}"
        metrics.append(run_bgl(bgl_buckets, template_count, parse_elapsed, alpha, z, update, f"BGL_10m_sketch_{suffix}"))
    write_csv(args.output, metrics)
    print(json.dumps([dict(zip([
        "name", "evaluated", "positive", "predicted", "tp", "fp", "fn", "tn", "precision", "recall",
        "f1", "elapsed_s", "resets", "state"], m.as_row())) for m in metrics], indent=2))


if __name__ == "__main__":
    main()
