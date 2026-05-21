/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

package com.amazon.randomcutforest.examples.parkservices;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import com.amazon.randomcutforest.RandomCutForest;

/**
 * Fast score-threshold tuning over precomputed log bucket features.
 */
public final class RcfBucketScoreTuningBenchmark {

    private RcfBucketScoreTuningBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        List<Result> results = new ArrayList<>();
        for (String dataset : config.datasets) {
            DatasetFeatures features = DatasetFeatures.load(config.featurePrefix, dataset);
            List<RunConfig> runConfigs = RunConfig.suite();
            System.out.printf(Locale.ROOT, "dataset=%s configs=%d rows=%d train=%d%n", dataset, runConfigs.size(),
                    features.size(), features.trainCount);
            int count = 0;
            for (RunConfig runConfig : runConfigs) {
                ++count;
                if (count == 1 || count % 10 == 0) {
                    System.out.printf(Locale.ROOT, "  %s %d/%d %s%n", dataset, count, runConfigs.size(),
                            runConfig.summary());
                }
                FeatureData data = features.featureData(runConfig.representation);
                if (runConfig.normalize) {
                    data = data.normalized();
                }
                for (boolean online : new boolean[] { true, false }) {
                    results.addAll(run(dataset, data, runConfig, online, config));
                }
            }
        }
        writeResults(config.output, results);
        printTop(results, config.top);
    }

    private static List<Result> run(String dataset, FeatureData data, RunConfig runConfig, boolean online,
            Config config) {
        RandomCutForest.Builder<?> builder = RandomCutForest.builder().compact(true)
                .dimensions(data.values[0].length * runConfig.shingleSize).sampleSize(runConfig.sampleSize)
                .numberOfTrees(runConfig.numberOfTrees).randomSeed(config.seed).outputAfter(config.outputAfter);
        if (runConfig.timeDecay > 0.0) {
            builder.timeDecay(runConfig.timeDecay);
        }
        RandomCutForest forest = builder.build();
        List<Double> calibrationScores = new ArrayList<>();
        List<ScoredLabel> testRows = new ArrayList<>();
        for (int i = 0; i < data.values.length; i++) {
            double[] point = shingle(data.values, i, runConfig.shingleSize);
            double score = forest.isOutputReady() ? forest.getAnomalyScore(point) : 0.0;
            if (i < data.trainCount) {
                if (i >= config.outputAfter && Double.isFinite(score)) {
                    calibrationScores.add(score);
                }
                forest.update(point);
            } else {
                testRows.add(new ScoredLabel(score, data.labels[i]));
                if (online) {
                    forest.update(point);
                }
            }
        }

        List<Result> results = new ArrayList<>();
        for (double alertRate : alertRates(dataset)) {
            double threshold = quantile(calibrationScores, 1.0 - alertRate);
            Metrics metrics = Metrics.from(testRows, threshold);
            results.add(metrics.toResult(dataset, runConfig, online, alertRate, threshold));
        }
        return results;
    }

    private static double[] shingle(double[][] values, int index, int shingleSize) {
        int baseDimensions = values[0].length;
        double[] point = new double[baseDimensions * shingleSize];
        for (int offset = 0; offset < shingleSize; offset++) {
            int source = index - shingleSize + 1 + offset;
            if (source >= 0) {
                System.arraycopy(values[source], 0, point, offset * baseDimensions, baseDimensions);
            }
        }
        return point;
    }

    private static double quantile(List<Double> values, double quantile) {
        if (values.isEmpty()) {
            return Double.POSITIVE_INFINITY;
        }
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int index = (int) Math.floor(Math.max(0.0, Math.min(1.0, quantile)) * (sorted.size() - 1));
        return sorted.get(index);
    }

    private static double[] alertRates(String dataset) {
        if ("bgl".equals(dataset)) {
            return new double[] { 0.02, 0.05, 0.087, 0.10 };
        }
        if ("thunderbird".equals(dataset)) {
            return new double[] { 0.02, 0.05, 0.10, 0.20 };
        }
        return new double[] { 0.02, 0.05, 0.10 };
    }

    private static void writeResults(String output, List<Result> results) throws IOException {
        Path path = Paths.get(output);
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write(Result.header());
            writer.newLine();
            for (Result result : results) {
                writer.write(result.toCsv());
                writer.newLine();
            }
        }
    }

    private static void printTop(List<Result> results, int top) {
        List<Result> sorted = new ArrayList<>(results);
        sorted.sort((a, b) -> Double.compare(b.f1, a.f1));
        System.out.println(Result.header());
        for (int i = 0; i < Math.min(top, sorted.size()); i++) {
            System.out.println(sorted.get(i).toCsv());
        }
    }

    private static final class DatasetFeatures {
        private final FeatureData a;
        private final FeatureData b;
        private final FeatureData c;
        private final int trainCount;

        private DatasetFeatures(FeatureData a, FeatureData b, FeatureData c) {
            this.a = a;
            this.b = b;
            this.c = c;
            this.trainCount = a.trainCount;
            if (a.size() != b.size() || a.size() != c.size() || a.trainCount != b.trainCount
                    || a.trainCount != c.trainCount) {
                throw new IllegalArgumentException("feature files are not aligned");
            }
        }

        private static DatasetFeatures load(String prefix, String dataset) throws IOException {
            return new DatasetFeatures(FeatureData.load(prefix + "_" + dataset + "_a.csv", "A"),
                    FeatureData.load(prefix + "_" + dataset + "_b.csv", "B"),
                    FeatureData.load(prefix + "_" + dataset + "_c.csv", "C"));
        }

        private int size() {
            return a.size();
        }

        private FeatureData featureData(String representation) {
            switch (representation) {
            case "A":
                return a;
            case "B":
                return b;
            case "C":
                return c;
            case "B25":
                return deriveB25();
            case "BR":
                return deriveBR();
            default:
                throw new IllegalArgumentException("unknown representation " + representation);
            }
        }

        private FeatureData deriveB25() {
            double[][] values = new double[b.size()][30];
            for (int i = 0; i < b.size(); i++) {
                System.arraycopy(b.values[i], 0, values[i], 0, 25);
                double other = Math.expm1(b.values[i][50]);
                for (int j = 25; j < 50; j++) {
                    other += Math.expm1(b.values[i][j]);
                }
                values[i][25] = Math.log1p(Math.max(0.0, other));
                values[i][26] = b.values[i][51];
                values[i][27] = b.values[i][52];
                values[i][28] = b.values[i][53];
                values[i][29] = b.values[i][54];
            }
            return new FeatureData("B25", values, b.labels, b.trainCount);
        }

        private FeatureData deriveBR() {
            double[][] values = new double[b.size()][62];
            for (int i = 0; i < b.size(); i++) {
                System.arraycopy(b.values[i], 0, values[i], 0, 55);
                double total = Math.max(1.0, Math.expm1(a.values[i][0]));
                values[i][55] = a.values[i][2] / total;
                values[i][56] = a.values[i][3] / total;
                values[i][57] = a.values[i][1] / total;
                values[i][58] = a.values[i][8];
                values[i][59] = a.values[i][7];
                values[i][60] = a.values[i][4];
                values[i][61] = a.values[i][6];
            }
            return new FeatureData("BR", values, b.labels, b.trainCount);
        }
    }

    private static final class FeatureData {
        private final String name;
        private final double[][] values;
        private final boolean[] labels;
        private final int trainCount;

        private FeatureData(String name, double[][] values, boolean[] labels, int trainCount) {
            this.name = name;
            this.values = values;
            this.labels = labels;
            this.trainCount = trainCount;
        }

        private static FeatureData load(String path, String name) throws IOException {
            List<double[]> rows = new ArrayList<>();
            List<Boolean> labels = new ArrayList<>();
            int trainCount = 0;
            try (BufferedReader reader = Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8)) {
                String header = reader.readLine();
                if (header == null) {
                    throw new IOException("empty feature file " + path);
                }
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split(",");
                    if ("1".equals(parts[1])) {
                        ++trainCount;
                    }
                    labels.add("1".equals(parts[2]));
                    double[] values = new double[parts.length - 3];
                    for (int i = 3; i < parts.length; i++) {
                        values[i - 3] = Double.parseDouble(parts[i]);
                    }
                    rows.add(values);
                }
            }
            boolean[] labelArray = new boolean[labels.size()];
            for (int i = 0; i < labels.size(); i++) {
                labelArray[i] = labels.get(i);
            }
            return new FeatureData(name, rows.toArray(new double[0][]), labelArray, trainCount);
        }

        private int size() {
            return values.length;
        }

        private FeatureData normalized() {
            int dimensions = values[0].length;
            double[] mean = new double[dimensions];
            double[] variance = new double[dimensions];
            for (int i = 0; i < trainCount; i++) {
                for (int j = 0; j < dimensions; j++) {
                    mean[j] += values[i][j];
                }
            }
            for (int j = 0; j < dimensions; j++) {
                mean[j] /= Math.max(1, trainCount);
            }
            for (int i = 0; i < trainCount; i++) {
                for (int j = 0; j < dimensions; j++) {
                    double delta = values[i][j] - mean[j];
                    variance[j] += delta * delta;
                }
            }
            double[][] transformed = new double[values.length][dimensions];
            for (int j = 0; j < dimensions; j++) {
                variance[j] = Math.sqrt(variance[j] / Math.max(1, trainCount));
                if (variance[j] < 1e-9) {
                    variance[j] = 1.0;
                }
            }
            for (int i = 0; i < values.length; i++) {
                for (int j = 0; j < dimensions; j++) {
                    transformed[i][j] = (values[i][j] - mean[j]) / variance[j];
                }
            }
            return new FeatureData(name + "Z", transformed, labels, trainCount);
        }
    }

    private static final class RunConfig {
        private final String experiment;
        private final String representation;
        private final int shingleSize;
        private final int numberOfTrees;
        private final int sampleSize;
        private final double timeDecay;
        private final boolean normalize;

        private RunConfig(String experiment, String representation, int shingleSize, int numberOfTrees, int sampleSize,
                double timeDecay, boolean normalize) {
            this.experiment = experiment;
            this.representation = representation;
            this.shingleSize = shingleSize;
            this.numberOfTrees = numberOfTrees;
            this.sampleSize = sampleSize;
            this.timeDecay = timeDecay;
            this.normalize = normalize;
        }

        private static List<RunConfig> suite() {
            List<RunConfig> configs = new ArrayList<>();
            for (int shingle : new int[] { 1, 4, 6, 12 }) {
                configs.add(new RunConfig("shingle", "A", shingle, 50, 256, -1.0, false));
            }
            for (int shingle : new int[] { 1, 2, 4, 6 }) {
                configs.add(new RunConfig("shingle", "B", shingle, 50, 256, -1.0, false));
                configs.add(new RunConfig("shingle", "B", shingle, 50, 256, -1.0, true));
            }
            for (int shingle : new int[] { 2, 4, 6 }) {
                configs.add(new RunConfig("top25", "B25", shingle, 50, 256, -1.0, false));
                configs.add(new RunConfig("augmented_rates", "BR", shingle, 50, 256, -1.0, false));
                configs.add(new RunConfig("augmented_rates_norm", "BR", shingle, 50, 256, -1.0, true));
            }
            for (int shingle : new int[] { 1, 2, 4 }) {
                configs.add(new RunConfig("hash", "C", shingle, 50, 256, -1.0, false));
            }
            for (int sampleSize : new int[] { 64, 128, 256, 512 }) {
                configs.add(new RunConfig("sample_size", "B", 4, 50, sampleSize, -1.0, false));
            }
            for (int sampleSize : new int[] { 128, 256 }) {
                configs.add(new RunConfig("trees_100", "B", 4, 100, sampleSize, -1.0, false));
            }
            for (double timeDecay : new double[] { 1.0 / 500.0, 1.0 / 1000.0, 1.0 / 2000.0 }) {
                configs.add(new RunConfig("time_decay", "B", 4, 50, 128, timeDecay, false));
            }
            return configs;
        }

        private String summary() {
            return String.format(Locale.ROOT, "%s rep=%s shingle=%d trees=%d sample=%d decay=%s normalize=%s",
                    experiment, representation, shingleSize, numberOfTrees, sampleSize,
                    timeDecay > 0.0 ? Double.toString(timeDecay) : "default", normalize);
        }
    }

    private static final class ScoredLabel {
        private final double score;
        private final boolean label;

        private ScoredLabel(double score, boolean label) {
            this.score = score;
            this.label = label;
        }
    }

    private static final class Metrics {
        private final long positives;
        private final long predicted;
        private final long tp;
        private final long fp;
        private final long fn;
        private final long tn;
        private final double precision;
        private final double recall;
        private final double f1;
        private final int k;
        private final double precisionAtK;
        private final double recallAtK;
        private final int segments;
        private final int segmentHits0;
        private final int segmentHits1;
        private final int segmentHits2;

        private Metrics(long positives, long predicted, long tp, long fp, long fn, long tn, double precision,
                double recall, double f1, int k, double precisionAtK, double recallAtK, int segments,
                int segmentHits0, int segmentHits1, int segmentHits2) {
            this.positives = positives;
            this.predicted = predicted;
            this.tp = tp;
            this.fp = fp;
            this.fn = fn;
            this.tn = tn;
            this.precision = precision;
            this.recall = recall;
            this.f1 = f1;
            this.k = k;
            this.precisionAtK = precisionAtK;
            this.recallAtK = recallAtK;
            this.segments = segments;
            this.segmentHits0 = segmentHits0;
            this.segmentHits1 = segmentHits1;
            this.segmentHits2 = segmentHits2;
        }

        private static Metrics from(List<ScoredLabel> rows, double threshold) {
            boolean[] predictions = new boolean[rows.size()];
            long positives = 0;
            long predicted = 0;
            long tp = 0;
            long fp = 0;
            long fn = 0;
            long tn = 0;
            for (int i = 0; i < rows.size(); i++) {
                ScoredLabel row = rows.get(i);
                boolean prediction = row.score >= threshold;
                predictions[i] = prediction;
                if (row.label) {
                    ++positives;
                }
                if (prediction) {
                    ++predicted;
                }
                if (row.label && prediction) {
                    ++tp;
                } else if (!row.label && prediction) {
                    ++fp;
                } else if (row.label) {
                    ++fn;
                } else {
                    ++tn;
                }
            }
            double precision = tp + fp == 0 ? 0.0 : tp / (double) (tp + fp);
            double recall = tp + fn == 0 ? 0.0 : tp / (double) (tp + fn);
            double f1 = precision + recall == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);
            int k = (int) positives;
            long topKHits = topKHits(rows, k);
            int[] segmentHits = segmentHits(rows, predictions);
            return new Metrics(positives, predicted, tp, fp, fn, tn, precision, recall, f1, k,
                    k == 0 ? 0.0 : topKHits / (double) k, positives == 0 ? 0.0 : topKHits / (double) positives,
                    segmentHits[0], segmentHits[1], segmentHits[2], segmentHits[3]);
        }

        private Result toResult(String dataset, RunConfig runConfig, boolean online, double alertRate, double threshold) {
            return new Result(dataset, runConfig, online, alertRate, threshold, positives, predicted, tp, fp, fn, tn,
                    precision, recall, f1, k, precisionAtK, recallAtK, segments, segmentHits0, segmentHits1,
                    segmentHits2);
        }

        private static long topKHits(List<ScoredLabel> rows, int k) {
            if (k <= 0) {
                return 0;
            }
            List<ScoredLabel> sorted = new ArrayList<>(rows);
            sorted.sort((a, b) -> Double.compare(b.score, a.score));
            long hits = 0;
            for (int i = 0; i < Math.min(k, sorted.size()); i++) {
                if (sorted.get(i).label) {
                    ++hits;
                }
            }
            return hits;
        }

        private static int[] segmentHits(List<ScoredLabel> rows, boolean[] predictions) {
            int segments = 0;
            int hits0 = 0;
            int hits1 = 0;
            int hits2 = 0;
            int i = 0;
            while (i < rows.size()) {
                if (!rows.get(i).label) {
                    ++i;
                    continue;
                }
                int start = i;
                while (i < rows.size() && rows.get(i).label) {
                    ++i;
                }
                int end = i - 1;
                ++segments;
                if (hasPrediction(predictions, start, end, 0)) {
                    ++hits0;
                }
                if (hasPrediction(predictions, start, end, 1)) {
                    ++hits1;
                }
                if (hasPrediction(predictions, start, end, 2)) {
                    ++hits2;
                }
            }
            return new int[] { segments, hits0, hits1, hits2 };
        }

        private static boolean hasPrediction(boolean[] predictions, int start, int end, int window) {
            int from = Math.max(0, start - window);
            int to = Math.min(predictions.length - 1, end + window);
            for (int i = from; i <= to; i++) {
                if (predictions[i]) {
                    return true;
                }
            }
            return false;
        }
    }

    private static final class Result {
        private final String dataset;
        private final RunConfig runConfig;
        private final boolean online;
        private final double alertRate;
        private final double threshold;
        private final long positives;
        private final long predicted;
        private final long tp;
        private final long fp;
        private final long fn;
        private final long tn;
        private final double precision;
        private final double recall;
        private final double f1;
        private final int k;
        private final double precisionAtK;
        private final double recallAtK;
        private final int segments;
        private final int segmentHits0;
        private final int segmentHits1;
        private final int segmentHits2;

        private Result(String dataset, RunConfig runConfig, boolean online, double alertRate, double threshold,
                long positives, long predicted, long tp, long fp, long fn, long tn, double precision, double recall,
                double f1, int k, double precisionAtK, double recallAtK, int segments, int segmentHits0,
                int segmentHits1, int segmentHits2) {
            this.dataset = dataset;
            this.runConfig = runConfig;
            this.online = online;
            this.alertRate = alertRate;
            this.threshold = threshold;
            this.positives = positives;
            this.predicted = predicted;
            this.tp = tp;
            this.fp = fp;
            this.fn = fn;
            this.tn = tn;
            this.precision = precision;
            this.recall = recall;
            this.f1 = f1;
            this.k = k;
            this.precisionAtK = precisionAtK;
            this.recallAtK = recallAtK;
            this.segments = segments;
            this.segmentHits0 = segmentHits0;
            this.segmentHits1 = segmentHits1;
            this.segmentHits2 = segmentHits2;
        }

        private static String header() {
            return String.join(",", Arrays.asList("dataset", "experiment", "representation", "mode", "alert_rate",
                    "threshold", "shingle_size", "trees", "sample_size", "time_decay", "normalize", "positives",
                    "predicted", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "k", "precision_at_k",
                    "recall_at_k", "segments", "segments_hit_w0", "segment_recall_w0", "segments_hit_w1",
                    "segment_recall_w1", "segments_hit_w2", "segment_recall_w2"));
        }

        private String toCsv() {
            return String.format(Locale.ROOT,
                    "%s,%s,%s,%s,%.3f,%.8f,%d,%d,%d,%s,%s,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%d,%d,%.6f,%d,%.6f,%d,%.6f",
                    dataset, runConfig.experiment, runConfig.representation, online ? "online" : "frozen", alertRate,
                    threshold, runConfig.shingleSize, runConfig.numberOfTrees, runConfig.sampleSize,
                    runConfig.timeDecay > 0.0 ? String.format(Locale.ROOT, "%.8f", runConfig.timeDecay) : "default",
                    runConfig.normalize, positives, predicted, tp, fp, fn, tn, precision, recall, f1, k, precisionAtK,
                    recallAtK, segments, segmentHits0, segments == 0 ? 0.0 : segmentHits0 / (double) segments,
                    segmentHits1, segments == 0 ? 0.0 : segmentHits1 / (double) segments, segmentHits2,
                    segments == 0 ? 0.0 : segmentHits2 / (double) segments);
        }
    }

    private static final class Config {
        private final List<String> datasets;
        private final String featurePrefix;
        private final String output;
        private final int outputAfter;
        private final long seed;
        private final int top;

        private Config(List<String> datasets, String featurePrefix, String output, int outputAfter, long seed, int top) {
            this.datasets = datasets;
            this.featurePrefix = featurePrefix;
            this.output = output;
            this.outputAfter = outputAfter;
            this.seed = seed;
            this.top = top;
        }

        private static Config parse(String[] args) {
            Map<String, String> values = new LinkedHashMap<>();
            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                if (!arg.startsWith("--") || i + 1 >= args.length) {
                    throw new IllegalArgumentException("expected --key value near " + arg);
                }
                values.put(arg.substring(2), args[++i]);
            }
            return new Config(split(values.getOrDefault("datasets", "bgl,thunderbird")),
                    values.getOrDefault("feature-prefix", "benchmark-results/log-ad/bucket_features"),
                    values.getOrDefault("output", "benchmark-results/log-ad/rcf_bucket_score_tuning.csv"),
                    Integer.parseInt(values.getOrDefault("output-after", "32")),
                    Long.parseLong(values.getOrDefault("seed", "42")),
                    Integer.parseInt(values.getOrDefault("top", "20")));
        }

        private static List<String> split(String csv) {
            List<String> result = new ArrayList<>();
            for (String item : csv.split(",")) {
                String trimmed = item.trim().toLowerCase(Locale.ROOT);
                if (!trimmed.isEmpty()) {
                    result.add(trimmed);
                }
            }
            return result;
        }
    }
}
