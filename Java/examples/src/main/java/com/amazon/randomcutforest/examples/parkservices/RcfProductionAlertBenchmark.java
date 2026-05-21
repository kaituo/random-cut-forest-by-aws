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
 * Production-style alert-budget evaluation over bucket features and stricter labels.
 */
public final class RcfProductionAlertBenchmark {

    private RcfProductionAlertBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        List<Result> results = new ArrayList<>();
        for (String dataset : config.datasets) {
            Metadata metadata = Metadata.load(config.metadataPrefix + "_" + dataset + ".csv");
            for (RunConfig runConfig : RunConfig.suite()) {
                FeatureData data = FeatureData.load(
                        config.featurePrefix + "_" + dataset + "_" + runConfig.featureName + ".csv");
                data.validate(metadata);
                System.out.printf(Locale.ROOT, "dataset=%s feature=%s shingle=%d rows=%d train=%d%n", dataset,
                        runConfig.featureName, runConfig.shingleSize, data.size(), data.trainCount);
                results.addAll(run(dataset, data, metadata, runConfig, config));
            }
        }
        writeResults(config.output, results);
        printTop(results, config.top);
    }

    private static List<Result> run(String dataset, FeatureData data, Metadata metadata, RunConfig runConfig,
            Config config) {
        RandomCutForest.Builder<?> builder = RandomCutForest.builder().compact(true)
                .dimensions(data.values[0].length * runConfig.shingleSize).sampleSize(runConfig.sampleSize)
                .numberOfTrees(runConfig.numberOfTrees).randomSeed(config.seed).outputAfter(config.outputAfter);
        if (runConfig.timeDecay > 0.0) {
            builder.timeDecay(runConfig.timeDecay);
        }
        RandomCutForest forest = builder.build();
        List<Double> calibrationScores = new ArrayList<>();
        List<Double> testScores = new ArrayList<>();
        List<Integer> testAnomalousCounts = new ArrayList<>();
        List<Double> testAnomalousFractions = new ArrayList<>();
        for (int i = 0; i < data.values.length; i++) {
            double[] point = shingle(data.values, i, runConfig.shingleSize);
            double score = forest.isOutputReady() ? forest.getAnomalyScore(point) : 0.0;
            if (i < data.trainCount) {
                if (i >= config.outputAfter && Double.isFinite(score)) {
                    calibrationScores.add(score);
                }
                forest.update(point);
            } else {
                testScores.add(score);
                testAnomalousCounts.add(metadata.anomalousCounts[i]);
                testAnomalousFractions.add(metadata.anomalousFractions[i]);
                forest.update(point);
            }
        }

        List<Result> results = new ArrayList<>();
        for (double alertRate : config.alertRates) {
            double threshold = quantile(calibrationScores, 1.0 - alertRate);
            boolean[] predictions = new boolean[testScores.size()];
            for (int i = 0; i < testScores.size(); i++) {
                predictions[i] = testScores.get(i) >= threshold;
            }
            for (LabelSpec labelSpec : LabelSpec.suite()) {
                boolean[] labels = labelSpec.labels(testAnomalousCounts, testAnomalousFractions);
                results.add(Metrics.from(labels, predictions, config.bucketSeconds)
                        .toResult(dataset, runConfig, labelSpec, alertRate, threshold));
            }
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

    private static final class FeatureData {
        private final boolean[] trainFlags;
        private final double[][] values;
        private final int trainCount;

        private FeatureData(boolean[] trainFlags, double[][] values, int trainCount) {
            this.trainFlags = trainFlags;
            this.values = values;
            this.trainCount = trainCount;
        }

        private static FeatureData load(String path) throws IOException {
            List<Boolean> train = new ArrayList<>();
            List<double[]> rows = new ArrayList<>();
            int trainCount = 0;
            try (BufferedReader reader = Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8)) {
                String header = reader.readLine();
                if (header == null) {
                    throw new IOException("empty feature file " + path);
                }
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split(",");
                    boolean isTrain = "1".equals(parts[1]);
                    train.add(isTrain);
                    if (isTrain) {
                        ++trainCount;
                    }
                    double[] values = new double[parts.length - 3];
                    for (int i = 3; i < parts.length; i++) {
                        values[i - 3] = Double.parseDouble(parts[i]);
                    }
                    rows.add(values);
                }
            }
            boolean[] trainFlags = new boolean[train.size()];
            for (int i = 0; i < train.size(); i++) {
                trainFlags[i] = train.get(i);
            }
            return new FeatureData(trainFlags, rows.toArray(new double[0][]), trainCount);
        }

        private int size() {
            return values.length;
        }

        private void validate(Metadata metadata) {
            if (metadata.trainFlags.length != trainFlags.length) {
                throw new IllegalArgumentException("metadata/feature length mismatch");
            }
            for (int i = 0; i < trainFlags.length; i++) {
                if (metadata.trainFlags[i] != trainFlags[i]) {
                    throw new IllegalArgumentException("metadata/feature train mismatch at " + i);
                }
            }
        }
    }

    private static final class Metadata {
        private final boolean[] trainFlags;
        private final int[] anomalousCounts;
        private final double[] anomalousFractions;

        private Metadata(boolean[] trainFlags, int[] anomalousCounts, double[] anomalousFractions) {
            this.trainFlags = trainFlags;
            this.anomalousCounts = anomalousCounts;
            this.anomalousFractions = anomalousFractions;
        }

        private static Metadata load(String path) throws IOException {
            List<Boolean> train = new ArrayList<>();
            List<Integer> counts = new ArrayList<>();
            List<Double> fractions = new ArrayList<>();
            try (BufferedReader reader = Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8)) {
                String header = reader.readLine();
                if (header == null) {
                    throw new IOException("empty metadata file " + path);
                }
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split(",");
                    train.add("1".equals(parts[1]));
                    counts.add(Integer.parseInt(parts[4]));
                    fractions.add(Double.parseDouble(parts[5]));
                }
            }
            boolean[] trainFlags = new boolean[train.size()];
            int[] anomalousCounts = new int[counts.size()];
            double[] anomalousFractions = new double[fractions.size()];
            for (int i = 0; i < train.size(); i++) {
                trainFlags[i] = train.get(i);
                anomalousCounts[i] = counts.get(i);
                anomalousFractions[i] = fractions.get(i);
            }
            return new Metadata(trainFlags, anomalousCounts, anomalousFractions);
        }
    }

    private static final class RunConfig {
        private final String featureName;
        private final int shingleSize;
        private final int numberOfTrees;
        private final int sampleSize;
        private final double timeDecay;

        private RunConfig(String featureName, int shingleSize, int numberOfTrees, int sampleSize, double timeDecay) {
            this.featureName = featureName;
            this.shingleSize = shingleSize;
            this.numberOfTrees = numberOfTrees;
            this.sampleSize = sampleSize;
            this.timeDecay = timeDecay;
        }

        private static List<RunConfig> suite() {
            List<RunConfig> configs = new ArrayList<>();
            for (String feature : new String[] { "evidence", "b25e", "b50e" }) {
                configs.add(new RunConfig(feature, 1, 50, 256, -1.0));
                configs.add(new RunConfig(feature, 4, 50, 256, -1.0));
            }
            configs.add(new RunConfig("b25e", 4, 50, 128, 1.0 / 1000.0));
            configs.add(new RunConfig("b25e", 4, 50, 64, -1.0));
            return configs;
        }
    }

    private static final class LabelSpec {
        private final String name;
        private final int minCount;
        private final double minFraction;

        private LabelSpec(String name, int minCount, double minFraction) {
            this.name = name;
            this.minCount = minCount;
            this.minFraction = minFraction;
        }

        private static List<LabelSpec> suite() {
            return Arrays.asList(new LabelSpec("count_ge_1", 1, -1.0), new LabelSpec("count_ge_2", 2, -1.0),
                    new LabelSpec("count_ge_3", 3, -1.0), new LabelSpec("count_ge_5", 5, -1.0),
                    new LabelSpec("count_ge_10", 10, -1.0), new LabelSpec("count_ge_20", 20, -1.0),
                    new LabelSpec("count_ge_50", 50, -1.0), new LabelSpec("count_ge_100", 100, -1.0),
                    new LabelSpec("count_ge_200", 200, -1.0), new LabelSpec("count_ge_500", 500, -1.0),
                    new LabelSpec("count_ge_1000", 1000, -1.0), new LabelSpec("fraction_ge_0.001", -1, 0.001),
                    new LabelSpec("fraction_ge_0.005", -1, 0.005), new LabelSpec("fraction_ge_0.010", -1, 0.010),
                    new LabelSpec("fraction_ge_0.050", -1, 0.050), new LabelSpec("fraction_ge_0.100", -1, 0.100),
                    new LabelSpec("fraction_ge_0.200", -1, 0.200), new LabelSpec("fraction_ge_0.500", -1, 0.500));
        }

        private boolean[] labels(List<Integer> anomalousCounts, List<Double> anomalousFractions) {
            boolean[] labels = new boolean[anomalousCounts.size()];
            for (int i = 0; i < labels.length; i++) {
                labels[i] = minCount >= 0 ? anomalousCounts.get(i) >= minCount
                        : anomalousFractions.get(i) >= minFraction;
            }
            return labels;
        }
    }

    private static final class Metrics {
        private final int records;
        private final int positives;
        private final int predicted;
        private final int tp;
        private final int fp;
        private final int fn;
        private final int tn;
        private final double precision;
        private final double recall;
        private final double f1;
        private final int labelSegments;
        private final int segmentHits0;
        private final int segmentHits1;
        private final int segmentHits2;
        private final int alertGroups;
        private final int alertGroupsHit;
        private final double alertsPerDay;

        private Metrics(int records, int positives, int predicted, int tp, int fp, int fn, int tn, double precision,
                double recall, double f1, int labelSegments, int segmentHits0, int segmentHits1, int segmentHits2,
                int alertGroups, int alertGroupsHit, double alertsPerDay) {
            this.records = records;
            this.positives = positives;
            this.predicted = predicted;
            this.tp = tp;
            this.fp = fp;
            this.fn = fn;
            this.tn = tn;
            this.precision = precision;
            this.recall = recall;
            this.f1 = f1;
            this.labelSegments = labelSegments;
            this.segmentHits0 = segmentHits0;
            this.segmentHits1 = segmentHits1;
            this.segmentHits2 = segmentHits2;
            this.alertGroups = alertGroups;
            this.alertGroupsHit = alertGroupsHit;
            this.alertsPerDay = alertsPerDay;
        }

        private static Metrics from(boolean[] labels, boolean[] predictions, int bucketSeconds) {
            int positives = 0;
            int predicted = 0;
            int tp = 0;
            int fp = 0;
            int fn = 0;
            int tn = 0;
            for (int i = 0; i < labels.length; i++) {
                if (labels[i]) {
                    ++positives;
                }
                if (predictions[i]) {
                    ++predicted;
                }
                if (labels[i] && predictions[i]) {
                    ++tp;
                } else if (!labels[i] && predictions[i]) {
                    ++fp;
                } else if (labels[i]) {
                    ++fn;
                } else {
                    ++tn;
                }
            }
            double precision = tp + fp == 0 ? 0.0 : tp / (double) (tp + fp);
            double recall = tp + fn == 0 ? 0.0 : tp / (double) (tp + fn);
            double f1 = precision + recall == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);
            int[] segmentHits = segmentHits(labels, predictions);
            int[] alertGroupStats = alertGroupStats(labels, predictions, 2);
            double days = labels.length * bucketSeconds / 86400.0;
            return new Metrics(labels.length, positives, predicted, tp, fp, fn, tn, precision, recall, f1,
                    segmentHits[0], segmentHits[1], segmentHits[2], segmentHits[3], alertGroupStats[0],
                    alertGroupStats[1], days == 0.0 ? 0.0 : predicted / days);
        }

        private Result toResult(String dataset, RunConfig runConfig, LabelSpec labelSpec, double alertRate,
                double threshold) {
            return new Result(dataset, runConfig, labelSpec.name, alertRate, threshold, records, positives, predicted,
                    tp, fp, fn, tn, precision, recall, f1, labelSegments, segmentHits0, segmentHits1, segmentHits2,
                    alertGroups, alertGroupsHit, alertsPerDay);
        }

        private static int[] segmentHits(boolean[] labels, boolean[] predictions) {
            int segments = 0;
            int hits0 = 0;
            int hits1 = 0;
            int hits2 = 0;
            int i = 0;
            while (i < labels.length) {
                if (!labels[i]) {
                    ++i;
                    continue;
                }
                int start = i;
                while (i < labels.length && labels[i]) {
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

        private static int[] alertGroupStats(boolean[] labels, boolean[] predictions, int mergeGap) {
            int groups = 0;
            int hits = 0;
            int i = 0;
            while (i < predictions.length) {
                if (!predictions[i]) {
                    ++i;
                    continue;
                }
                int start = i;
                int last = i;
                ++i;
                int gap = 0;
                while (i < predictions.length && gap <= mergeGap) {
                    if (predictions[i]) {
                        last = i;
                        gap = 0;
                    } else {
                        ++gap;
                    }
                    ++i;
                }
                ++groups;
                if (hasPrediction(labels, Math.max(0, start - 2), Math.min(labels.length - 1, last + 2), 0)) {
                    ++hits;
                }
            }
            return new int[] { groups, hits };
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
        private final String labelSpec;
        private final double alertRate;
        private final double threshold;
        private final int records;
        private final int positives;
        private final int predicted;
        private final int tp;
        private final int fp;
        private final int fn;
        private final int tn;
        private final double precision;
        private final double recall;
        private final double f1;
        private final int labelSegments;
        private final int segmentHits0;
        private final int segmentHits1;
        private final int segmentHits2;
        private final int alertGroups;
        private final int alertGroupsHit;
        private final double alertsPerDay;

        private Result(String dataset, RunConfig runConfig, String labelSpec, double alertRate, double threshold,
                int records, int positives, int predicted, int tp, int fp, int fn, int tn, double precision,
                double recall, double f1, int labelSegments, int segmentHits0, int segmentHits1, int segmentHits2,
                int alertGroups, int alertGroupsHit, double alertsPerDay) {
            this.dataset = dataset;
            this.runConfig = runConfig;
            this.labelSpec = labelSpec;
            this.alertRate = alertRate;
            this.threshold = threshold;
            this.records = records;
            this.positives = positives;
            this.predicted = predicted;
            this.tp = tp;
            this.fp = fp;
            this.fn = fn;
            this.tn = tn;
            this.precision = precision;
            this.recall = recall;
            this.f1 = f1;
            this.labelSegments = labelSegments;
            this.segmentHits0 = segmentHits0;
            this.segmentHits1 = segmentHits1;
            this.segmentHits2 = segmentHits2;
            this.alertGroups = alertGroups;
            this.alertGroupsHit = alertGroupsHit;
            this.alertsPerDay = alertsPerDay;
        }

        private static String header() {
            return String.join(",", Arrays.asList("dataset", "feature", "shingle_size", "trees", "sample_size",
                    "time_decay", "label_spec", "label_rate", "target_alert_rate", "actual_alert_rate", "threshold",
                    "records", "positives", "predicted", "tp", "fp", "fn", "tn", "precision", "recall", "f1",
                    "label_segments", "segments_hit_w0", "segment_recall_w0", "segments_hit_w1",
                    "segment_recall_w1", "segments_hit_w2", "segment_recall_w2", "alert_groups", "alert_groups_hit",
                    "alert_group_precision", "alerts_per_day"));
        }

        private String toCsv() {
            return String.format(Locale.ROOT,
                    "%s,%s,%d,%d,%d,%s,%s,%.8f,%.4f,%.8f,%.8f,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d,%d,%.6f,%d,%.6f,%d,%.6f,%d,%d,%.6f,%.4f",
                    dataset, runConfig.featureName, runConfig.shingleSize, runConfig.numberOfTrees, runConfig.sampleSize,
                    runConfig.timeDecay > 0.0 ? String.format(Locale.ROOT, "%.8f", runConfig.timeDecay) : "default",
                    labelSpec, records == 0 ? 0.0 : positives / (double) records, alertRate,
                    records == 0 ? 0.0 : predicted / (double) records, threshold, records, positives, predicted, tp, fp,
                    fn, tn, precision, recall, f1, labelSegments, segmentHits0,
                    labelSegments == 0 ? 0.0 : segmentHits0 / (double) labelSegments, segmentHits1,
                    labelSegments == 0 ? 0.0 : segmentHits1 / (double) labelSegments, segmentHits2,
                    labelSegments == 0 ? 0.0 : segmentHits2 / (double) labelSegments, alertGroups, alertGroupsHit,
                    alertGroups == 0 ? 0.0 : alertGroupsHit / (double) alertGroups, alertsPerDay);
        }
    }

    private static final class Config {
        private final List<String> datasets;
        private final String featurePrefix;
        private final String metadataPrefix;
        private final String output;
        private final int outputAfter;
        private final int bucketSeconds;
        private final double[] alertRates;
        private final long seed;
        private final int top;

        private Config(List<String> datasets, String featurePrefix, String metadataPrefix, String output, int outputAfter,
                int bucketSeconds, double[] alertRates, long seed, int top) {
            this.datasets = datasets;
            this.featurePrefix = featurePrefix;
            this.metadataPrefix = metadataPrefix;
            this.output = output;
            this.outputAfter = outputAfter;
            this.bucketSeconds = bucketSeconds;
            this.alertRates = alertRates;
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
                    values.getOrDefault("feature-prefix", "benchmark-results/log-ad/production_bucket_features"),
                    values.getOrDefault("metadata-prefix", "benchmark-results/log-ad/bucket_metadata"),
                    values.getOrDefault("output", "benchmark-results/log-ad/rcf_production_alert_results.csv"),
                    Integer.parseInt(values.getOrDefault("output-after", "32")),
                    Integer.parseInt(values.getOrDefault("bucket-seconds", "600")),
                    splitDoubles(values.getOrDefault("alert-rates", "0.005,0.01,0.02")),
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

        private static double[] splitDoubles(String csv) {
            String[] parts = csv.split(",");
            double[] result = new double[parts.length];
            for (int i = 0; i < parts.length; i++) {
                result[i] = Double.parseDouble(parts[i].trim());
            }
            return result;
        }
    }
}
