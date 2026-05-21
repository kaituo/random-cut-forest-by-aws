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

import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.config.ScoringStrategy;

/**
 * Measures whether TRCF anomalyRate controls actual grade-positive rate.
 */
public final class TrcfAnomalyRateCalibrationBenchmark {

    private TrcfAnomalyRateCalibrationBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        List<Result> results = new ArrayList<>();
        for (String dataset : config.datasets) {
            Metadata metadata = Metadata.load(config.metadataPrefix + "_" + dataset + ".csv");
            for (String featureName : config.features) {
                FeatureData data = FeatureData.load(config.featurePrefix + "_" + dataset + "_" + featureName + ".csv");
                data.validate(metadata);
                List<RunConfig> runConfigs = RunConfig.suite(config.suite, featureName);
                System.out.printf(Locale.ROOT, "dataset=%s feature=%s suite=%s configs=%d rows=%d train=%d%n", dataset,
                        featureName, config.suite, runConfigs.size(), data.size(), data.trainCount);
                int count = 0;
                for (RunConfig runConfig : runConfigs) {
                    ++count;
                    if (count == 1 || count % 10 == 0) {
                        System.out.printf(Locale.ROOT, "  %s %s %d/%d %s%n", dataset, featureName, count,
                                runConfigs.size(), runConfig.summary());
                    }
                    results.addAll(run(dataset, data, metadata, runConfig, config));
                }
            }
        }
        writeResults(config.output, results);
        printCalibration(results, config.top);
    }

    private static List<Result> run(String dataset, FeatureData data, Metadata metadata, RunConfig runConfig,
            Config config) {
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder()
                .dimensions(data.values[0].length * runConfig.shingleSize).shingleSize(runConfig.shingleSize)
                .numberOfTrees(runConfig.numberOfTrees).sampleSize(runConfig.sampleSize).randomSeed(config.seed)
                .outputAfter(config.outputAfter).anomalyRate(runConfig.anomalyRate).autoAdjust(runConfig.autoAdjust)
                .zFactor(runConfig.zFactor).transformMethod(runConfig.transformMethod)
                .scoringStrategy(ScoringStrategy.EXPECTED_INVERSE_DEPTH).build();

        boolean[] predictions = new boolean[data.size() - data.trainCount];
        double[] scores = new double[predictions.length];
        double[] thresholds = new double[predictions.length];
        int predictionIndex = 0;
        for (int i = 0; i < data.values.length; i++) {
            AnomalyDescriptor result = forest.process(data.values[i], i);
            if (i >= data.trainCount) {
                predictions[predictionIndex] = result.getAnomalyGrade() > 0.0;
                scores[predictionIndex] = result.getRCFScore();
                thresholds[predictionIndex] = result.getThreshold();
                ++predictionIndex;
            }
        }

        int predicted = 0;
        for (boolean prediction : predictions) {
            if (prediction) {
                ++predicted;
            }
        }
        double actualGradeRate = predictions.length == 0 ? 0.0 : predicted / (double) predictions.length;
        List<Result> results = new ArrayList<>();
        for (LabelSpec labelSpec : LabelSpec.suite(dataset)) {
            boolean[] labels = labelSpec.labels(metadata, data.trainCount);
            results.add(Metrics.from(labels, predictions, scores, config.bucketSeconds).toResult(dataset, runConfig,
                    labelSpec, actualGradeRate, mean(thresholds), mean(scores)));
        }
        return results;
    }

    private static double mean(double[] values) {
        if (values.length == 0) {
            return 0.0;
        }
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
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

    private static void printCalibration(List<Result> results, int top) {
        System.out.println(Result.header());
        List<Result> calibrationRows = new ArrayList<>();
        for (Result result : results) {
            if ("count_ge_1".equals(result.labelSpec)) {
                calibrationRows.add(result);
            }
        }
        calibrationRows.sort((a, b) -> Double.compare(Math.abs(a.actualGradeRate - a.runConfig.anomalyRate),
                Math.abs(b.actualGradeRate - b.runConfig.anomalyRate)));
        for (int i = 0; i < Math.min(top, calibrationRows.size()); i++) {
            System.out.println(calibrationRows.get(i).toCsv());
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
        private final double anomalyRate;
        private final double zFactor;
        private final boolean autoAdjust;
        private final TransformMethod transformMethod;

        private RunConfig(String featureName, int shingleSize, int numberOfTrees, int sampleSize, double anomalyRate,
                double zFactor, boolean autoAdjust, TransformMethod transformMethod) {
            this.featureName = featureName;
            this.shingleSize = shingleSize;
            this.numberOfTrees = numberOfTrees;
            this.sampleSize = sampleSize;
            this.anomalyRate = anomalyRate;
            this.zFactor = zFactor;
            this.autoAdjust = autoAdjust;
            this.transformMethod = transformMethod;
        }

        private static List<RunConfig> suite(String suite, String featureName) {
            List<RunConfig> configs = new ArrayList<>();
            if ("quick".equals(suite)) {
                for (int shingle : new int[] { 1, 4 }) {
                    for (double anomalyRate : new double[] { 0.001, 0.002, 0.005, 0.01, 0.02 }) {
                        for (double zFactor : new double[] { 3.0, 2.5, 2.0 }) {
                            for (boolean autoAdjust : new boolean[] { true, false }) {
                                configs.add(new RunConfig(featureName, shingle, 100, 256, anomalyRate, zFactor,
                                        autoAdjust, TransformMethod.NONE));
                            }
                        }
                    }
                }
                return configs;
            }
            if ("transform_check".equals(suite)) {
                for (TransformMethod transformMethod : new TransformMethod[] { TransformMethod.NONE,
                        TransformMethod.DIFFERENCE, TransformMethod.NORMALIZE_DIFFERENCE }) {
                    for (double anomalyRate : new double[] { 0.001, 0.002, 0.005, 0.01, 0.02 }) {
                        for (boolean autoAdjust : new boolean[] { true, false }) {
                            configs.add(new RunConfig(featureName, 4, 100, 256, anomalyRate, 2.0, autoAdjust,
                                    transformMethod));
                        }
                    }
                }
                return configs;
            }
            if ("rate_control".equals(suite)) {
                for (int shingle : new int[] { 1, 4 }) {
                    for (double anomalyRate : new double[] { 0.001, 0.002, 0.005, 0.01, 0.02 }) {
                        configs.add(new RunConfig(featureName, shingle, 100, 256, anomalyRate, 3.0, true,
                                TransformMethod.NONE));
                    }
                }
                return configs;
            }
            throw new IllegalArgumentException("unknown suite " + suite);
        }

        private String summary() {
            return String.format(Locale.ROOT, "shingle=%d trees=%d sample=%d anomalyRate=%.4f z=%.1f auto=%s transform=%s",
                    shingleSize, numberOfTrees, sampleSize, anomalyRate, zFactor, autoAdjust, transformMethod);
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

        private static List<LabelSpec> suite(String dataset) {
            List<LabelSpec> labels = new ArrayList<>();
            labels.add(new LabelSpec("count_ge_1", 1, -1.0));
            labels.add(new LabelSpec("count_ge_10", 10, -1.0));
            if ("bgl".equals(dataset)) {
                labels.add(new LabelSpec("count_ge_500", 500, -1.0));
                labels.add(new LabelSpec("count_ge_1000", 1000, -1.0));
            } else {
                labels.add(new LabelSpec("count_ge_100", 100, -1.0));
                labels.add(new LabelSpec("count_ge_200", 200, -1.0));
                labels.add(new LabelSpec("fraction_ge_0.200", -1, 0.200));
            }
            return labels;
        }

        private boolean[] labels(Metadata metadata, int trainCount) {
            boolean[] labels = new boolean[metadata.anomalousCounts.length - trainCount];
            for (int i = 0; i < labels.length; i++) {
                int index = trainCount + i;
                labels[i] = minCount >= 0 ? metadata.anomalousCounts[index] >= minCount
                        : metadata.anomalousFractions[index] >= minFraction;
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
        private final int segmentHits2;
        private final int alertGroups;
        private final int alertGroupsHit;

        private Metrics(int records, int positives, int predicted, int tp, int fp, int fn, int tn, double precision,
                double recall, double f1, int labelSegments, int segmentHits2, int alertGroups, int alertGroupsHit) {
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
            this.segmentHits2 = segmentHits2;
            this.alertGroups = alertGroups;
            this.alertGroupsHit = alertGroupsHit;
        }

        private static Metrics from(boolean[] labels, boolean[] predictions, double[] scores, int bucketSeconds) {
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
            int[] alertGroups = alertGroupStats(labels, predictions, 2);
            return new Metrics(labels.length, positives, predicted, tp, fp, fn, tn, precision, recall, f1,
                    segmentHits[0], segmentHits[3], alertGroups[0], alertGroups[1]);
        }

        private Result toResult(String dataset, RunConfig runConfig, LabelSpec labelSpec, double actualGradeRate,
                double meanThreshold, double meanScore) {
            return new Result(dataset, runConfig, labelSpec.name, actualGradeRate, meanThreshold, meanScore, records,
                    positives, predicted, tp, fp, fn, tn, precision, recall, f1, labelSegments, segmentHits2,
                    alertGroups, alertGroupsHit);
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
        private final double actualGradeRate;
        private final double meanThreshold;
        private final double meanScore;
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
        private final int segmentHits2;
        private final int alertGroups;
        private final int alertGroupsHit;

        private Result(String dataset, RunConfig runConfig, String labelSpec, double actualGradeRate,
                double meanThreshold, double meanScore, int records, int positives, int predicted, int tp, int fp,
                int fn, int tn, double precision, double recall, double f1, int labelSegments, int segmentHits2,
                int alertGroups, int alertGroupsHit) {
            this.dataset = dataset;
            this.runConfig = runConfig;
            this.labelSpec = labelSpec;
            this.actualGradeRate = actualGradeRate;
            this.meanThreshold = meanThreshold;
            this.meanScore = meanScore;
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
            this.segmentHits2 = segmentHits2;
            this.alertGroups = alertGroups;
            this.alertGroupsHit = alertGroupsHit;
        }

        private static String header() {
            return String.join(",", Arrays.asList("dataset", "feature", "shingle_size", "trees", "sample_size",
                    "transform_method", "scoring_strategy", "configured_anomaly_rate", "actual_grade_rate",
                    "z_factor", "auto_adjust", "mean_threshold", "mean_score", "label_spec", "label_rate",
                    "records", "positives", "predicted", "tp", "fp", "fn", "tn", "precision", "recall", "f1",
                    "label_segments", "segments_hit_w2", "segment_recall_w2", "alert_groups", "alert_groups_hit",
                    "alert_group_precision"));
        }

        private String toCsv() {
            return String.format(Locale.ROOT,
                    "%s,%s,%d,%d,%d,%s,%s,%.4f,%.8f,%.4f,%s,%.8f,%.8f,%s,%.8f,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d,%d,%.6f,%d,%d,%.6f",
                    dataset, runConfig.featureName, runConfig.shingleSize, runConfig.numberOfTrees,
                    runConfig.sampleSize, runConfig.transformMethod, ScoringStrategy.EXPECTED_INVERSE_DEPTH,
                    runConfig.anomalyRate, actualGradeRate, runConfig.zFactor, runConfig.autoAdjust, meanThreshold,
                    meanScore, labelSpec, records == 0 ? 0.0 : positives / (double) records, records, positives,
                    predicted, tp, fp, fn, tn, precision, recall, f1, labelSegments, segmentHits2,
                    labelSegments == 0 ? 0.0 : segmentHits2 / (double) labelSegments, alertGroups, alertGroupsHit,
                    alertGroups == 0 ? 0.0 : alertGroupsHit / (double) alertGroups);
        }
    }

    private static final class Config {
        private final List<String> datasets;
        private final List<String> features;
        private final String featurePrefix;
        private final String metadataPrefix;
        private final String output;
        private final String suite;
        private final int outputAfter;
        private final int bucketSeconds;
        private final long seed;
        private final int top;

        private Config(List<String> datasets, List<String> features, String featurePrefix, String metadataPrefix,
                String output, String suite, int outputAfter, int bucketSeconds, long seed, int top) {
            this.datasets = datasets;
            this.features = features;
            this.featurePrefix = featurePrefix;
            this.metadataPrefix = metadataPrefix;
            this.output = output;
            this.suite = suite;
            this.outputAfter = outputAfter;
            this.bucketSeconds = bucketSeconds;
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
                    split(values.getOrDefault("features", "b25e")),
                    values.getOrDefault("feature-prefix", "benchmark-results/log-ad/production_bucket_features"),
                    values.getOrDefault("metadata-prefix", "benchmark-results/log-ad/bucket_metadata"),
                    values.getOrDefault("output", "benchmark-results/log-ad/trcf_anomaly_rate_calibration.csv"),
                    values.getOrDefault("suite", "quick"),
                    Integer.parseInt(values.getOrDefault("output-after", "32")),
                    Integer.parseInt(values.getOrDefault("bucket-seconds", "600")),
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
