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
 * Focused TRCF tuning runner over precomputed log bucket feature CSVs.
 */
public final class TrcfBucketTuningBenchmark {

    private TrcfBucketTuningBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        List<Result> results = new ArrayList<>();
        for (String dataset : config.datasets) {
            DatasetFeatures features = DatasetFeatures.load(config.featurePrefix, dataset);
            List<RunConfig> runConfigs = RunConfig.suite(config.suite, dataset);
            System.out.printf(Locale.ROOT, "dataset=%s suite=%s configs=%d rows=%d train=%d%n", dataset, config.suite,
                    runConfigs.size(), features.size(), features.trainCount);
            int count = 0;
            for (RunConfig runConfig : runConfigs) {
                ++count;
                if (count == 1 || count % 10 == 0) {
                    System.out.printf(Locale.ROOT, "  %s %d/%d %s%n", dataset, count, runConfigs.size(),
                            runConfig.summary());
                }
                FeatureData featureData = features.featureData(runConfig.representation);
                results.addAll(runTrcf(dataset, featureData, runConfig, config));
                System.out.printf(Locale.ROOT, "  finished %s %d/%d%n", dataset, count, runConfigs.size());
            }
        }
        writeResults(config.output, results);
        printTop(results, config.top);
    }

    private static List<Result> runTrcf(String dataset, FeatureData data, RunConfig runConfig, Config config) {
        int outputAfter = Math.max(1, Math.min(config.outputAfter, Math.max(1, data.trainCount / 2)));
        ThresholdedRandomCutForest.Builder<?> builder = ThresholdedRandomCutForest.builder()
                .dimensions(data.values[0].length * runConfig.shingleSize).shingleSize(runConfig.shingleSize)
                .sampleSize(runConfig.sampleSize).numberOfTrees(runConfig.numberOfTrees).randomSeed(config.seed)
                .outputAfter(outputAfter).transformMethod(runConfig.transformMethod)
                .scoringStrategy(runConfig.scoringStrategy).anomalyRate(config.anomalyRate)
                .autoAdjust(runConfig.autoAdjust).zFactor(runConfig.zFactor);
        if (runConfig.timeDecay > 0.0) {
            builder.timeDecay(runConfig.timeDecay);
        }
        ThresholdedRandomCutForest forest = builder.build();
        forest.setLowerThreshold(runConfig.lowerThreshold);

        List<Double> calibrationScores = new ArrayList<>();
        List<ScoredPrediction> testRows = new ArrayList<>();
        for (int i = 0; i < data.values.length; i++) {
            AnomalyDescriptor result = forest.process(data.values[i], i);
            double score = result.getRCFScore();
            if (i < data.trainCount) {
                if (i >= outputAfter && Double.isFinite(score)) {
                    calibrationScores.add(score);
                }
            } else {
                testRows.add(new ScoredPrediction(data.labels[i], score, result.getAnomalyGrade() > 0.0));
            }
        }

        List<Result> results = new ArrayList<>();
        results.add(Metrics.from(testRows, prediction -> prediction.gradePrediction)
                .toResult(dataset, runConfig, "grade>0", Double.NaN));
        for (double alertRate : alertRates(dataset)) {
            double threshold = quantile(calibrationScores, 1.0 - alertRate);
            results.add(Metrics.from(testRows, prediction -> prediction.score >= threshold)
                    .toResult(dataset, runConfig, String.format(Locale.ROOT, "score_train_quantile_alert_%.3f", alertRate),
                            threshold));
        }
        return results;
    }

    private interface PredictionRule {
        boolean predict(ScoredPrediction prediction);
    }

    private static double quantile(List<Double> values, double quantile) {
        if (values.isEmpty()) {
            return Double.POSITIVE_INFINITY;
        }
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        double clipped = Math.max(0.0, Math.min(1.0, quantile));
        int index = (int) Math.floor(clipped * (sorted.size() - 1));
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
        private final String dataset;
        private final FeatureData a;
        private final FeatureData b;
        private final FeatureData c;
        private final int trainCount;

        private DatasetFeatures(String dataset, FeatureData a, FeatureData b, FeatureData c) {
            this.dataset = dataset;
            this.a = a;
            this.b = b;
            this.c = c;
            this.trainCount = a.trainCount;
            if (a.size() != b.size() || a.size() != c.size() || a.trainCount != b.trainCount
                    || a.trainCount != c.trainCount) {
                throw new IllegalArgumentException("feature files are not aligned for " + dataset);
            }
        }

        private static DatasetFeatures load(String prefix, String dataset) throws IOException {
            return new DatasetFeatures(dataset, FeatureData.load(prefix + "_" + dataset + "_a.csv", "A"),
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
                throw new IllegalArgumentException("unknown representation " + representation + " for " + dataset);
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
                double unique = a.values[i][1];
                double rare = a.values[i][2];
                double oov = a.values[i][3];
                double maxRarity = a.values[i][4];
                double avgRarity = a.values[i][6];
                double entropy = a.values[i][7];
                double dominantRatio = a.values[i][8];
                values[i][55] = rare / total;
                values[i][56] = oov / total;
                values[i][57] = unique / total;
                values[i][58] = dominantRatio;
                values[i][59] = entropy;
                values[i][60] = maxRarity;
                values[i][61] = avgRarity;
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
                    if (parts.length < 4) {
                        continue;
                    }
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
    }

    private static final class RunConfig {
        private final String experiment;
        private final String representation;
        private final int shingleSize;
        private final int numberOfTrees;
        private final int sampleSize;
        private final double timeDecay;
        private final double zFactor;
        private final double lowerThreshold;
        private final boolean autoAdjust;
        private final ScoringStrategy scoringStrategy;
        private final TransformMethod transformMethod;

        private RunConfig(String experiment, String representation, int shingleSize, int numberOfTrees, int sampleSize,
                double timeDecay, double zFactor, double lowerThreshold, boolean autoAdjust,
                ScoringStrategy scoringStrategy, TransformMethod transformMethod) {
            this.experiment = experiment;
            this.representation = representation;
            this.shingleSize = shingleSize;
            this.numberOfTrees = numberOfTrees;
            this.sampleSize = sampleSize;
            this.timeDecay = timeDecay;
            this.zFactor = zFactor;
            this.lowerThreshold = lowerThreshold;
            this.autoAdjust = autoAdjust;
            this.scoringStrategy = scoringStrategy;
            this.transformMethod = transformMethod;
        }

        private static List<RunConfig> suite(String suite, String dataset) {
            switch (suite) {
            case "threshold":
                return thresholdSuite();
            case "threshold_quick":
                return thresholdQuickSuite();
            case "threshold_example":
                return thresholdExampleSuite();
            case "threshold_auto":
                return thresholdAutoSuite();
            case "strategy":
                return strategySuite();
            case "strategy_fast":
                return strategyFastSuite();
            case "shingle":
                return shingleSuite();
            case "model":
                return modelSuite();
            case "all":
                List<RunConfig> all = new ArrayList<>();
                all.addAll(thresholdSuite());
                all.addAll(strategySuite());
                all.addAll(shingleSuite());
                all.addAll(modelSuite());
                return all;
            default:
                throw new IllegalArgumentException("unknown suite " + suite + " for " + dataset);
            }
        }

        private static List<RunConfig> thresholdSuite() {
            List<RunConfig> configs = new ArrayList<>();
            for (double z : new double[] { 3.0, 2.75, 2.5, 2.25, 2.0 }) {
                for (double lower : new double[] { 0.8, 0.6, 0.4, 0.2 }) {
                    for (boolean auto : new boolean[] { false, true }) {
                        configs.add(new RunConfig("threshold_grid", "B", 4, 50, 256, -1.0, z, lower, auto,
                                ScoringStrategy.MULTI_MODE_RECALL, TransformMethod.NORMALIZE));
                    }
                }
            }
            return configs;
        }

        private static List<RunConfig> thresholdQuickSuite() {
            List<RunConfig> configs = new ArrayList<>();
            for (double z : new double[] { 3.0, 2.5, 2.0 }) {
                for (double lower : new double[] { 0.8, 0.4, 0.2 }) {
                    configs.add(new RunConfig("threshold_quick", "B", 4, 50, 256, -1.0, z, lower, true,
                            ScoringStrategy.MULTI_MODE_RECALL, TransformMethod.NORMALIZE));
                }
            }
            configs.add(new RunConfig("threshold_quick_auto_false", "B", 4, 50, 256, -1.0, 2.0, 0.2, false,
                    ScoringStrategy.MULTI_MODE_RECALL, TransformMethod.NORMALIZE));
            return configs;
        }

        private static List<RunConfig> thresholdExampleSuite() {
            List<RunConfig> configs = new ArrayList<>();
            for (String representation : new String[] { "B", "C" }) {
                for (double z : new double[] { 3.0, 2.5, 2.0 }) {
                    for (double lower : new double[] { 0.8, 0.4, 0.2 }) {
                        configs.add(new RunConfig("threshold_example", representation, 1, 100, 256, -1.0, z, lower,
                                true, ScoringStrategy.EXPECTED_INVERSE_DEPTH, TransformMethod.NONE));
                    }
                }
            }
            return configs;
        }

        private static List<RunConfig> thresholdAutoSuite() {
            List<RunConfig> configs = new ArrayList<>();
            for (String representation : new String[] { "B", "C" }) {
                for (boolean auto : new boolean[] { false, true }) {
                    configs.add(new RunConfig("threshold_auto", representation, 1, 100, 256, -1.0, 2.0, 0.2, auto,
                            ScoringStrategy.EXPECTED_INVERSE_DEPTH, TransformMethod.NONE));
                }
            }
            return configs;
        }

        private static List<RunConfig> strategySuite() {
            List<RunConfig> configs = new ArrayList<>();
            for (String representation : new String[] { "B", "C" }) {
                for (ScoringStrategy scoringStrategy : new ScoringStrategy[] { ScoringStrategy.EXPECTED_INVERSE_DEPTH,
                        ScoringStrategy.DISTANCE, ScoringStrategy.MULTI_MODE_RECALL }) {
                    configs.add(new RunConfig("scoring_strategy", representation, 4, 50, 256, -1.0, 2.0, 0.2, true,
                            scoringStrategy, TransformMethod.NORMALIZE));
                }
            }
            for (TransformMethod transformMethod : new TransformMethod[] { TransformMethod.NONE,
                    TransformMethod.NORMALIZE }) {
                configs.add(new RunConfig("transform", "B", 4, 50, 256, -1.0, 2.0, 0.2, true,
                        ScoringStrategy.MULTI_MODE_RECALL, transformMethod));
                configs.add(new RunConfig("transform_augmented", "BR", 4, 50, 256, -1.0, 2.0, 0.2, true,
                        ScoringStrategy.MULTI_MODE_RECALL, transformMethod));
            }
            return configs;
        }

        private static List<RunConfig> strategyFastSuite() {
            List<RunConfig> configs = new ArrayList<>();
            for (String representation : new String[] { "B", "C" }) {
                for (ScoringStrategy scoringStrategy : new ScoringStrategy[] { ScoringStrategy.EXPECTED_INVERSE_DEPTH,
                        ScoringStrategy.DISTANCE, ScoringStrategy.MULTI_MODE_RECALL }) {
                    configs.add(new RunConfig("strategy_fast", representation, 1, 100, 256, -1.0, 2.0, 0.2, false,
                            scoringStrategy, TransformMethod.NONE));
                }
            }
            return configs;
        }

        private static List<RunConfig> shingleSuite() {
            List<RunConfig> configs = new ArrayList<>();
            for (int shingle : new int[] { 1, 4, 6, 12 }) {
                configs.add(tuned("shingle", "A", shingle, 50, 256, -1.0));
            }
            for (int shingle : new int[] { 1, 2, 4 }) {
                configs.add(tuned("shingle", "B", shingle, 50, 256, -1.0));
                configs.add(tuned("shingle", "C", shingle, 50, 256, -1.0));
            }
            for (int shingle : new int[] { 2, 4, 6 }) {
                configs.add(tuned("shingle_top25", "B25", shingle, 50, 256, -1.0));
            }
            for (int shingle : new int[] { 2, 4 }) {
                configs.add(tuned("shingle_augmented", "BR", shingle, 50, 256, -1.0));
            }
            return configs;
        }

        private static List<RunConfig> modelSuite() {
            List<RunConfig> configs = new ArrayList<>();
            for (int sampleSize : new int[] { 64, 128, 256, 512 }) {
                configs.add(tuned("sample_size", "B", 4, 50, sampleSize, -1.0));
            }
            for (int sampleSize : new int[] { 128, 256 }) {
                configs.add(tuned("trees_100", "B", 4, 100, sampleSize, -1.0));
            }
            for (double timeDecay : new double[] { 1.0 / 500.0, 1.0 / 1000.0, 1.0 / 2000.0 }) {
                configs.add(tuned("time_decay", "B", 4, 50, 128, timeDecay));
            }
            return configs;
        }

        private static RunConfig tuned(String experiment, String representation, int shingleSize, int numberOfTrees,
                int sampleSize, double timeDecay) {
            return new RunConfig(experiment, representation, shingleSize, numberOfTrees, sampleSize, timeDecay, 2.0,
                    0.2, true, ScoringStrategy.MULTI_MODE_RECALL, TransformMethod.NORMALIZE);
        }

        private String summary() {
            return String.format(Locale.ROOT,
                    "%s rep=%s shingle=%d trees=%d sample=%d decay=%s z=%.2f lower=%.2f auto=%s scoring=%s transform=%s",
                    experiment, representation, shingleSize, numberOfTrees, sampleSize,
                    timeDecay > 0.0 ? Double.toString(timeDecay) : "default", zFactor, lowerThreshold, autoAdjust,
                    scoringStrategy, transformMethod);
        }
    }

    private static final class ScoredPrediction {
        private final boolean label;
        private final double score;
        private final boolean gradePrediction;

        private ScoredPrediction(boolean label, double score, boolean gradePrediction) {
            this.label = label;
            this.score = score;
            this.gradePrediction = gradePrediction;
        }
    }

    private static final class Metrics {
        private final List<ScoredPrediction> rows;
        private final boolean[] predictions;
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

        private Metrics(List<ScoredPrediction> rows, boolean[] predictions, long positives, long predicted, long tp,
                long fp, long fn, long tn, double precision, double recall, double f1, int k, double precisionAtK,
                double recallAtK, int segments, int segmentHits0, int segmentHits1, int segmentHits2) {
            this.rows = rows;
            this.predictions = predictions;
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

        private static Metrics from(List<ScoredPrediction> rows, PredictionRule rule) {
            boolean[] predictions = new boolean[rows.size()];
            long positives = 0;
            long predicted = 0;
            long tp = 0;
            long fp = 0;
            long fn = 0;
            long tn = 0;
            for (int i = 0; i < rows.size(); i++) {
                ScoredPrediction row = rows.get(i);
                boolean prediction = rule.predict(row);
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
            long hits = topKHits(rows, k);
            int[] segmentHits = segmentHits(rows, predictions);
            return new Metrics(rows, predictions, positives, predicted, tp, fp, fn, tn, precision, recall, f1, k,
                    k == 0 ? 0.0 : hits / (double) k, positives == 0 ? 0.0 : hits / (double) positives,
                    segmentHits[0], segmentHits[1], segmentHits[2], segmentHits[3]);
        }

        private Result toResult(String dataset, RunConfig runConfig, String thresholdType, double threshold) {
            return new Result(dataset, runConfig.experiment, runConfig.representation, thresholdType, threshold,
                    runConfig.shingleSize, runConfig.numberOfTrees, runConfig.sampleSize, runConfig.timeDecay,
                    runConfig.zFactor, runConfig.lowerThreshold, runConfig.autoAdjust, runConfig.scoringStrategy,
                    runConfig.transformMethod, rows.size(), positives, predicted, tp, fp, fn, tn, precision, recall,
                    f1, k, precisionAtK, recallAtK, segments, segmentHits0, segmentHits1, segmentHits2);
        }

        private static long topKHits(List<ScoredPrediction> rows, int k) {
            if (k <= 0) {
                return 0;
            }
            List<ScoredPrediction> sorted = new ArrayList<>(rows);
            sorted.sort((a, b) -> Double.compare(b.score, a.score));
            long hits = 0;
            for (int i = 0; i < Math.min(k, sorted.size()); i++) {
                if (sorted.get(i).label) {
                    ++hits;
                }
            }
            return hits;
        }

        private static int[] segmentHits(List<ScoredPrediction> rows, boolean[] predictions) {
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
        private final String experiment;
        private final String representation;
        private final String thresholdType;
        private final double threshold;
        private final int shingleSize;
        private final int numberOfTrees;
        private final int sampleSize;
        private final double timeDecay;
        private final double zFactor;
        private final double lowerThreshold;
        private final boolean autoAdjust;
        private final ScoringStrategy scoringStrategy;
        private final TransformMethod transformMethod;
        private final int testRecords;
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

        private Result(String dataset, String experiment, String representation, String thresholdType, double threshold,
                int shingleSize, int numberOfTrees, int sampleSize, double timeDecay, double zFactor,
                double lowerThreshold, boolean autoAdjust, ScoringStrategy scoringStrategy,
                TransformMethod transformMethod, int testRecords, long positives, long predicted, long tp, long fp,
                long fn, long tn, double precision, double recall, double f1, int k, double precisionAtK,
                double recallAtK, int segments, int segmentHits0, int segmentHits1, int segmentHits2) {
            this.dataset = dataset;
            this.experiment = experiment;
            this.representation = representation;
            this.thresholdType = thresholdType;
            this.threshold = threshold;
            this.shingleSize = shingleSize;
            this.numberOfTrees = numberOfTrees;
            this.sampleSize = sampleSize;
            this.timeDecay = timeDecay;
            this.zFactor = zFactor;
            this.lowerThreshold = lowerThreshold;
            this.autoAdjust = autoAdjust;
            this.scoringStrategy = scoringStrategy;
            this.transformMethod = transformMethod;
            this.testRecords = testRecords;
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
            return String.join(",", Arrays.asList("dataset", "experiment", "representation", "threshold_type",
                    "threshold", "shingle_size", "trees", "sample_size", "time_decay", "z_factor",
                    "lower_threshold", "auto_adjust", "scoring_strategy", "transform_method", "test_records",
                    "positives", "predicted", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "k",
                    "precision_at_k", "recall_at_k", "segments", "segments_hit_w0", "segment_recall_w0",
                    "segments_hit_w1", "segment_recall_w1", "segments_hit_w2", "segment_recall_w2"));
        }

        private String toCsv() {
            return String.format(Locale.ROOT,
                    "%s,%s,%s,%s,%.8f,%d,%d,%d,%s,%.4f,%.4f,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%d,%d,%.6f,%d,%.6f,%d,%.6f",
                    dataset, experiment, representation, thresholdType, threshold, shingleSize, numberOfTrees,
                    sampleSize, timeDecay > 0.0 ? String.format(Locale.ROOT, "%.8f", timeDecay) : "default", zFactor,
                    lowerThreshold, autoAdjust, scoringStrategy, transformMethod, testRecords, positives, predicted, tp,
                    fp, fn, tn, precision, recall, f1, k, precisionAtK, recallAtK, segments, segmentHits0,
                    segments == 0 ? 0.0 : segmentHits0 / (double) segments, segmentHits1,
                    segments == 0 ? 0.0 : segmentHits1 / (double) segments, segmentHits2,
                    segments == 0 ? 0.0 : segmentHits2 / (double) segments);
        }
    }

    private static final class Config {
        private final List<String> datasets;
        private final String featurePrefix;
        private final String output;
        private final String suite;
        private final int outputAfter;
        private final double anomalyRate;
        private final long seed;
        private final int top;

        private Config(List<String> datasets, String featurePrefix, String output, String suite, int outputAfter,
                double anomalyRate, long seed, int top) {
            this.datasets = datasets;
            this.featurePrefix = featurePrefix;
            this.output = output;
            this.suite = suite;
            this.outputAfter = outputAfter;
            this.anomalyRate = anomalyRate;
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
                    values.getOrDefault("output", "benchmark-results/log-ad/trcf_bucket_tuning.csv"),
                    values.getOrDefault("suite", "threshold"),
                    Integer.parseInt(values.getOrDefault("output-after", "32")),
                    Double.parseDouble(values.getOrDefault("anomaly-rate", "0.02")),
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
