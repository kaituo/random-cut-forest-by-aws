/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package com.amazon.randomcutforest.examples.parkservices;

import static java.lang.Math.max;
import static java.lang.Math.min;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.config.ScoringStrategy;

/**
 * Line-level log anomaly benchmark:
 *
 * 1. Parse raw BGL/Thunderbird-style logs into Drain-like template ids. 2.
 * Build template rarity, OOV, transition surprise, and bucket spike scores. 3.
 * Optionally use TRCF as a high-recall bucket context detector. 4. Tune the
 * final line-score threshold on validation buckets and report held-out line
 * precision/recall/F1.
 */
public final class LineLevelLogAnomalyBenchmark {

    private static final Charset RAW_LOG_CHARSET = StandardCharsets.ISO_8859_1;

    private LineLevelLogAnomalyBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        if ("diagnostic".equals(config.thresholdMode)) {
            List<TemplateDiagnostic> results = new ArrayList<>();
            for (String datasetName : config.datasets) {
                results.addAll(evaluateDatasetTemplateDiagnostic(datasetName, config));
            }
            writeTemplateDiagnostics(config.output, results);
            printTemplateDiagnostics(results, config.top);
        } else if ("oracle".equals(config.thresholdMode)) {
            List<OracleResult> results = new ArrayList<>();
            for (String datasetName : config.datasets) {
                results.addAll(evaluateDatasetOracle(datasetName, config));
            }
            writeOracleResults(config.output, results);
            printOracleSummary(results, config.top);
        } else if ("online-oracle".equals(config.thresholdMode)) {
            List<OracleResult> results = new ArrayList<>();
            for (String datasetName : config.datasets) {
                results.addAll(evaluateDatasetOnlineOracle(datasetName, config));
            }
            writeOracleResults(config.output, results);
            printOracleSummary(results, config.top);
        } else if ("adaptive".equals(config.thresholdMode)) {
            List<FdrResult> results = new ArrayList<>();
            for (String datasetName : config.datasets) {
                results.addAll(evaluateDatasetAdaptive(datasetName, config));
            }
            writeFdrResults(config.output, results);
            printFdrSummary(results, config.top);
        } else if ("tail".equals(config.thresholdMode)) {
            List<FdrResult> results = new ArrayList<>();
            for (String datasetName : config.datasets) {
                results.addAll(evaluateDatasetTailExcess(datasetName, config));
            }
            writeFdrResults(config.output, results);
            printFdrSummary(results, config.top);
        } else if ("quantile".equals(config.thresholdMode)) {
            List<FdrResult> results = new ArrayList<>();
            for (String datasetName : config.datasets) {
                results.addAll(evaluateDatasetQuantile(datasetName, config));
            }
            writeFdrResults(config.output, results);
            printFdrSummary(results, config.top);
        } else if ("online".equals(config.thresholdMode)) {
            List<FdrResult> results = new ArrayList<>();
            for (String datasetName : config.datasets) {
                results.addAll(evaluateDatasetOnlineDecayed(datasetName, config));
            }
            writeFdrResults(config.output, results);
            printFdrSummary(results, config.top);
        } else if ("anchored-dynamic".equals(config.thresholdMode) || "anchored_dynamic".equals(config.thresholdMode)) {
            List<FdrResult> results = new ArrayList<>();
            for (String datasetName : config.datasets) {
                results.addAll(evaluateDatasetAnchoredDynamic(datasetName, config));
            }
            writeFdrResults(config.output, results);
            printFdrSummary(results, config.top);
        } else if ("anchored-online".equals(config.thresholdMode) || "anchored_online".equals(config.thresholdMode)) {
            List<FdrResult> results = new ArrayList<>();
            for (String datasetName : config.datasets) {
                results.addAll(evaluateDatasetAnchoredOnline(datasetName, config));
            }
            writeFdrResults(config.output, results);
            printFdrSummary(results, config.top);
        } else if ("streaming-anchored-online".equals(config.thresholdMode)
                || "streaming_anchored_online".equals(config.thresholdMode)
                || "anchored-online-streaming".equals(config.thresholdMode)
                || "anchored_online_streaming".equals(config.thresholdMode)) {
            List<FdrResult> results = new ArrayList<>();
            for (String datasetName : config.datasets) {
                results.add(evaluateDatasetStreamingAnchoredOnline(datasetName, config));
            }
            writeFdrResults(config.output, results);
            printFdrSummary(results, config.top);
        } else if ("estimate-memory-static".equals(config.thresholdMode)
                || "estimate_memory_static".equals(config.thresholdMode)) {
            printStaticMemoryEstimate(config);
        } else if ("predict".equals(config.thresholdMode)) {
            writePredictionDumps(config);
        } else if ("topk".equals(config.thresholdMode)) {
            List<FdrResult> results = new ArrayList<>();
            for (String datasetName : config.datasets) {
                results.addAll(evaluateDatasetTopKPolicy(datasetName, config));
            }
            writeFdrResults(config.output, results);
            printFdrSummary(results, config.top);
        } else if ("fdr".equals(config.thresholdMode)) {
            List<FdrResult> results = new ArrayList<>();
            for (String datasetName : config.datasets) {
                results.addAll(evaluateDatasetFdr(datasetName, config));
            }
            writeFdrResults(config.output, results);
            printFdrSummary(results, config.top);
        } else {
            List<Result> results = new ArrayList<>();
            for (String datasetName : config.datasets) {
                results.addAll(evaluateDatasetSupervised(datasetName, config));
            }
            writeResults(config.output, results);
            printSummary(results, config.top);
        }
    }

    private static void writePredictionDumps(Config config) throws IOException {
        Path path = Paths.get(config.output);
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write(PredictionRecord.header());
            writer.newLine();
            for (String datasetName : config.datasets) {
                writeDatasetPredictions(datasetName, config, writer);
            }
        }
    }

    private static void writeDatasetPredictions(String datasetName, Config config, BufferedWriter writer)
            throws IOException {
        Dataset dataset = loadDataset(datasetName, config);
        int warmupEnd = max(1,
                min(dataset.buckets.size() - 1, (int) Math.floor(config.trainFraction * dataset.buckets.size())));
        int scoreEnd = min(dataset.buckets.size(),
                warmupEnd + max(1, (int) Math.floor(config.predictionFraction * dataset.buckets.size())));
        TrainStats stats = TrainStats.fromRange(dataset, 0, warmupEnd, config);
        ContextSet context = ContextSet.all("none", dataset.buckets.size());
        double[] warmupScores = calibrationLineScores(dataset, stats, context, 0, warmupEnd, config);
        java.util.Arrays.sort(warmupScores);
        System.out.printf(Locale.ROOT,
                "prediction_dump dataset=%s warmup_buckets=[0,%d) score_buckets=[%d,%d) warmup_lines=%d score_lines=%d%n",
                datasetName, warmupEnd, warmupEnd, scoreEnd, dataset.countLines(0, warmupEnd),
                dataset.countLines(warmupEnd, scoreEnd));
        for (double quantile : config.fdrQValues) {
            double threshold = quantile(warmupScores, quantile);
            Metrics metrics = new Metrics(0, 0, 0, 0);
            for (int bucketIndex = warmupEnd; bucketIndex < scoreEnd; bucketIndex++) {
                Bucket bucket = dataset.buckets.get(bucketIndex);
                for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                    double score = lineScore(dataset, stats, context, i, config.scoreMode);
                    boolean label = dataset.labels.values[i] != 0;
                    boolean prediction = score >= threshold;
                    metrics.add(label, prediction);
                    writer.write(new PredictionRecord(datasetName, quantile, threshold, i, bucketIndex,
                            bucket.bucketKey, dataset.entityIds.values[i], dataset.componentIds.values[i],
                            dataset.levelIds.values[i], dataset.eventIds.values[i], score, prediction, label, dataset)
                                    .toCsv());
                    writer.newLine();
                }
            }
            System.out.printf(Locale.ROOT,
                    "prediction_summary dataset=%s q=%.5f threshold=%.8f tp=%d fp=%d fn=%d tn=%d precision=%.6f recall=%.6f f1=%.6f%n",
                    datasetName, quantile, threshold, metrics.tp, metrics.fp, metrics.fn, metrics.tn,
                    metrics.precision(), metrics.recall(), metrics.f1());
        }
    }

    private static List<Result> evaluateDatasetSupervised(String datasetName, Config config) throws IOException {
        Dataset dataset = loadDataset(datasetName, config);
        Split split = Split.from(dataset, config);
        TrainStats stats = TrainStats.from(dataset, split.trainEnd, config);
        List<ContextSet> contexts = contexts(dataset, stats, split.trainEnd, config);
        List<Result> results = new ArrayList<>();
        for (LineScorer scorer : LineScorer.suite()) {
            for (ContextSet context : contexts) {
                if (!config.includeUngated && context.isUngated()) {
                    continue;
                }
                Result result = evaluate(datasetName, dataset, stats, scorer, context, split.trainEnd,
                        split.calibrationEnd, config);
                results.add(result);
                System.out.println(result.toCsv());
            }
        }
        return results;
    }

    private static List<FdrResult> evaluateDatasetAdaptive(String datasetName, Config config) throws IOException {
        Dataset dataset = loadDataset(datasetName, config);
        Split split = Split.from(dataset, config);
        TrainStats stats = statsForSplit(dataset, split, config);
        List<ContextSet> contexts = contexts(dataset, stats, split.trainEnd, config);
        List<FdrResult> results = new ArrayList<>();
        for (ContextSet context : contexts) {
            double contextRecall = contextLineRecall(dataset, context, split.calibrationEnd, dataset.buckets.size());
            for (double q : config.fdrQValues) {
                FdrResult rolling = evaluateRollingLineBh(datasetName, dataset, stats, context, q, split.calibrationEnd,
                        dataset.buckets.size(), contextRecall, config, false);
                results.add(rolling);
                System.out.println(rolling.toCsv());
                FdrResult conditional = evaluateRollingLineBh(datasetName, dataset, stats, context, q,
                        split.calibrationEnd, dataset.buckets.size(), contextRecall, config, true);
                results.add(conditional);
                System.out.println(conditional.toCsv());
                FdrResult spike = evaluateRollingTemplateSpikeBh(datasetName, dataset, context, q, split.calibrationEnd,
                        dataset.buckets.size(), contextRecall, config);
                results.add(spike);
                System.out.println(spike.toCsv());
                FdrResult seedExpand = evaluateSeedExpand(datasetName, dataset, stats, context, q, split.calibrationEnd,
                        dataset.buckets.size(), contextRecall, config);
                results.add(seedExpand);
                System.out.println(seedExpand.toCsv());
            }
        }
        return results;
    }

    private static List<FdrResult> evaluateDatasetFdr(String datasetName, Config config) throws IOException {
        Dataset dataset = loadDataset(datasetName, config);
        Split split = Split.from(dataset, config);
        TrainStats stats = statsForSplit(dataset, split, config);
        List<ContextSet> contexts = contexts(dataset, stats, split.trainEnd, config);
        List<FdrResult> results = new ArrayList<>();
        for (ContextSet context : contexts) {
            int calibrationStart = calibrationStartBucket(split, config);
            double[] lineCalibrationScores = calibrationLineScores(dataset, stats, context, calibrationStart,
                    split.calibrationEnd, config);
            double[] cellCalibrationScores = calibrationCellScores(dataset, stats, context, calibrationStart,
                    split.calibrationEnd);
            java.util.Arrays.sort(lineCalibrationScores);
            java.util.Arrays.sort(cellCalibrationScores);
            double contextRecall = contextLineRecall(dataset, context, split.calibrationEnd, dataset.buckets.size());
            for (double q : config.fdrQValues) {
                FdrResult line = evaluateLineBh(datasetName, dataset, stats, context, lineCalibrationScores, q,
                        split.calibrationEnd, dataset.buckets.size(), contextRecall, config);
                results.add(line);
                System.out.println(line.toCsv());
                FdrResult cell = evaluateCellBh(datasetName, dataset, stats, context, cellCalibrationScores, q,
                        split.calibrationEnd, dataset.buckets.size(), contextRecall);
                results.add(cell);
                System.out.println(cell.toCsv());
            }
        }
        return results;
    }

    private static List<FdrResult> evaluateDatasetTailExcess(String datasetName, Config config) throws IOException {
        Dataset dataset = loadDataset(datasetName, config);
        Split split = Split.from(dataset, config);
        TrainStats stats = statsForSplit(dataset, split, config);
        List<ContextSet> contexts = contexts(dataset, stats, split.trainEnd, config);
        List<FdrResult> results = new ArrayList<>();
        for (ContextSet context : contexts) {
            double contextRecall = contextLineRecall(dataset, context, split.calibrationEnd, dataset.buckets.size());
            int calibrationStart = calibrationStartBucket(split, config);
            TailCalibration lineCalibration = tailCalibration(dataset, stats, context, calibrationStart,
                    split.calibrationEnd, config, false);
            TailCalibration conditionalCalibration = tailCalibration(dataset, stats, context, calibrationStart,
                    split.calibrationEnd, config, true);
            for (double targetPrecision : config.fdrQValues) {
                FdrResult line = evaluateTailExcess(datasetName, dataset, stats, context, lineCalibration,
                        targetPrecision, split.calibrationEnd, dataset.buckets.size(), contextRecall, config, false);
                results.add(line);
                System.out.println(line.toCsv());
                FdrResult conditional = evaluateTailExcess(datasetName, dataset, stats, context, conditionalCalibration,
                        targetPrecision, split.calibrationEnd, dataset.buckets.size(), contextRecall, config, true);
                results.add(conditional);
                System.out.println(conditional.toCsv());
                FdrResult seedExpand = evaluateTailSeedExpand(datasetName, dataset, stats, context, lineCalibration,
                        targetPrecision, split.calibrationEnd, dataset.buckets.size(), contextRecall, config);
                results.add(seedExpand);
                System.out.println(seedExpand.toCsv());
            }
        }
        return results;
    }

    private static List<FdrResult> evaluateDatasetQuantile(String datasetName, Config config) throws IOException {
        Dataset dataset = loadDataset(datasetName, config);
        Split split = Split.from(dataset, config);
        TrainStats stats = statsForSplit(dataset, split, config);
        List<ContextSet> contexts = contexts(dataset, stats, split.trainEnd, config);
        List<FdrResult> results = new ArrayList<>();
        for (ContextSet context : contexts) {
            int calibrationStart = calibrationStartBucket(split, config);
            double[] calibrationScores = calibrationLineScores(dataset, stats, context, calibrationStart,
                    split.calibrationEnd, config);
            java.util.Arrays.sort(calibrationScores);
            double contextRecall = contextLineRecall(dataset, context, split.calibrationEnd, dataset.buckets.size());
            for (double quantile : config.fdrQValues) {
                FdrResult fixed = evaluateFixedQuantile(datasetName, dataset, stats, context, calibrationScores,
                        quantile, split.calibrationEnd, dataset.buckets.size(), contextRecall, config);
                results.add(fixed);
                System.out.println(fixed.toCsv());
            }
            for (QuantilePair pair : config.seedExpandPairs) {
                FdrResult seedExpand = evaluateQuantileSeedExpand(datasetName, dataset, stats, context,
                        calibrationScores, pair.seedQuantile, pair.expandQuantile, split.calibrationEnd,
                        dataset.buckets.size(), contextRecall, config);
                results.add(seedExpand);
                System.out.println(seedExpand.toCsv());
            }
        }
        return results;
    }

    private static List<FdrResult> evaluateDatasetOnlineDecayed(String datasetName, Config config) throws IOException {
        Dataset dataset = loadDataset(datasetName, config);
        Split split = Split.from(dataset, config);
        int warmupEnd = split.trainEnd;
        OnlineBaselineState seed = OnlineBaselineState.fromWarmup(dataset, 0, warmupEnd, config);
        OnlineThresholdState seedThresholds = new OnlineThresholdState(config);
        int warmupScores = seedOnlineThresholds(dataset, seed, seedThresholds, 0, warmupEnd, config);
        List<FdrResult> results = new ArrayList<>();
        for (double quantile : config.fdrQValues) {
            OnlineBaselineState online = new OnlineBaselineState(seed);
            OnlineThresholdState thresholds = new OnlineThresholdState(seedThresholds);
            FdrResult result = evaluateOnlineDecayed(datasetName, dataset, online, thresholds, quantile, warmupEnd,
                    dataset.buckets.size(), warmupScores, config);
            results.add(result);
            System.out.println(result.toCsv());
        }
        return results;
    }

    private static List<OracleResult> evaluateDatasetOnlineOracle(String datasetName, Config config)
            throws IOException {
        Dataset dataset = loadDataset(datasetName, config);
        Split split = Split.from(dataset, config);
        int warmupEnd = split.trainEnd;
        OnlineBaselineState seed = OnlineBaselineState.fromWarmup(dataset, 0, warmupEnd, config);
        OnlineThresholdState seedThresholds = new OnlineThresholdState(config);
        seedOnlineThresholds(dataset, seed, seedThresholds, 0, warmupEnd, config);
        List<OracleResult> results = new ArrayList<>();
        for (double quantile : config.fdrQValues) {
            OnlineBaselineState online = new OnlineBaselineState(seed);
            OnlineThresholdState thresholds = new OnlineThresholdState(seedThresholds);
            OracleResult result = evaluateOnlineOracle(datasetName, dataset, online, thresholds, quantile, warmupEnd,
                    dataset.buckets.size(), config);
            results.add(result);
            System.out.println(result.toCsv());
        }
        return results;
    }

    private static List<FdrResult> evaluateDatasetAnchoredDynamic(String datasetName, Config config)
            throws IOException {
        Dataset dataset = loadDataset(datasetName, config);
        Split split = Split.from(dataset, config);
        int warmupEnd = split.trainEnd;
        OnlineBaselineState seed = OnlineBaselineState.fromWarmup(dataset, 0, warmupEnd, config);
        List<FdrResult> results = new ArrayList<>();
        for (double quantile : config.fdrQValues) {
            AnchoredDynamicThresholdState thresholds = AnchoredDynamicThresholdState.fromWarmup(dataset, seed, 0,
                    warmupEnd, quantile, config);
            OnlineBaselineState online = new OnlineBaselineState(seed);
            FdrResult result = evaluateAnchoredDynamic(datasetName, dataset, online, thresholds, quantile, warmupEnd,
                    dataset.buckets.size(), dataset.countLines(0, warmupEnd), config, config.scoreMode,
                    "anchored_dynamic");
            results.add(result);
            System.out.println(result.toCsv());
        }
        return results;
    }

    private static List<FdrResult> evaluateDatasetAnchoredOnline(String datasetName, Config config) throws IOException {
        long phaseStart = System.nanoTime();
        Dataset dataset = loadDataset(datasetName, config);
        printOnlinePhase(datasetName, "load", phaseStart, config);
        Split split = Split.from(dataset, config);
        int warmupEnd = split.trainEnd;
        if (config.anchorCalibrationFraction >= 1.0) {
            phaseStart = System.nanoTime();
            OnlineBaselineState seed = OnlineBaselineState.fromWarmup(dataset, 0, warmupEnd, config);
            printOnlinePhase(datasetName, "seed_full_warmup", phaseStart, config);
            phaseStart = System.nanoTime();
            AnchoredOnlineSelection selection = selectAnchoredOnline(datasetName, dataset, seed, 0, warmupEnd, config,
                    false);
            printOnlinePhase(datasetName, "select_score_anchor", phaseStart, config);
            phaseStart = System.nanoTime();
            AnchoredDynamicThresholdState thresholds = AnchoredDynamicThresholdState.fromWarmup(dataset, seed, 0,
                    warmupEnd, selection.anchorQuantile, config, selection.scoreMode, selection.anchorThreshold, false);
            printOnlinePhase(datasetName, "seed_thresholds", phaseStart, config);
            phaseStart = System.nanoTime();
            OnlineBaselineState online = OnlineBaselineState.fromWarmup(dataset, 0, warmupEnd, config,
                    selection.scoreMode);
            printOnlinePhase(datasetName, "seed_eval_baseline", phaseStart, config);
            FdrResult result = evaluateAnchoredDynamic(datasetName, dataset, online, thresholds,
                    selection.anchorQuantile, warmupEnd, dataset.buckets.size(), dataset.countLines(0, warmupEnd),
                    config, selection.scoreMode, selection.methodPrefix());
            List<FdrResult> results = new ArrayList<>();
            results.add(result);
            System.out.println(result.toCsv());
            return results;
        }
        int calibrationBuckets = max(1, (int) Math.ceil(config.anchorCalibrationFraction * warmupEnd));
        int countWarmupEnd = max(1, warmupEnd - calibrationBuckets);
        if (countWarmupEnd >= warmupEnd) {
            countWarmupEnd = max(1, warmupEnd - 1);
        }
        phaseStart = System.nanoTime();
        OnlineBaselineState countSeed = OnlineBaselineState.fromWarmup(dataset, 0, countWarmupEnd, config);
        printOnlinePhase(datasetName, "seed_count_warmup", phaseStart, config);
        phaseStart = System.nanoTime();
        AnchoredOnlineSelection selection = selectAnchoredOnline(datasetName, dataset, countSeed, countWarmupEnd,
                warmupEnd, config, true);
        printOnlinePhase(datasetName, "select_score_anchor", phaseStart, config);
        OnlineBaselineState thresholdSeed = new OnlineBaselineState(countSeed);
        phaseStart = System.nanoTime();
        AnchoredDynamicThresholdState thresholds = AnchoredDynamicThresholdState.fromWarmup(dataset, thresholdSeed,
                countWarmupEnd, warmupEnd, selection.anchorQuantile, config, selection.scoreMode,
                selection.anchorThreshold, true);
        printOnlinePhase(datasetName, "seed_thresholds", phaseStart, config);
        phaseStart = System.nanoTime();
        OnlineBaselineState online = OnlineBaselineState.fromWarmup(dataset, 0, warmupEnd, config, selection.scoreMode);
        printOnlinePhase(datasetName, "seed_eval_baseline", phaseStart, config);
        FdrResult result = evaluateAnchoredDynamic(datasetName, dataset, online, thresholds, selection.anchorQuantile,
                warmupEnd, dataset.buckets.size(), dataset.countLines(countWarmupEnd, warmupEnd), config,
                selection.scoreMode, selection.methodPrefix());
        List<FdrResult> results = new ArrayList<>();
        results.add(result);
        System.out.println(result.toCsv());
        return results;
    }

    private static void printOnlinePhase(String datasetName, String phase, long startNanos, Config config) {
        if (config.profileOnline) {
            System.out.printf(Locale.ROOT, "online_profile_phase dataset=%s phase=%s seconds=%.3f%n", datasetName,
                    phase, secondsSince(startNanos));
        }
    }

    private static FdrResult evaluateDatasetStreamingAnchoredOnline(String datasetName, Config config)
            throws IOException {
        if (!config.useCompactOnlineState()) {
            throw new IllegalArgumentException(
                    "streaming anchored-online currently requires --memory-profile compact/default-100mb/tiny");
        }
        long phaseStart = System.nanoTime();
        StreamingDatasetInfo info = scanStreamingDataset(datasetName, config);
        printStreamingPhase(datasetName, "scan", phaseStart, config);
        int warmupEnd = max(1, (int) Math.floor(config.trainFraction * info.buckets));
        warmupEnd = min(warmupEnd, max(1, info.buckets - 1));
        int evalEnd = info.buckets;
        if (config.profileMaxEvalBuckets > 0) {
            evalEnd = min(evalEnd, warmupEnd + config.profileMaxEvalBuckets);
            System.out.printf(Locale.ROOT,
                    "streaming_profile_limited_eval dataset=%s start_bucket=%d end_bucket=%d max_eval_buckets=%d%n",
                    datasetName, warmupEnd, evalEnd, config.profileMaxEvalBuckets);
        }

        phaseStart = System.nanoTime();
        OnlineBaselineState seed = streamingWarmupState(datasetName, 0, warmupEnd, config, config.scoreMode);
        printStreamingPhase(datasetName, "seed_full_warmup", phaseStart, config);

        phaseStart = System.nanoTime();
        StreamingAnchoredSelection selection = selectStreamingAnchoredOnline(datasetName, seed, 0, warmupEnd, config);
        printStreamingPhase(datasetName, "select_score_anchor", phaseStart, config);

        phaseStart = System.nanoTime();
        StreamingAnchoredThreshold thresholds = streamingThresholdsFromWarmup(datasetName, seed, 0, warmupEnd,
                selection, config);
        printStreamingPhase(datasetName, "seed_thresholds", phaseStart, config);

        phaseStart = System.nanoTime();
        OnlineBaselineState online = streamingWarmupState(datasetName, 0, warmupEnd, config, selection.scoreMode);
        printStreamingPhase(datasetName, "seed_eval_baseline", phaseStart, config);

        phaseStart = System.nanoTime();
        Metrics metrics = evaluateStreamingAnchored(datasetName, online, thresholds, warmupEnd, evalEnd, selection,
                config);
        printStreamingPhase(datasetName, "eval", phaseStart, config);

        String method = String.format(Locale.ROOT,
                "streaming_anchored_online_auto_global_%s_wq%.5f_z%.2f_h%.2fd_counts_online_thr_global_shrink%.0f",
                selection.scoreMode, selection.anchorQuantile, config.zFactor, config.anchoredThresholdHalfLifeDays,
                config.anchoredThresholdShrinkageK);
        if (config.anchoredLogScores) {
            method += "_log1p";
        }
        if (config.anchoredThresholdFloorParent) {
            method += "_parentfloor";
        }
        method += "_" + config.anchoredThresholdDynamicMode;
        if (config.onlineStrictThreshold) {
            method += "_strict";
        }
        FdrResult result = new FdrResult(datasetName, method, "none", 1.0, 1, selection.anchorQuantile,
                selection.warmupLines, 1.0, metrics);
        System.out.println(result.toCsv());
        return result;
    }

    private static StreamingDatasetInfo scanStreamingDataset(String datasetName, Config config) throws IOException {
        String input = datasetPath(datasetName, config);
        int sampleModulo = datasetSampleModulo(datasetName, config);
        long originalLine = 0;
        long sampledLines = 0;
        long anomalousLines = 0;
        int buckets = 0;
        long currentBucketKey = Long.MIN_VALUE;
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(input), RAW_LOG_CHARSET)) {
            String line;
            while ((line = reader.readLine()) != null) {
                ++originalLine;
                if (sampleModulo > 1 && positiveHash(Long.toString(originalLine)) % sampleModulo != 0) {
                    continue;
                }
                ParsedLine parsed = parseBglLikeLine(line);
                if (parsed == null || parsed.message.isEmpty()) {
                    continue;
                }
                long bucketKey = Math.floorDiv(parsed.epochSeconds, config.bucketSeconds);
                if (buckets == 0 || bucketKey != currentBucketKey) {
                    currentBucketKey = bucketKey;
                    ++buckets;
                }
                ++sampledLines;
                if (parsed.label) {
                    ++anomalousLines;
                }
            }
        }
        System.out.printf(Locale.ROOT,
                "streaming_loaded dataset=%s sampled_lines=%d buckets=%d anomalous_lines=%d sample_modulo=%d%n",
                datasetName, sampledLines, buckets, anomalousLines, sampleModulo);
        return new StreamingDatasetInfo(sampledLines, buckets, anomalousLines);
    }

    private static OnlineBaselineState streamingWarmupState(String datasetName, int startBucket, int endBucket,
            Config config, String scoreMode) throws IOException {
        OnlineBaselineState state = new OnlineBaselineState(config, scoreMode);
        streamLines(datasetName, config, startBucket, endBucket, new StreamingLineConsumer() {
            @Override
            public void accept(StreamLine line) {
                state.advanceTo(line.bucketKey);
                state.update(line, 1.0, 1.0, 1.0);
            }
        });
        return state;
    }

    private static StreamingAnchoredSelection selectStreamingAnchoredOnline(String datasetName,
            OnlineBaselineState seed, int startBucket, int endBucket, Config config) throws IOException {
        StreamingAnchoredSelection best = null;
        for (String scoreMode : anchoredScoreCandidates(config)) {
            StreamingScoreSummary summary = new StreamingScoreSummary(config);
            OnlineBaselineState scoringSeed = new OnlineBaselineState(seed);
            streamLines(datasetName, config, startBucket, endBucket, new StreamingLineConsumer() {
                @Override
                public void accept(StreamLine line) {
                    double rawScore = streamingLineScore(line, scoringSeed, config, scoreMode);
                    summary.add(anchoredThresholdScore(rawScore, config));
                }
            });
            double quantile = config.useHistogramAnchorQuantile(scoreMode) ? config.fixedAnchorQuantile()
                    : min(config.anchorMaxQuantile, scoreFamilyMaxAnchorQuantile(scoreMode, config));
            quantile = max(config.anchorMinQuantile, min(config.anchorMaxQuantile, quantile));
            double threshold = summary.quantile(quantile);
            double knee = summary.bestTailKneeStrength(config);
            double usefulTail = 1.0 / (1.0 + Math.abs(Math.log(max(1.0e-6, 1.0 - quantile) / 0.01)));
            double quality = streamingScoreModePrior(scoreMode, seed) * (knee + 0.25 * usefulTail);
            StreamingAnchoredSelection selection = new StreamingAnchoredSelection(scoreMode, quantile, threshold,
                    quality, knee, summary.total());
            System.out.printf(Locale.ROOT,
                    "streaming_anchored_candidate dataset=%s score_mode=%s anchor_q=%.6f anchor=%.8f quality=%.6f knee=%.6f warmup_lines=%d%n",
                    datasetName, scoreMode, selection.anchorQuantile, selection.anchorThreshold, selection.quality,
                    selection.kneeStrength, selection.warmupLines);
            if (best == null || selection.quality > best.quality) {
                best = selection;
            }
        }
        if (best == null) {
            throw new IllegalStateException("no streaming anchored-online score candidates");
        }
        System.out.printf(Locale.ROOT,
                "streaming_anchored_selected dataset=%s score_mode=%s anchor_q=%.6f anchor=%.8f quality=%.6f%n",
                datasetName, best.scoreMode, best.anchorQuantile, best.anchorThreshold, best.quality);
        return best;
    }

    private static double streamingScoreModePrior(String scoreMode, OnlineBaselineState seed) {
        if (!scoreMode.startsWith("component_")) {
            return 1.0;
        }
        // Component/stable scoring is intended for compact, low-template streams where
        // stable scary families dominate the tail. High template cardinality streams
        // tend
        // to behave more like Thunderbird, where global rarity is the safer operating
        // point.
        int templates = seed.slow.globalCounts.size();
        return templates > 500 ? 0.10 : 1.0;
    }

    private static StreamingAnchoredThreshold streamingThresholdsFromWarmup(String datasetName,
            OnlineBaselineState seed, int startBucket, int endBucket, StreamingAnchoredSelection selection,
            Config config) throws IOException {
        StreamingAnchoredThreshold thresholds = new StreamingAnchoredThreshold(config);
        OnlineBaselineState scoringSeed = new OnlineBaselineState(seed);
        streamLines(datasetName, config, startBucket, endBucket, new StreamingLineConsumer() {
            @Override
            public void accept(StreamLine line) {
                double rawScore = streamingLineScore(line, scoringSeed, config, selection.scoreMode);
                double score = anchoredThresholdScore(rawScore, config);
                thresholds.update(score, line.bucketKey);
            }
        });
        thresholds.setWarmupAnchor(selection.anchorThreshold);
        thresholds.captureWarmupDynamic();
        return thresholds;
    }

    private static Metrics evaluateStreamingAnchored(String datasetName, OnlineBaselineState online,
            StreamingAnchoredThreshold thresholds, int warmupEnd, int endBucket, StreamingAnchoredSelection selection,
            Config config) throws IOException {
        Metrics metrics = new Metrics(0, 0, 0, 0);
        StreamingProfile profile = new StreamingProfile(datasetName, warmupEnd, endBucket, config);
        StreamingParseState parseState = new StreamingParseState(config);
        streamLines(datasetName, config, warmupEnd, endBucket, parseState, new StreamingLineConsumer() {
            @Override
            public void accept(StreamLine line) {
                long timer = System.nanoTime();
                online.advanceTo(line.bucketKey);
                profile.countAdvanceNanos += System.nanoTime() - timer;
                timer = System.nanoTime();
                thresholds.advanceTo(line.bucketKey);
                profile.thresholdAdvanceNanos += System.nanoTime() - timer;
                timer = System.nanoTime();
                double rawScore = streamingLineScore(line, online, config, selection.scoreMode);
                double score = anchoredThresholdScore(rawScore, config);
                profile.scoreNanos += System.nanoTime() - timer;
                timer = System.nanoTime();
                double threshold = thresholds.threshold(line.bucketKey);
                profile.thresholdNanos += System.nanoTime() - timer;
                boolean prediction = onlinePrediction(score, threshold, config);
                metrics.add(line.label, prediction);
                if (config.onlineUpdateThresholds) {
                    double updateScore = prediction && config.onlineWinsorizeThresholdUpdates ? min(score, threshold)
                            : score;
                    timer = System.nanoTime();
                    thresholds.update(updateScore, line.bucketKey);
                    profile.thresholdUpdateNanos += System.nanoTime() - timer;
                }
                if (config.onlineUpdateCounts) {
                    timer = System.nanoTime();
                    online.updateAfterDecision(line, prediction, config);
                    profile.countUpdateNanos += System.nanoTime() - timer;
                }
                profile.record(line, score, threshold, prediction, thresholds.lastThresholdAnchorControlled());
                profile.maybeReport(line.bucketIndex, online);
            }
        });
        profile.report("final", endBucket - 1, online, thresholds, parseState);
        return metrics;
    }

    private static double streamingLineScore(StreamLine line, OnlineBaselineState online, Config config,
            String scoreMode) {
        int eventId = line.eventId;
        int keyword = line.keywordScore;
        switch (scoreMode) {
        case "global_rarity_keyword":
            return online.slow.globalRarity(eventId) + 3.0 * keyword;
        case "component_rarity_keyword":
            return online.slow.componentRarity(line.componentId, eventId) + 3.0 * keyword;
        case "component_rarity_keyword_stable":
            return streamingStableKeywordScore(line, online, true);
        default:
            throw new IllegalArgumentException("streaming compact mode does not support score mode " + scoreMode);
        }
    }

    private static double streamingStableKeywordScore(StreamLine line, OnlineBaselineState online,
            boolean componentRarity) {
        double rarity = componentRarity ? online.slow.componentRarity(line.componentId, line.eventId)
                : online.slow.globalRarity(line.eventId);
        double stablePenalty = online.stableSuppression(line.componentId, line.levelId, line.phraseHash, line.eventId);
        return max(0.0, rarity + 3.0 * line.keywordScore - 1.25 * stablePenalty);
    }

    private static void streamLines(String datasetName, Config config, int startBucket, int endBucket,
            StreamingLineConsumer consumer) throws IOException {
        streamLines(datasetName, config, startBucket, endBucket, new StreamingParseState(config), consumer);
    }

    private static void streamLines(String datasetName, Config config, int startBucket, int endBucket,
            StreamingParseState state, StreamingLineConsumer consumer) throws IOException {
        String input = datasetPath(datasetName, config);
        int sampleModulo = datasetSampleModulo(datasetName, config);
        long originalLine = 0;
        int bucketIndex = -1;
        long currentBucketKey = Long.MIN_VALUE;
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(input), RAW_LOG_CHARSET)) {
            String line;
            while ((line = reader.readLine()) != null) {
                ++originalLine;
                if (sampleModulo > 1 && positiveHash(Long.toString(originalLine)) % sampleModulo != 0) {
                    continue;
                }
                ParsedLine parsed = parseBglLikeLine(line);
                if (parsed == null || parsed.message.isEmpty()) {
                    continue;
                }
                long bucketKey = Math.floorDiv(parsed.epochSeconds, config.bucketSeconds);
                if (bucketIndex < 0 || bucketKey != currentBucketKey) {
                    currentBucketKey = bucketKey;
                    ++bucketIndex;
                    if (bucketIndex >= endBucket) {
                        break;
                    }
                }
                int entityId = state.ids.entityId(parsed.entity);
                int eventId = state.parser.parse(normalize(parsed.message));
                int componentId = state.ids.componentId(parsed.component);
                int levelId = state.ids.levelId(parsed.level);
                int prevEventId = state.previousEventId(entityId);
                state.updateLastEvent(entityId, eventId);
                if (bucketIndex < startBucket) {
                    continue;
                }
                StreamLine streamLine = new StreamLine(bucketIndex, bucketKey, eventId, entityId, componentId, levelId,
                        prevEventId, parameterHash(parsed.message), semanticPhraseHash(parsed.message),
                        parsed.keywordScore, parsed.label);
                consumer.accept(streamLine);
            }
        }
    }

    private static TemplateParser newTemplateParser(Config config) {
        return config.useFixedDepthDrainParser()
                ? new FixedDepthDrainParser(config.drainSimilarity, config.drainDepth, config.drainMaxChildren,
                        config.memoryLimits.maxTemplates, config.memoryLimits.maxParserNodes,
                        config.memoryLimits.parserCapBytes)
                : new SimpleDrainParser(config.drainSimilarity, config.memoryLimits.maxTemplates,
                        config.memoryLimits.parserCapBytes);
    }

    private static String datasetPath(String datasetName, Config config) {
        return "thunderbird".equals(datasetName) ? config.thunderbirdPath : config.bglPath;
    }

    private static int datasetSampleModulo(String datasetName, Config config) {
        return "thunderbird".equals(datasetName) ? config.thunderbirdSampleModulo : config.bglSampleModulo;
    }

    private static void printStreamingPhase(String datasetName, String phase, long startNanos, Config config) {
        if (config.profileOnline) {
            System.out.printf(Locale.ROOT,
                    "streaming_profile_phase dataset=%s phase=%s seconds=%.3f used_heap_mib=%.2f%n", datasetName, phase,
                    secondsSince(startNanos), usedHeapMiB());
        }
    }

    private static FdrResult evaluateAnchoredDynamic(String datasetName, Dataset dataset, OnlineBaselineState online,
            AnchoredDynamicThresholdState thresholds, double warmupQuantile, int startBucket, int endBucket,
            int calibrationHypotheses, Config config, String scoreMode, String methodPrefix) {
        if (config.profileMaxEvalBuckets > 0) {
            endBucket = min(endBucket, startBucket + config.profileMaxEvalBuckets);
            System.out.printf(Locale.ROOT,
                    "online_profile_limited_eval dataset=%s start_bucket=%d end_bucket=%d max_eval_buckets=%d lines=%d%n",
                    datasetName, startBucket, endBucket, config.profileMaxEvalBuckets,
                    dataset.countLines(startBucket, endBucket));
        }
        Metrics metrics = new Metrics(0, 0, 0, 0);
        OnlineProfile profile = config.profileOnline ? new OnlineProfile(datasetName, startBucket, endBucket, config)
                : null;
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            long bucketKey = dataset.buckets.get(bucketIndex).bucketKey;
            long timer = System.nanoTime();
            online.advanceTo(bucketKey);
            if (profile != null) {
                profile.countAdvanceNanos += System.nanoTime() - timer;
            }
            timer = System.nanoTime();
            thresholds.advanceTo(bucketKey);
            if (profile != null) {
                profile.thresholdAdvanceNanos += System.nanoTime() - timer;
            }
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                timer = System.nanoTime();
                double rawScore = onlineDecayedLineScore(dataset, online, i, config, scoreMode);
                double score = anchoredThresholdScore(rawScore, config);
                if (profile != null) {
                    profile.scoreNanos += System.nanoTime() - timer;
                }
                timer = System.nanoTime();
                double threshold = thresholds.threshold(dataset, i, bucketKey);
                if (profile != null) {
                    profile.thresholdNanos += System.nanoTime() - timer;
                }
                boolean prediction = onlinePrediction(score, threshold, config);
                if (profile != null) {
                    profile.recordDecision(dataset, i, score, threshold, prediction,
                            thresholds.lastThresholdAnchorControlled());
                }
                metrics.add(dataset.labels.values[i] != 0, prediction);
                if (config.onlineUpdateThresholds) {
                    double updateScore = prediction && config.onlineWinsorizeThresholdUpdates ? min(score, threshold)
                            : score;
                    timer = System.nanoTime();
                    thresholds.update(dataset, i, updateScore, bucketKey);
                    if (profile != null) {
                        profile.thresholdUpdateNanos += System.nanoTime() - timer;
                    }
                }
                if (config.onlineUpdateCounts) {
                    timer = System.nanoTime();
                    online.updateAfterDecision(dataset, i, prediction, config);
                    if (profile != null) {
                        profile.countUpdateNanos += System.nanoTime() - timer;
                    }
                }
                if (profile != null) {
                    profile.lines++;
                    profile.maybeReport(bucketIndex, online);
                }
            }
        }
        if (profile != null) {
            profile.report("final", endBucket - 1, online);
        }
        String method = String.format(Locale.ROOT, "%s_%s_wq%.5f_z%.2f_h%.2fd_counts_%s_thr_%s_shrink%.0f",
                methodPrefix, scoreMode, warmupQuantile, config.zFactor, config.anchoredThresholdHalfLifeDays,
                config.onlineUpdateCounts ? "online" : "static",
                config.onlineUpdateThresholds ? AnchoredDynamicThresholdState.effectiveGroupMode(config) : "static",
                config.anchoredThresholdShrinkageK);
        if (config.anchoredLogScores) {
            method += "_log1p";
        }
        if (config.anchoredThresholdFloorParent) {
            method += "_parentfloor";
        }
        method += "_" + config.anchoredThresholdDynamicMode;
        if (config.onlineStrictThreshold) {
            method += "_strict";
        }
        return new FdrResult(datasetName, method, "none", 1.0, 1, warmupQuantile, calibrationHypotheses, 1.0, metrics);
    }

    private static int seedOnlineThresholds(Dataset dataset, OnlineBaselineState seed, OnlineThresholdState thresholds,
            int startBucket, int endBucket, Config config) {
        int count = 0;
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            long bucketKey = dataset.buckets.get(bucketIndex).bucketKey;
            thresholds.advanceTo(bucketKey);
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                thresholds.update(dataset, i, onlineDecayedLineScore(dataset, seed, i, config), bucketKey);
                ++count;
            }
        }
        return count;
    }

    private static FdrResult evaluateOnlineDecayed(String datasetName, Dataset dataset, OnlineBaselineState online,
            OnlineThresholdState thresholds, double quantile, int startBucket, int endBucket, int calibrationHypotheses,
            Config config) {
        Metrics metrics = new Metrics(0, 0, 0, 0);
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            long bucketKey = dataset.buckets.get(bucketIndex).bucketKey;
            online.advanceTo(bucketKey);
            thresholds.advanceTo(bucketKey);
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                double threshold = thresholds.quantile(dataset, i, quantile, bucketKey);
                double score = onlineDecayedLineScore(dataset, online, i, config);
                boolean prediction = onlinePrediction(score, threshold, config);
                boolean updatePrediction = onlineUpdatePrediction(thresholds, dataset, i, quantile, bucketKey, score,
                        threshold, config);
                metrics.add(dataset.labels.values[i] != 0, prediction);
                if (config.onlineUpdateThresholds) {
                    updateOnlineThreshold(thresholds, dataset, i, bucketKey, score, threshold, updatePrediction,
                            config);
                }
                if (config.onlineUpdateCounts) {
                    online.updateAfterDecision(dataset, i, updatePrediction, config);
                }
            }
        }
        String method = String.format(Locale.ROOT, "online_decayed_%s_slow%.2fd_fast%.2fh_q%.2fd_counts_%s_thr_%s",
                config.scoreMode, config.onlineTemplateHalfLifeDays, config.onlineFastHalfLifeHours,
                config.onlineQuantileHalfLifeDays, config.onlineUpdateCounts ? "online" : "static",
                config.onlineUpdateThresholds ? config.onlineThresholdGroup : "static");
        if (config.onlineUpdateThresholds && !"global".equals(config.onlineThresholdGroup)) {
            method += "_min" + config.onlineThresholdMinCount;
        }
        if (Double.isFinite(config.onlineUpdateGuardQuantile)) {
            method += String.format(Locale.ROOT, "_guard%.5f", config.onlineUpdateGuardQuantile);
        }
        if (Math.abs(config.onlineTiebreakWeight - 0.001) > 1.0e-12) {
            method += String.format(Locale.ROOT, "_tb%.4f", config.onlineTiebreakWeight);
        }
        if (config.onlineStrictThreshold) {
            method += "_strict";
        }
        return new FdrResult(datasetName, method, "none", 1.0, 1, quantile, calibrationHypotheses, 1.0, metrics);
    }

    private static OracleResult evaluateOnlineOracle(String datasetName, Dataset dataset, OnlineBaselineState online,
            OnlineThresholdState thresholds, double updateQuantile, int startBucket, int endBucket, Config config) {
        int count = dataset.countLines(startBucket, endBucket);
        long[] packed = new long[count];
        int index = 0;
        long positives = 0;
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            long bucketKey = dataset.buckets.get(bucketIndex).bucketKey;
            online.advanceTo(bucketKey);
            thresholds.advanceTo(bucketKey);
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                double threshold = thresholds.quantile(dataset, i, updateQuantile, bucketKey);
                double score = onlineDecayedLineScore(dataset, online, i, config);
                boolean label = dataset.labels.values[i] != 0;
                packed[index++] = pack(score, label);
                if (label) {
                    ++positives;
                }
                boolean prediction = onlinePrediction(score, threshold, config);
                boolean updatePrediction = onlineUpdatePrediction(thresholds, dataset, i, updateQuantile, bucketKey,
                        score, threshold, config);
                if (config.onlineUpdateThresholds) {
                    updateOnlineThreshold(thresholds, dataset, i, bucketKey, score, threshold, updatePrediction,
                            config);
                }
                if (config.onlineUpdateCounts) {
                    online.updateAfterDecision(dataset, i, updatePrediction, config);
                }
            }
        }
        String scoreName = String.format(Locale.ROOT, "online_decayed_%s_update_q%.5f", config.scoreMode,
                updateQuantile);
        return oracleFromPacked(datasetName, scoreName, "none", 1.0, 1, packed, index, positives);
    }

    private static void updateOnlineThreshold(OnlineThresholdState thresholds, Dataset dataset, int lineIndex,
            long bucketKey, double score, double threshold, boolean prediction, Config config) {
        if (!prediction || !config.excludeAlertUpdates) {
            thresholds.update(dataset, lineIndex, score, bucketKey);
        } else if (score <= threshold + config.onlineWinsorizeScoreMargin) {
            thresholds.update(dataset, lineIndex, score, bucketKey);
        } else if (config.onlineWinsorizeThresholdUpdates && Double.isFinite(threshold)) {
            thresholds.update(dataset, lineIndex, min(score, threshold), bucketKey);
        }
    }

    private static boolean onlinePrediction(double score, double threshold, Config config) {
        return config.onlineStrictThreshold ? score > threshold : score >= threshold;
    }

    private static double anchoredThresholdScore(double score, Config config) {
        double clean = Double.isFinite(score) ? max(0.0, score) : config.rollingScoreMax;
        return config.anchoredLogScores ? Math.log1p(clean) : clean;
    }

    private static AnchoredOnlineSelection selectAnchoredOnline(String datasetName, Dataset dataset,
            OnlineBaselineState countSeed, int startBucket, int endBucket, Config config, boolean updateSeed) {
        List<String> candidates = anchoredScoreCandidates(config);
        AnchoredOnlineSelection best = null;
        for (String scoreMode : candidates) {
            TailKneeDiagnostic diagnostic = tailKneeDiagnostic(dataset, new OnlineBaselineState(countSeed), startBucket,
                    endBucket, scoreMode, config, updateSeed);
            AnchoredOnlineSelection selection = new AnchoredOnlineSelection(scoreMode,
                    AnchoredDynamicThresholdState.effectiveGroupMode(config), diagnostic.quantile, diagnostic.threshold,
                    diagnostic.quality, diagnostic);
            System.out.printf(Locale.ROOT,
                    "anchored_online_candidate dataset=%s score_mode=%s anchor_q=%.6f anchor=%.8f quality=%.6f knee=%.6f deployed_knee=%.6f stability=%.6f concentration=%.6f tail_count=%d%n",
                    datasetName, scoreMode, diagnostic.quantile, diagnostic.threshold, diagnostic.quality,
                    diagnostic.kneeStrength, diagnostic.deployedKneeStrength, diagnostic.stability,
                    diagnostic.concentration, diagnostic.tailCount);
            if (best == null || selection.quality > best.quality) {
                best = selection;
            }
        }
        if (best == null) {
            throw new IllegalStateException("no anchored-online score candidates");
        }
        System.out.printf(Locale.ROOT,
                "anchored_online_selected dataset=%s score_mode=%s threshold_group=%s anchor_q=%.6f anchor=%.8f quality=%.6f%n",
                datasetName, best.scoreMode, best.thresholdGroup, best.anchorQuantile, best.anchorThreshold,
                best.quality);
        return best;
    }

    private static List<String> anchoredScoreCandidates(Config config) {
        List<String> result = new ArrayList<>();
        if (!"auto".equals(config.scoreMode)) {
            result.add(config.scoreMode);
            return result;
        }
        if (config.useCompactOnlineState()) {
            result.add("global_rarity_keyword");
            result.add("component_rarity_keyword_stable");
        } else {
            result.add("global_rarity_keyword_tiebreak");
            result.add("component_rarity_keyword_stable_tiebreak");
        }
        return result;
    }

    private static TailKneeDiagnostic tailKneeDiagnostic(Dataset dataset, OnlineBaselineState online, int startBucket,
            int endBucket, String scoreMode, Config config, boolean updateSeed) {
        OnlineBaselineState base = new OnlineBaselineState(online);
        List<Double> cappedScores = new ArrayList<>();
        List<Double> allScores = new ArrayList<>();
        Map<Long, Integer> signatureCounts = new HashMap<>();
        OnlineThresholdState histogramAnchor = config.useHistogramAnchorQuantile(scoreMode)
                ? new OnlineThresholdState(config)
                : null;
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            long bucketKey = dataset.buckets.get(bucketIndex).bucketKey;
            if (updateSeed) {
                online.advanceTo(bucketKey);
            }
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                double rawScore = onlineDecayedLineScore(dataset, online, i, config, scoreMode);
                double score = anchoredThresholdScore(rawScore, config);
                allScores.add(score);
                if (histogramAnchor != null) {
                    histogramAnchor.update(dataset, i, rawScore, bucketKey);
                }
                long key = thresholdSelectionSignature(dataset, i);
                int seen = signatureCounts.getOrDefault(key, 0);
                if (seen < config.anchorSignatureCap) {
                    cappedScores.add(score);
                }
                signatureCounts.put(key, seen + 1);
                if (updateSeed) {
                    online.update(dataset, i, 1.0, 1.0, 1.0);
                }
            }
        }
        if (allScores.isEmpty()) {
            return new TailKneeDiagnostic(0.995, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 0.0);
        }
        Collections.sort(allScores);
        Collections.sort(cappedScores);
        List<Double> selectionScores = cappedScores.isEmpty() ? allScores : cappedScores;
        int n = selectionScores.size();
        int start = min(max((int) Math.floor(config.anchorMinQuantile * n), 0), max(0, n - 2));
        int end = min(max((int) Math.ceil(config.anchorMaxQuantile * n), start + 1), n - 1);
        double tailScale = tailScale(selectionScores, start, end);
        double bestScore = -1.0;
        double bestGap = -1.0;
        double bestSupport = 1.0;
        int bestIndex = start;
        for (int i = start; i < end; i++) {
            double gap = selectionScores.get(i + 1) - selectionScores.get(i);
            double q = (i + 1.0) / n;
            double support = max(0.02, (1.0 - q) / max(1.0e-6, 1.0 - config.anchorMinQuantile));
            double adjusted = gap * support;
            if (adjusted > bestScore) {
                bestScore = adjusted;
                bestGap = adjusted;
                bestSupport = support;
                bestIndex = i;
            }
        }
        double threshold = selectionScores.get(bestIndex);
        double quantile = actualQuantile(allScores, threshold);
        boolean policyAnchor = config.useFixedAnchorQuantile() || config.useHistogramAnchorQuantile(scoreMode);
        if (policyAnchor) {
            quantile = config.fixedAnchorQuantile();
            threshold = config.useHistogramAnchorQuantile(scoreMode)
                    ? anchoredThresholdScore(histogramAnchor.global.quantile(quantile), config)
                    : listQuantile(allScores, quantile);
        } else {
            quantile = max(config.anchorMinQuantile, min(config.anchorMaxQuantile, quantile));
            quantile = min(quantile, scoreFamilyMaxAnchorQuantile(scoreMode, config));
            threshold = listQuantile(allScores, quantile);
        }
        int tailCount = countAboveOrEqual(dataset, base, startBucket, endBucket, scoreMode, threshold, config);
        double concentration = tailConcentration(dataset, base, startBucket, endBucket, scoreMode, threshold, config,
                max(1, tailCount));
        double stability = tailStability(dataset, base, startBucket, endBucket, scoreMode, quantile, threshold, config);
        double kneeStrength = bestGap / max(1.0e-6, tailScale);
        double deployedKneeStrength = policyAnchor ? localKneeStrength(allScores, threshold, tailScale, config)
                : kneeStrength;
        double usefulTail = 1.0 / (1.0 + Math.abs(Math.log(max(1.0e-6, 1.0 - quantile) / 0.01)));
        double quality = kneeStrength + stability + 0.25 * usefulTail - 0.75 * concentration + 0.10 * bestSupport;
        return new TailKneeDiagnostic(quantile, threshold, kneeStrength, deployedKneeStrength, stability, concentration,
                tailCount, quality);
    }

    private static double scoreFamilyMaxAnchorQuantile(String scoreMode, Config config) {
        if (config.useFixedAnchorQuantile()) {
            return config.anchorMaxQuantile;
        }
        // Policy prior: component/stable score families are meant to suppress stable
        // scary
        // templates, so they may use a broader p98-style operating point. Global score
        // families stay conservative by default because tied template scores can
        // otherwise
        // admit high-volume normal families.
        if (scoreMode.startsWith("component_")) {
            return config.anchorMinQuantile;
        }
        return min(config.anchorMaxQuantile, 0.995);
    }

    private static long thresholdSelectionSignature(Dataset dataset, int lineIndex) {
        int componentId = dataset.componentIds.values[lineIndex];
        int levelId = dataset.levelIds.values[lineIndex];
        int eventId = dataset.eventIds.values[lineIndex];
        long key = 1469598103934665603L;
        key = (key ^ componentId) * 1099511628211L;
        key = (key ^ levelId) * 1099511628211L;
        key = (key ^ eventId) * 1099511628211L;
        return key;
    }

    private static double tailScale(List<Double> sorted, int start, int end) {
        if (end <= start) {
            return 1.0;
        }
        List<Double> gaps = new ArrayList<>();
        for (int i = start; i < end; i++) {
            gaps.add(max(0.0, sorted.get(i + 1) - sorted.get(i)));
        }
        Collections.sort(gaps);
        return max(1.0e-6, gaps.get(gaps.size() / 2));
    }

    private static double localKneeStrength(List<Double> sortedScores, double threshold, double tailScale,
            Config config) {
        if (sortedScores.size() <= 1) {
            return 0.0;
        }
        int boundary = config.onlineStrictThreshold ? upperBound(sortedScores, threshold)
                : lowerBound(sortedScores, threshold);
        if (boundary <= 0 || boundary >= sortedScores.size()) {
            return 0.0;
        }
        double gap = max(0.0, sortedScores.get(boundary) - sortedScores.get(boundary - 1));
        return gap / max(1.0e-6, tailScale);
    }

    private static double actualQuantile(List<Double> sortedScores, double threshold) {
        int index = upperBound(sortedScores, threshold);
        return index / (double) sortedScores.size();
    }

    private static double secondsSince(long startNanos) {
        return (System.nanoTime() - startNanos) / 1.0e9;
    }

    private static double seconds(long nanos) {
        return nanos / 1.0e9;
    }

    private static double usedHeapMiB() {
        Runtime runtime = Runtime.getRuntime();
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024.0 * 1024.0);
    }

    private static double bytesToMiB(long bytes) {
        return bytes / (1024.0 * 1024.0);
    }

    private static long hashMapBytes(int entries) {
        int table = 16;
        while (table < max(1, entries) * 2) {
            table <<= 1;
        }
        return 48L + 4L * table + 96L * entries;
    }

    private static long intDoubleMapBytes(Map<Integer, Double> map) {
        return hashMapBytes(map.size());
    }

    private static long longDoubleMapBytes(Map<Long, Double> map) {
        return hashMapBytes(map.size());
    }

    private static long intLongMapBytes(Map<Integer, Long> map) {
        return hashMapBytes(map.size());
    }

    private static long longLongMapBytes(Map<Long, Long> map) {
        return hashMapBytes(map.size());
    }

    private static long intIntMapBytes(Map<Integer, Integer> map) {
        return hashMapBytes(map.size());
    }

    private static long stringIntMapBytes(Map<String, Integer> map) {
        long bytes = hashMapBytes(map.size());
        for (String key : map.keySet()) {
            bytes += stringBytes(key);
        }
        return bytes;
    }

    private static long stringBytes(String value) {
        return value == null ? 0L : 40L + 2L * value.length();
    }

    private static long stringDictionaryCapBytes(int keys, int averageChars) {
        return hashMapBytes(keys) + 40L * keys + 2L * averageChars * keys;
    }

    private static void printStaticMemoryEstimate(Config config) {
        MemoryLimits limits = config.memoryLimits;
        if (!limits.bounded) {
            System.out.printf(Locale.ROOT,
                    "static_memory_estimate profile=%s bounded=false status=UNBOUNDED reason=rich_profile_has_no_hard_caps%n",
                    config.memoryProfile);
            return;
        }
        List<String> candidates = anchoredScoreCandidates(config);
        StaticMemoryEstimate selected = null;
        for (String scoreMode : candidates) {
            StaticMemoryEstimate estimate = staticMemoryEstimate(config, scoreMode);
            if (selected == null || estimate.bytes > selected.bytes) {
                selected = estimate;
            }
            System.out.printf(Locale.ROOT,
                    "static_memory_candidate profile=%s score_mode=%s estimated_max_mib=%.4f budget_mib=%.4f status=%s caps=%s%n",
                    config.memoryProfile, scoreMode, bytesToMiB(estimate.bytes), limits.budgetMiB,
                    estimate.bytes <= limits.budgetBytes() ? "OK" : "OVER_BUDGET", limits.summary());
        }
        if (selected != null) {
            System.out.printf(Locale.ROOT,
                    "static_memory_estimate profile=%s score_mode=%s estimated_max_mib=%.4f budget_mib=%.4f status=%s possible_scores=%s%n",
                    config.memoryProfile, selected.scoreMode, bytesToMiB(selected.bytes), limits.budgetMiB,
                    selected.bytes <= limits.budgetBytes() ? "OK" : "OVER_BUDGET", String.join(";", candidates));
            System.out.printf(Locale.ROOT,
                    "static_memory_breakdown online_counts_mib=%.4f threshold_mib=%.4f parser_cap_mib=%.4f id_state_mib=%.4f stream_context_mib=%.4f%n",
                    bytesToMiB(selected.onlineCountsBytes), bytesToMiB(selected.thresholdBytes),
                    bytesToMiB(selected.parserBytes), bytesToMiB(selected.idStateBytes),
                    bytesToMiB(selected.streamContextBytes));
            System.out.printf(Locale.ROOT, "static_memory_largest_components components=%s%n",
                    largestStaticMemoryComponents(selected));
        }
    }

    private static String largestStaticMemoryComponents(StaticMemoryEstimate estimate) {
        List<MemoryComponent> components = new ArrayList<>();
        components.add(new MemoryComponent("online_counts", estimate.onlineCountsBytes));
        components.add(new MemoryComponent("threshold", estimate.thresholdBytes));
        components.add(new MemoryComponent("parser_cap", estimate.parserBytes));
        components.add(new MemoryComponent("id_state", estimate.idStateBytes));
        components.add(new MemoryComponent("stream_context", estimate.streamContextBytes));
        components.sort((a, b) -> Long.compare(b.bytes, a.bytes));
        StringBuilder builder = new StringBuilder();
        for (MemoryComponent component : components) {
            if (builder.length() > 0) {
                builder.append(';');
            }
            builder.append(component.name).append(':')
                    .append(String.format(Locale.ROOT, "%.4fMiB", bytesToMiB(component.bytes)));
        }
        return builder.toString();
    }

    private static StaticMemoryEstimate staticMemoryEstimate(Config config, String scoreMode) {
        MemoryLimits limits = config.memoryLimits;
        long statsEmpty = 192L + 18L * hashMapBytes(0);
        long onlineCountsBytes;
        boolean globalScore = "global_rarity_keyword".equals(scoreMode);
        boolean componentStableScore = "component_rarity_keyword".equals(scoreMode)
                || "component_rarity_keyword_stable".equals(scoreMode);
        if (!globalScore && !componentStableScore) {
            throw new IllegalArgumentException(
                    "static memory estimate supports compact bounded scores only; got " + scoreMode);
        }
        if (componentStableScore) {
            long componentTemplateCap = min((long) limits.maxGroupTemplatePairs,
                    (long) limits.maxGroups * max(1L, (long) limits.maxTemplates));
            long statsComponentSlow = 192L + hashMapBytes(limits.maxTemplates)
                    + hashMapBytes((int) componentTemplateCap) + hashMapBytes(limits.maxGroups) + 15L * hashMapBytes(0);
            long statsStable = 192L + 3L * hashMapBytes(limits.maxTemplates)
                    + 3L * hashMapBytes(limits.maxStableSignatures) + 12L * hashMapBytes(0);
            onlineCountsBytes = 64L + statsComponentSlow + statsEmpty + statsStable;
        } else {
            long statsGlobal = 192L + hashMapBytes(limits.maxTemplates) + 17L * hashMapBytes(0);
            onlineCountsBytes = 64L + statsGlobal + 2L * statsEmpty;
        }
        long thresholdBytes = 160L;
        long parserBytes = limits.parserCapBytes;
        long idStateBytes = 48L;
        if (limits.entityEnabled()) {
            idStateBytes += stringDictionaryCapBytes(limits.maxEntities, limits.averageEntityChars);
        } else {
            idStateBytes += hashMapBytes(0);
        }
        int groupKeys = globalScore ? 1 : limits.maxGroups;
        // The implementation keeps component and level dictionaries separately, but one
        // profile cap controls both. This over-bounds the one logical group cap.
        idStateBytes += 2L * stringDictionaryCapBytes(groupKeys, limits.averageGroupChars);
        long streamContextBytes = 64L + (limits.entityEnabled() ? hashMapBytes(limits.maxEntities) : hashMapBytes(0));
        long total = onlineCountsBytes + thresholdBytes + parserBytes + idStateBytes + streamContextBytes;
        return new StaticMemoryEstimate(scoreMode, total, onlineCountsBytes, thresholdBytes, parserBytes, idStateBytes,
                streamContextBytes);
    }

    private static int countAboveOrEqual(Dataset dataset, OnlineBaselineState seed, int startBucket, int endBucket,
            String scoreMode, double threshold, Config config) {
        OnlineBaselineState online = new OnlineBaselineState(seed);
        int count = 0;
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            long bucketKey = dataset.buckets.get(bucketIndex).bucketKey;
            online.advanceTo(bucketKey);
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                double score = anchoredThresholdScore(onlineDecayedLineScore(dataset, online, i, config, scoreMode),
                        config);
                if (onlinePrediction(score, threshold, config)) {
                    ++count;
                }
                online.update(dataset, i, 1.0, 1.0, 1.0);
            }
        }
        return count;
    }

    private static double tailConcentration(Dataset dataset, OnlineBaselineState seed, int startBucket, int endBucket,
            String scoreMode, double threshold, Config config, int tailCount) {
        OnlineBaselineState online = new OnlineBaselineState(seed);
        Map<Long, Integer> counts = new HashMap<>();
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            long bucketKey = dataset.buckets.get(bucketIndex).bucketKey;
            online.advanceTo(bucketKey);
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                double score = anchoredThresholdScore(onlineDecayedLineScore(dataset, online, i, config, scoreMode),
                        config);
                if (onlinePrediction(score, threshold, config)) {
                    long key = thresholdSelectionSignature(dataset, i);
                    counts.put(key, counts.getOrDefault(key, 0) + 1);
                }
                online.update(dataset, i, 1.0, 1.0, 1.0);
            }
        }
        int maxCount = 0;
        for (int value : counts.values()) {
            maxCount = max(maxCount, value);
        }
        return maxCount / (double) max(1, tailCount);
    }

    private static double tailStability(Dataset dataset, OnlineBaselineState seed, int startBucket, int endBucket,
            String scoreMode, double quantile, double globalThreshold, Config config) {
        int windows = 4;
        double sum = 0.0;
        double sumSq = 0.0;
        int used = 0;
        for (int w = 0; w < windows; w++) {
            int from = startBucket + (int) Math.floor((endBucket - startBucket) * (w / (double) windows));
            int to = startBucket + (int) Math.floor((endBucket - startBucket) * ((w + 1.0) / windows));
            if (to <= from) {
                continue;
            }
            OnlineBaselineState online = new OnlineBaselineState(seed);
            List<Double> scores = new ArrayList<>();
            for (int bucketIndex = startBucket; bucketIndex < to; bucketIndex++) {
                long bucketKey = dataset.buckets.get(bucketIndex).bucketKey;
                online.advanceTo(bucketKey);
                for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                    double score = anchoredThresholdScore(onlineDecayedLineScore(dataset, online, i, config, scoreMode),
                            config);
                    if (bucketIndex >= from) {
                        scores.add(score);
                    }
                    online.update(dataset, i, 1.0, 1.0, 1.0);
                }
            }
            if (!scores.isEmpty()) {
                double value = listQuantile(scores, quantile);
                sum += value;
                sumSq += value * value;
                ++used;
            }
        }
        if (used <= 1) {
            return 0.0;
        }
        double mean = sum / used;
        double variance = max(0.0, sumSq / used - mean * mean);
        double cv = Math.sqrt(variance) / max(1.0e-6, Math.abs(globalThreshold));
        return 1.0 / (1.0 + cv);
    }

    private static final class AnchoredOnlineSelection {
        private final String scoreMode;
        private final String thresholdGroup;
        private final double anchorQuantile;
        private final double anchorThreshold;
        private final double quality;
        private final TailKneeDiagnostic diagnostic;

        private AnchoredOnlineSelection(String scoreMode, String thresholdGroup, double anchorQuantile,
                double anchorThreshold, double quality, TailKneeDiagnostic diagnostic) {
            this.scoreMode = scoreMode;
            this.thresholdGroup = thresholdGroup;
            this.anchorQuantile = anchorQuantile;
            this.anchorThreshold = anchorThreshold;
            this.quality = quality;
            this.diagnostic = diagnostic;
        }

        private String methodPrefix() {
            return "anchored_online_auto_" + thresholdGroup;
        }
    }

    private static final class TailKneeDiagnostic {
        private final double quantile;
        private final double threshold;
        private final double kneeStrength;
        private final double deployedKneeStrength;
        private final double stability;
        private final double concentration;
        private final int tailCount;
        private final double quality;

        private TailKneeDiagnostic(double quantile, double threshold, double kneeStrength, double deployedKneeStrength,
                double stability, double concentration, int tailCount, double quality) {
            this.quantile = quantile;
            this.threshold = threshold;
            this.kneeStrength = kneeStrength;
            this.deployedKneeStrength = deployedKneeStrength;
            this.stability = stability;
            this.concentration = concentration;
            this.tailCount = tailCount;
            this.quality = quality;
        }
    }

    private static final class OnlineProfile {
        private final String datasetName;
        private final int startBucket;
        private final int endBucket;
        private final int intervalLines;
        private final long startNanos = System.nanoTime();
        private long lines;
        private long lastReportLines;
        private long countAdvanceNanos;
        private long thresholdAdvanceNanos;
        private long scoreNanos;
        private long thresholdNanos;
        private long thresholdUpdateNanos;
        private long countUpdateNanos;
        private long predictions;
        private long anchorControlled;
        private long dynamicControlled;
        private double scoreSum;
        private double thresholdSum;
        private final Map<Integer, Long> predictedTemplates = new HashMap<>();

        private OnlineProfile(String datasetName, int startBucket, int endBucket, Config config) {
            this.datasetName = datasetName;
            this.startBucket = startBucket;
            this.endBucket = endBucket;
            intervalLines = max(1, config.progressInterval);
        }

        private void maybeReport(int bucketIndex, OnlineBaselineState online) {
            if (lines - lastReportLines >= intervalLines) {
                report("progress", bucketIndex, online);
                lastReportLines = lines;
            }
        }

        private void recordDecision(Dataset dataset, int lineIndex, double score, double threshold, boolean prediction,
                boolean anchorControlledDecision) {
            scoreSum += score;
            thresholdSum += threshold;
            if (anchorControlledDecision) {
                ++anchorControlled;
            } else {
                ++dynamicControlled;
            }
            if (prediction) {
                ++predictions;
                int eventId = dataset.eventIds.values[lineIndex];
                predictedTemplates.put(eventId, predictedTemplates.getOrDefault(eventId, 0L) + 1L);
            }
        }

        private void report(String phase, int bucketIndex, OnlineBaselineState online) {
            double elapsed = secondsSince(startNanos);
            long measured = countAdvanceNanos + thresholdAdvanceNanos + scoreNanos + thresholdNanos
                    + thresholdUpdateNanos + countUpdateNanos;
            System.out.printf(Locale.ROOT,
                    "online_profile_eval dataset=%s phase=%s buckets=%d/%d lines=%d elapsed=%.3f lines_per_sec=%.1f count_advance=%.3f threshold_advance=%.3f score=%.3f threshold=%.3f threshold_update=%.3f count_update=%.3f measured=%.3f prediction_rate=%.6f anchor_controlled=%.6f dynamic_controlled=%.6f avg_score=%.6f avg_threshold=%.6f top_pred_templates=%s count_state=%s cap_state=%s%n",
                    datasetName, phase, max(0, bucketIndex - startBucket + 1), max(0, endBucket - startBucket), lines,
                    elapsed, lines / max(1.0e-9, elapsed), seconds(countAdvanceNanos), seconds(thresholdAdvanceNanos),
                    seconds(scoreNanos), seconds(thresholdNanos), seconds(thresholdUpdateNanos),
                    seconds(countUpdateNanos), seconds(measured), predictions / max(1.0, (double) lines),
                    anchorControlled / max(1.0, (double) lines), dynamicControlled / max(1.0, (double) lines),
                    scoreSum / max(1.0, (double) lines), thresholdSum / max(1.0, (double) lines),
                    topPredictedTemplates(), online.entrySummary(), online.capSummary());
        }

        private String topPredictedTemplates() {
            if (predictedTemplates.isEmpty()) {
                return "-";
            }
            List<Map.Entry<Integer, Long>> entries = new ArrayList<>(predictedTemplates.entrySet());
            entries.sort((a, b) -> Long.compare(b.getValue(), a.getValue()));
            StringBuilder builder = new StringBuilder();
            int limit = min(5, entries.size());
            for (int i = 0; i < limit; i++) {
                if (i > 0) {
                    builder.append(';');
                }
                Map.Entry<Integer, Long> entry = entries.get(i);
                builder.append(entry.getKey()).append(':').append(entry.getValue());
            }
            return builder.toString();
        }
    }

    private static boolean onlineUpdatePrediction(OnlineThresholdState thresholds, Dataset dataset, int lineIndex,
            double decisionQuantile, long bucketKey, double score, double decisionThreshold, Config config) {
        if (!Double.isFinite(config.onlineUpdateGuardQuantile)) {
            return onlinePrediction(score, decisionThreshold, config);
        }
        double guardQuantile = max(0.0, min(1.0, config.onlineUpdateGuardQuantile));
        double guardThreshold = guardQuantile == decisionQuantile ? decisionThreshold
                : thresholds.quantile(dataset, lineIndex, guardQuantile, bucketKey);
        return onlinePrediction(score, guardThreshold, config);
    }

    private static List<FdrResult> evaluateDatasetTopKPolicy(String datasetName, Config config) throws IOException {
        Dataset dataset = loadDataset(datasetName, config);
        Split split = Split.from(dataset, config);
        TrainStats stats = statsForSplit(dataset, split, config);
        List<ContextSet> contexts = contexts(dataset, stats, split.trainEnd, config);
        List<FdrResult> results = new ArrayList<>();
        for (ContextSet context : contexts) {
            double contextRecall = contextLineRecall(dataset, context, split.calibrationEnd, dataset.buckets.size());
            for (double value : config.fdrQValues) {
                int k = max(1, (int) Math.round(value));
                FdrResult topK = evaluateTopKPerBucket(datasetName, dataset, stats, context, k, split.calibrationEnd,
                        dataset.buckets.size(), contextRecall, config);
                results.add(topK);
                System.out.println(topK.toCsv());
            }
        }
        return results;
    }

    private static List<OracleResult> evaluateDatasetOracle(String datasetName, Config config) throws IOException {
        Dataset dataset = loadDataset(datasetName, config);
        Split split = Split.from(dataset, config);
        TrainStats stats = statsForSplit(dataset, split, config);
        List<ContextSet> contexts = contexts(dataset, stats, split.trainEnd, config);
        List<OracleResult> results = new ArrayList<>();
        for (ContextSet context : contexts) {
            OracleResult configured = oracleDiagnosticScoreMode(datasetName, config.scoreMode, dataset, stats, context,
                    split.calibrationEnd, dataset.buckets.size());
            results.add(configured);
            System.out.println(configured.toCsv());
            OracleResult combined = oracleDiagnostic(datasetName, "combined_line_score", dataset, stats, context,
                    split.calibrationEnd, dataset.buckets.size(), true);
            results.add(combined);
            System.out.println(combined.toCsv());
            for (LineScorer scorer : LineScorer.suite()) {
                OracleResult result = oracleDiagnostic(datasetName, scorer.name(), dataset, stats, context,
                        split.calibrationEnd, dataset.buckets.size(), false);
                results.add(result);
                System.out.println(result.toCsv());
            }
        }
        return results;
    }

    private static List<TemplateDiagnostic> evaluateDatasetTemplateDiagnostic(String datasetName, Config config)
            throws IOException {
        Dataset dataset = loadDataset(datasetName, config);
        Split split = Split.from(dataset, config);
        TrainStats stats = statsForSplit(dataset, split, config);
        ContextSet context = ContextSet.all("none", dataset.buckets.size());
        DiagnosticThreshold threshold;
        if (Double.isFinite(config.diagnosticQuantile)) {
            double[] calibrationScores = calibrationLineScores(dataset, stats, context,
                    calibrationStartBucket(split, config), split.calibrationEnd, config);
            java.util.Arrays.sort(calibrationScores);
            double value = quantile(calibrationScores, config.diagnosticQuantile);
            threshold = evaluateFixedThreshold(dataset, stats, context, split.calibrationEnd, dataset.buckets.size(),
                    config.scoreMode, value);
        } else {
            threshold = chooseScoreModeThreshold(dataset, stats, context, split.calibrationEnd, dataset.buckets.size(),
                    config.scoreMode, config.diagnosticRecallTarget);
        }
        System.out.printf(Locale.ROOT,
                "diagnostic dataset=%s score_mode=%s threshold=%.8f precision=%.6f recall=%.6f f1=%.6f%n", datasetName,
                config.scoreMode, threshold.threshold, threshold.precision, threshold.recall, threshold.f1);
        TemplateAggregate[] aggregates = new TemplateAggregate[max(1, dataset.templateCount)];
        for (int eventId = 0; eventId < aggregates.length; eventId++) {
            aggregates[eventId] = new TemplateAggregate(eventId);
        }
        for (int bucketIndex = split.calibrationEnd; bucketIndex < dataset.buckets.size(); bucketIndex++) {
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                int eventId = dataset.eventIds.values[i];
                double score = lineScore(dataset, stats, context, i, config.scoreMode);
                boolean label = dataset.labels.values[i] != 0;
                boolean prediction = score >= threshold.threshold;
                aggregates[eventId].add(dataset, i, score, label, prediction, config.diagnosticSamples);
            }
        }
        List<TemplateDiagnostic> results = new ArrayList<>();
        collectTemplateDiagnostics(datasetName, dataset, stats, config, threshold, aggregates, "fp", results);
        collectTemplateDiagnostics(datasetName, dataset, stats, config, threshold, aggregates, "fn", results);
        collectTemplateDiagnostics(datasetName, dataset, stats, config, threshold, aggregates, "tp", results);
        return results;
    }

    private static DiagnosticThreshold evaluateFixedThreshold(Dataset dataset, TrainStats stats, ContextSet context,
            int startBucket, int endBucket, String scoreMode, double threshold) {
        Metrics metrics = new Metrics(0, 0, 0, 0);
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                boolean prediction = context.candidates[bucketIndex]
                        && lineScore(dataset, stats, context, i, scoreMode) >= threshold;
                metrics.add(dataset.labels.values[i] != 0, prediction);
            }
        }
        return new DiagnosticThreshold(threshold, metrics.precision(), metrics.recall(), metrics.f1());
    }

    private static DiagnosticThreshold chooseScoreModeThreshold(Dataset dataset, TrainStats stats, ContextSet context,
            int startBucket, int endBucket, String scoreMode, double targetRecall) {
        int count = dataset.countLines(startBucket, endBucket, context);
        long[] packed = new long[count];
        int index = 0;
        long positives = 0;
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            if (!context.candidates[bucketIndex]) {
                continue;
            }
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                boolean label = dataset.labels.values[i] != 0;
                packed[index++] = pack(lineScore(dataset, stats, context, i, scoreMode), label);
                if (label) {
                    ++positives;
                }
            }
        }
        if (index == 0 || positives == 0) {
            return new DiagnosticThreshold(Double.POSITIVE_INFINITY, 0, 0, 0);
        }
        java.util.Arrays.sort(packed, 0, index);
        long tp = 0;
        double bestPrecision = -1.0;
        double bestRecall = 0.0;
        double bestF1 = 0.0;
        double bestThreshold = Double.POSITIVE_INFINITY;
        for (int i = 0; i < index; i++) {
            if ((packed[i] & 1L) != 0) {
                ++tp;
            }
            double precision = tp / (double) (i + 1L);
            double recall = tp / (double) positives;
            double f1 = precision + recall == 0 ? 0.0 : 2.0 * precision * recall / (precision + recall);
            if (recall >= targetRecall && precision > bestPrecision) {
                bestPrecision = precision;
                bestRecall = recall;
                bestF1 = f1;
                bestThreshold = unpackScore(packed[i]);
            }
        }
        if (bestPrecision < 0.0) {
            return new DiagnosticThreshold(Double.POSITIVE_INFINITY, 0, 0, 0);
        }
        return new DiagnosticThreshold(bestThreshold, bestPrecision, bestRecall, bestF1);
    }

    private static void collectTemplateDiagnostics(String datasetName, Dataset dataset, TrainStats stats, Config config,
            DiagnosticThreshold threshold, TemplateAggregate[] aggregates, String group,
            List<TemplateDiagnostic> results) {
        List<TemplateAggregate> sorted = new ArrayList<>();
        for (TemplateAggregate aggregate : aggregates) {
            if ("fp".equals(group) && aggregate.fp > 0 || "fn".equals(group) && aggregate.fn > 0
                    || "tp".equals(group) && aggregate.tp > 0) {
                sorted.add(aggregate);
            }
        }
        sorted.sort((a, b) -> Long.compare(b.countFor(group), a.countFor(group)));
        for (int i = 0; i < min(config.diagnosticTopTemplates, sorted.size()); i++) {
            TemplateAggregate aggregate = sorted.get(i);
            results.add(aggregate.toDiagnostic(datasetName, dataset, stats, config.scoreMode, group, threshold));
        }
    }

    private static Dataset loadDataset(String datasetName, Config config) throws IOException {
        int sampleModulo = "thunderbird".equals(datasetName) ? config.thunderbirdSampleModulo : config.bglSampleModulo;
        String input = "thunderbird".equals(datasetName) ? config.thunderbirdPath : config.bglPath;
        return Dataset.load(datasetName, input, sampleModulo, config);
    }

    private static TrainStats statsForSplit(Dataset dataset, Split split, Config config) {
        switch (config.statsPeriod) {
        case "calibration":
            return TrainStats.fromRange(dataset, split.trainEnd, split.calibrationEnd, config);
        case "warmup":
            return TrainStats.fromRange(dataset, 0, split.calibrationEnd, config);
        case "train":
        default:
            return TrainStats.from(dataset, split.trainEnd, config);
        }
    }

    private static int calibrationStartBucket(Split split, Config config) {
        return "warmup".equals(config.statsPeriod) ? 0 : split.trainEnd;
    }

    private static List<ContextSet> contexts(Dataset dataset, TrainStats stats, int trainEnd, Config config) {
        List<ContextSet> contexts = new ArrayList<>();
        contexts.add(ContextSet.all("none", dataset.buckets.size()));
        for (double anomalyRate : config.contextAnomalyRates) {
            contexts.add(ContextSet.trcf(dataset, stats, trainEnd, anomalyRate, 1, config));
            contexts.add(ContextSet.trcf(dataset, stats, trainEnd, anomalyRate, 4, config));
        }
        return contexts;
    }

    private static Result evaluate(String datasetName, Dataset dataset, TrainStats stats, LineScorer scorer,
            ContextSet context, int trainEnd, int validationEnd, Config config) {
        ThresholdChoice threshold = chooseThreshold(dataset, stats, scorer, context, trainEnd, validationEnd);
        Metrics testMetrics = lineMetrics(dataset, stats, scorer, context, threshold.threshold, validationEnd,
                dataset.buckets.size());
        TopKMetrics topK = topKMetrics(dataset, stats, scorer, context, validationEnd, dataset.buckets.size(), 5);
        long testAnomalies = anomalousLines(dataset, validationEnd, dataset.buckets.size());
        long coveredAnomalies = coveredAnomalousLines(dataset, context, validationEnd, dataset.buckets.size());
        double candidateRecall = testAnomalies == 0 ? 0.0 : coveredAnomalies / (double) testAnomalies;
        return new Result(datasetName, scorer.name(), context.name, context.anomalyRate, context.shingleSize,
                threshold.threshold, threshold.validationPrecision, threshold.validationRecall, threshold.validationF1,
                candidateRecall, testMetrics, topK);
    }

    private static ThresholdChoice chooseThreshold(Dataset dataset, TrainStats stats, LineScorer scorer,
            ContextSet context, int startBucket, int endBucket) {
        int count = dataset.countLines(startBucket, endBucket, context);
        long[] packed = new long[count];
        int index = 0;
        long positives = 0;
        for (int i = 0; i < dataset.labels.size; i++) {
            int bucketIndex = dataset.bucketIndexes.values[i];
            if (bucketIndex < startBucket || bucketIndex >= endBucket || !context.candidates[bucketIndex]) {
                continue;
            }
            double score = scorer.score(dataset, stats, i);
            packed[index++] = pack(score, dataset.labels.values[i] != 0);
            if (dataset.labels.values[i] != 0) {
                ++positives;
            }
        }
        if (index == 0 || positives == 0) {
            return new ThresholdChoice(Double.POSITIVE_INFINITY, 0, 0, 0);
        }
        java.util.Arrays.sort(packed, 0, index);
        long tp = 0;
        double bestF1 = 0;
        double bestPrecision = 0;
        double bestRecall = 0;
        double bestThreshold = Double.POSITIVE_INFINITY;
        for (int i = 0; i < index; i++) {
            if ((packed[i] & 1L) != 0) {
                ++tp;
            }
            long predicted = i + 1L;
            double precision = tp / (double) predicted;
            double recall = tp / (double) positives;
            double f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
            if (f1 > bestF1) {
                bestF1 = f1;
                bestPrecision = precision;
                bestRecall = recall;
                bestThreshold = unpackScore(packed[i]);
            }
        }
        return new ThresholdChoice(bestThreshold, bestPrecision, bestRecall, bestF1);
    }

    private static Metrics lineMetrics(Dataset dataset, TrainStats stats, LineScorer scorer, ContextSet context,
            double threshold, int startBucket, int endBucket) {
        long tp = 0;
        long fp = 0;
        long fn = 0;
        long tn = 0;
        for (int i = 0; i < dataset.labels.size; i++) {
            int bucketIndex = dataset.bucketIndexes.values[i];
            if (bucketIndex < startBucket || bucketIndex >= endBucket) {
                continue;
            }
            boolean label = dataset.labels.values[i] != 0;
            boolean prediction = context.candidates[bucketIndex] && scorer.score(dataset, stats, i) >= threshold;
            if (label && prediction) {
                ++tp;
            } else if (prediction) {
                ++fp;
            } else if (label) {
                ++fn;
            } else {
                ++tn;
            }
        }
        return new Metrics(tp, fp, fn, tn);
    }

    private static TopKMetrics topKMetrics(Dataset dataset, TrainStats stats, LineScorer scorer, ContextSet context,
            int startBucket, int endBucket, int maxK) {
        TopK[] topByBucket = new TopK[dataset.buckets.size()];
        long totalAnomalies = 0;
        for (int i = 0; i < dataset.labels.size; i++) {
            int bucketIndex = dataset.bucketIndexes.values[i];
            if (bucketIndex < startBucket || bucketIndex >= endBucket) {
                continue;
            }
            if (dataset.labels.values[i] != 0) {
                ++totalAnomalies;
            }
            if (!context.candidates[bucketIndex]) {
                continue;
            }
            TopK top = topByBucket[bucketIndex];
            if (top == null) {
                top = new TopK(maxK);
                topByBucket[bucketIndex] = top;
            }
            top.add(scorer.score(dataset, stats, i), dataset.labels.values[i] != 0);
        }
        int[] predicted = new int[maxK];
        int[] truePositive = new int[maxK];
        for (TopK top : topByBucket) {
            if (top == null) {
                continue;
            }
            top.sortDescending();
            for (int k = 1; k <= maxK; k++) {
                int selected = min(k, top.size);
                predicted[k - 1] += selected;
                for (int i = 0; i < selected; i++) {
                    if (top.labels[i]) {
                        ++truePositive[k - 1];
                    }
                }
            }
        }
        return new TopKMetrics(predicted, truePositive, totalAnomalies);
    }

    private static double[] calibrationLineScores(Dataset dataset, TrainStats stats, ContextSet context,
            int startBucket, int endBucket, Config config) {
        double[] scores = new double[dataset.countLines(startBucket, endBucket)];
        int index = 0;
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                scores[index++] = lineScore(dataset, stats, context, i, config.scoreMode);
            }
        }
        return scores;
    }

    private static double[] calibrationCellScores(Dataset dataset, TrainStats stats, ContextSet context,
            int startBucket, int endBucket) {
        int count = 0;
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            count += dataset.buckets.get(bucketIndex).eventCounts.size();
        }
        double[] scores = new double[count];
        int index = 0;
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            Bucket bucket = dataset.buckets.get(bucketIndex);
            for (int eventId : bucket.eventCounts.keySet()) {
                scores[index++] = fdrCellScore(bucket, stats, context, bucketIndex, eventId);
            }
        }
        return scores;
    }

    private static FdrResult evaluateLineBh(String datasetName, Dataset dataset, TrainStats stats, ContextSet context,
            double[] calibrationScores, double q, int startBucket, int endBucket, double contextLineRecall) {
        throw new UnsupportedOperationException("use overload with config");
    }

    private static FdrResult evaluateLineBh(String datasetName, Dataset dataset, TrainStats stats, ContextSet context,
            double[] calibrationScores, double q, int startBucket, int endBucket, double contextLineRecall,
            Config config) {
        Metrics metrics = new Metrics(0, 0, 0, 0);
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            int start = dataset.lineStarts[bucketIndex];
            int end = dataset.lineStarts[bucketIndex + 1];
            int size = end - start;
            if (size <= 0) {
                continue;
            }
            double[] pValues = new double[size];
            for (int offset = 0; offset < size; offset++) {
                pValues[offset] = empiricalUpperTailPValue(calibrationScores,
                        lineScore(dataset, stats, context, start + offset, config.scoreMode));
            }
            double cutoff = bhCutoff(pValues, q);
            for (int offset = 0; offset < size; offset++) {
                metrics = metrics.add(dataset.labels.values[start + offset] != 0,
                        cutoff >= 0.0 && pValues[offset] <= cutoff);
            }
        }
        return new FdrResult(datasetName, "line_bh_" + config.scoreMode, context.name, context.anomalyRate,
                context.shingleSize, q, calibrationScores.length, contextLineRecall, metrics);
    }

    private static FdrResult evaluateCellBh(String datasetName, Dataset dataset, TrainStats stats, ContextSet context,
            double[] calibrationScores, double q, int startBucket, int endBucket, double contextLineRecall) {
        Metrics metrics = new Metrics(0, 0, 0, 0);
        boolean[] selectedEvents = new boolean[max(1, dataset.templateCount)];
        List<Integer> selected = new ArrayList<>();
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            Bucket bucket = dataset.buckets.get(bucketIndex);
            int size = bucket.eventCounts.size();
            if (size <= 0) {
                continue;
            }
            int[] eventIds = new int[size];
            double[] pValues = new double[size];
            int index = 0;
            for (int eventId : bucket.eventCounts.keySet()) {
                eventIds[index] = eventId;
                pValues[index] = empiricalUpperTailPValue(calibrationScores,
                        fdrCellScore(bucket, stats, context, bucketIndex, eventId));
                ++index;
            }
            double cutoff = bhCutoff(pValues, q);
            if (cutoff >= 0.0) {
                for (int i = 0; i < size; i++) {
                    if (pValues[i] <= cutoff) {
                        selectedEvents[eventIds[i]] = true;
                        selected.add(eventIds[i]);
                    }
                }
            }
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                metrics = metrics.add(dataset.labels.values[i] != 0, selectedEvents[dataset.eventIds.values[i]]);
            }
            for (int eventId : selected) {
                selectedEvents[eventId] = false;
            }
            selected.clear();
        }
        return new FdrResult(datasetName, "cell_bh", context.name, context.anomalyRate, context.shingleSize, q,
                calibrationScores.length, contextLineRecall, metrics);
    }

    private static FdrResult evaluateFixedQuantile(String datasetName, Dataset dataset, TrainStats stats,
            ContextSet context, double[] calibrationScores, double quantile, int startBucket, int endBucket,
            double contextLineRecall, Config config) {
        double threshold = quantile(calibrationScores, quantile);
        Metrics metrics = new Metrics(0, 0, 0, 0);
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                boolean prediction = context.candidates[bucketIndex]
                        && lineScore(dataset, stats, context, i, config.scoreMode) >= threshold;
                metrics.add(dataset.labels.values[i] != 0, prediction);
            }
        }
        return new FdrResult(datasetName,
                String.format(Locale.ROOT, "fixed_quantile_%s_p%.5f", config.scoreMode, quantile), context.name,
                context.anomalyRate, context.shingleSize, quantile, calibrationScores.length, contextLineRecall,
                metrics);
    }

    private static FdrResult evaluateQuantileSeedExpand(String datasetName, Dataset dataset, TrainStats stats,
            ContextSet context, double[] calibrationScores, double seedQuantile, double expandQuantile, int startBucket,
            int endBucket, double contextLineRecall, Config config) {
        double seedThreshold = quantile(calibrationScores, seedQuantile);
        double expandThreshold = quantile(calibrationScores, expandQuantile);
        Metrics metrics = new Metrics(0, 0, 0, 0);
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            Map<Long, Boolean> seedKeys = new HashMap<>();
            for (int seedBucket = max(startBucket, bucketIndex - config.expandBuckets); seedBucket <= min(endBucket - 1,
                    bucketIndex + config.expandBuckets); seedBucket++) {
                for (int i = dataset.lineStarts[seedBucket]; i < dataset.lineStarts[seedBucket + 1]; i++) {
                    if (context.candidates[seedBucket]
                            && lineScore(dataset, stats, context, i, config.scoreMode) >= seedThreshold) {
                        seedKeys.put(entityEventKey(dataset.entityIds.values[i], dataset.eventIds.values[i]),
                                Boolean.TRUE);
                    }
                }
            }
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                long key = entityEventKey(dataset.entityIds.values[i], dataset.eventIds.values[i]);
                boolean prediction = context.candidates[bucketIndex] && seedKeys.containsKey(key)
                        && lineScore(dataset, stats, context, i, config.scoreMode) >= expandThreshold;
                metrics.add(dataset.labels.values[i] != 0, prediction);
            }
        }
        double q = seedQuantile + expandQuantile / 100.0;
        return new FdrResult(datasetName,
                String.format(Locale.ROOT, "quantile_seed_expand_%s_s%.5f_x%.5f_e%d", config.scoreMode, seedQuantile,
                        expandQuantile, config.expandBuckets),
                context.name, context.anomalyRate, context.shingleSize, q, calibrationScores.length, contextLineRecall,
                metrics);
    }

    private static FdrResult evaluateTopKPerBucket(String datasetName, Dataset dataset, TrainStats stats,
            ContextSet context, int k, int startBucket, int endBucket, double contextLineRecall, Config config) {
        Metrics metrics = new Metrics(0, 0, 0, 0);
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            int start = dataset.lineStarts[bucketIndex];
            int end = dataset.lineStarts[bucketIndex + 1];
            int size = end - start;
            if (size <= 0) {
                continue;
            }
            long[] packed = new long[size];
            for (int offset = 0; offset < size; offset++) {
                int lineIndex = start + offset;
                double score = context.candidates[bucketIndex]
                        ? lineScore(dataset, stats, context, lineIndex, config.scoreMode)
                        : Double.NEGATIVE_INFINITY;
                packed[offset] = pack(score, dataset.labels.values[lineIndex] != 0);
            }
            java.util.Arrays.sort(packed);
            int selected = min(k, size);
            for (int offset = 0; offset < size; offset++) {
                boolean prediction = offset < selected && context.candidates[bucketIndex];
                boolean label = (packed[offset] & 1L) != 0;
                metrics.add(label, prediction);
            }
        }
        return new FdrResult(datasetName, "topk_per_bucket_" + config.scoreMode + "_k" + k, context.name,
                context.anomalyRate, context.shingleSize, k, 0, contextLineRecall, metrics);
    }

    private static FdrResult evaluateRollingLineBh(String datasetName, Dataset dataset, TrainStats stats,
            ContextSet context, double q, int startBucket, int endBucket, double contextLineRecall, Config config,
            boolean conditional) {
        RollingLineCalibrator calibrator = new RollingLineCalibrator(config.rollingScoreMax, config.rollingBins,
                conditional);
        int initStart = max(0, startBucket - config.rollingHorizonBuckets);
        for (int bucketIndex = initStart; bucketIndex < startBucket; bucketIndex++) {
            calibrator.add(bucketScores(dataset, stats, context, bucketIndex, config, conditional));
        }
        while (calibrator.size() > config.rollingHorizonBuckets) {
            calibrator.removeOldest();
        }
        Metrics metrics = new Metrics(0, 0, 0, 0);
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            ScoreBucket current = bucketScores(dataset, stats, context, bucketIndex, config, conditional);
            double[] pValues = new double[current.scores.length];
            for (int i = 0; i < current.scores.length; i++) {
                pValues[i] = calibrator.pValue(current.scores[i], current.classes[i]);
            }
            double cutoff = bhCutoff(pValues, q);
            boolean[] predictions = new boolean[current.scores.length];
            int kept = 0;
            for (int offset = 0; offset < current.scores.length; offset++) {
                boolean prediction = cutoff >= 0.0 && pValues[offset] <= cutoff;
                predictions[offset] = prediction;
                metrics.add(dataset.labels.values[current.startLine + offset] != 0, prediction);
                if (!prediction || !config.excludeAlertUpdates) {
                    current.scores[kept] = current.scores[offset];
                    current.classes[kept] = current.classes[offset];
                    ++kept;
                }
            }
            calibrator.add(current.truncate(kept));
            while (calibrator.size() > config.rollingHorizonBuckets) {
                calibrator.removeOldest();
            }
        }
        String method = (conditional ? "rolling_conditional_bh_" : "rolling_line_bh_") + config.scoreMode + "_h"
                + config.rollingHorizonBuckets;
        return new FdrResult(datasetName, method, context.name, context.anomalyRate, context.shingleSize, q,
                calibrator.totalCount(), contextLineRecall, metrics);
    }

    private static ScoreBucket bucketScores(Dataset dataset, TrainStats stats, ContextSet context, int bucketIndex,
            Config config, boolean conditional) {
        int start = dataset.lineStarts[bucketIndex];
        int end = dataset.lineStarts[bucketIndex + 1];
        double[] scores = new double[end - start];
        int[] classes = new int[end - start];
        for (int i = start; i < end; i++) {
            int offset = i - start;
            scores[offset] = lineScore(dataset, stats, context, i, config.scoreMode);
            classes[offset] = conditional ? keywordClass(dataset.keywordScores.values[i]) : 0;
        }
        return new ScoreBucket(start, scores, classes);
    }

    private static int keywordClass(int keyword) {
        if (keyword >= 8) {
            return 3;
        }
        if (keyword >= 4) {
            return 2;
        }
        if (keyword >= 1) {
            return 1;
        }
        return 0;
    }

    private static FdrResult evaluateRollingTemplateSpikeBh(String datasetName, Dataset dataset, ContextSet context,
            double q, int startBucket, int endBucket, double contextLineRecall, Config config) {
        RollingTemplateBaseline baseline = new RollingTemplateBaseline(dataset.templateCount);
        int initStart = max(0, startBucket - config.rollingHorizonBuckets);
        for (int bucketIndex = initStart; bucketIndex < startBucket; bucketIndex++) {
            baseline.add(bucketIndex, dataset.buckets.get(bucketIndex));
        }
        while (baseline.size() > config.rollingHorizonBuckets) {
            baseline.removeOldest(dataset);
        }
        Metrics metrics = new Metrics(0, 0, 0, 0);
        boolean[] selectedEvents = new boolean[max(1, dataset.templateCount)];
        List<Integer> selected = new ArrayList<>();
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            Bucket bucket = dataset.buckets.get(bucketIndex);
            int size = bucket.eventCounts.size();
            int[] eventIds = new int[size];
            double[] pValues = new double[size];
            int index = 0;
            for (Map.Entry<Integer, Integer> entry : bucket.eventCounts.entrySet()) {
                eventIds[index] = entry.getKey();
                pValues[index] = baseline.poissonUpperTail(bucket.totalCount, entry.getKey(), entry.getValue());
                ++index;
            }
            double cutoff = bhCutoff(pValues, q);
            if (cutoff >= 0.0) {
                for (int i = 0; i < size; i++) {
                    if (pValues[i] <= cutoff) {
                        selectedEvents[eventIds[i]] = true;
                        selected.add(eventIds[i]);
                    }
                }
            }
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                boolean prediction = selectedEvents[dataset.eventIds.values[i]];
                metrics.add(dataset.labels.values[i] != 0, prediction);
            }
            for (int eventId : selected) {
                selectedEvents[eventId] = false;
            }
            selected.clear();
            baseline.add(bucketIndex, bucket);
            while (baseline.size() > config.rollingHorizonBuckets) {
                baseline.removeOldest(dataset);
            }
        }
        return new FdrResult(datasetName, "rolling_template_spike_bh_h" + config.rollingHorizonBuckets, context.name,
                context.anomalyRate, context.shingleSize, q, baseline.total, contextLineRecall, metrics);
    }

    private static FdrResult evaluateSeedExpand(String datasetName, Dataset dataset, TrainStats stats,
            ContextSet context, double q, int startBucket, int endBucket, double contextLineRecall, Config config) {
        RollingLineCalibrator calibrator = new RollingLineCalibrator(config.rollingScoreMax, config.rollingBins, false);
        int initStart = max(0, startBucket - config.rollingHorizonBuckets);
        for (int bucketIndex = initStart; bucketIndex < startBucket; bucketIndex++) {
            calibrator.add(bucketScores(dataset, stats, context, bucketIndex, config, false));
        }
        while (calibrator.size() > config.rollingHorizonBuckets) {
            calibrator.removeOldest();
        }
        Metrics metrics = new Metrics(0, 0, 0, 0);
        int[] activeEvents = new int[max(1, dataset.templateCount)];
        List<Integer> touched = new ArrayList<>();
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            ScoreBucket current = bucketScores(dataset, stats, context, bucketIndex, config, false);
            boolean[] seedEvent = new boolean[activeEvents.length];
            for (int offset = 0; offset < current.scores.length; offset++) {
                int lineIndex = current.startLine + offset;
                double pValue = calibrator.pValue(current.scores[offset], 0);
                int keyword = dataset.keywordScores.values[lineIndex];
                boolean strongSeed = keyword >= 8 && pValue <= min(0.05, q * 20.0);
                boolean scoreSeed = pValue <= q;
                if (scoreSeed || strongSeed) {
                    int eventId = dataset.eventIds.values[lineIndex];
                    if (!seedEvent[eventId]) {
                        seedEvent[eventId] = true;
                        touched.add(eventId);
                    }
                    activeEvents[eventId] = max(activeEvents[eventId], config.expandBuckets + 1);
                }
            }
            for (int offset = 0; offset < current.scores.length; offset++) {
                int lineIndex = current.startLine + offset;
                int eventId = dataset.eventIds.values[lineIndex];
                boolean prediction = activeEvents[eventId] > 0;
                metrics.add(dataset.labels.values[lineIndex] != 0, prediction);
            }
            calibrator.add(current);
            while (calibrator.size() > config.rollingHorizonBuckets) {
                calibrator.removeOldest();
            }
            for (int eventId = 0; eventId < activeEvents.length; eventId++) {
                if (activeEvents[eventId] > 0) {
                    --activeEvents[eventId];
                }
            }
            for (int eventId : touched) {
                seedEvent[eventId] = false;
            }
            touched.clear();
        }
        return new FdrResult(datasetName,
                "seed_expand_" + config.scoreMode + "_h" + config.rollingHorizonBuckets + "_e" + config.expandBuckets,
                context.name, context.anomalyRate, context.shingleSize, q, calibrator.totalCount(), contextLineRecall,
                metrics);
    }

    private static TailCalibration tailCalibration(Dataset dataset, TrainStats stats, ContextSet context,
            int startBucket, int endBucket, Config config, boolean conditional) {
        List<Double>[] byClass = new List[conditional ? 4 : 1];
        for (int i = 0; i < byClass.length; i++) {
            byClass[i] = new ArrayList<>();
        }
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                int scoreClass = conditional ? keywordClass(dataset.keywordScores.values[i]) : 0;
                byClass[scoreClass].add(lineScore(dataset, stats, context, i, config.scoreMode));
            }
        }
        double[][] sorted = new double[byClass.length][];
        int total = 0;
        for (int i = 0; i < byClass.length; i++) {
            sorted[i] = new double[byClass[i].size()];
            for (int j = 0; j < sorted[i].length; j++) {
                sorted[i][j] = byClass[i].get(j);
            }
            java.util.Arrays.sort(sorted[i]);
            total += sorted[i].length;
        }
        return new TailCalibration(sorted, total);
    }

    private static FdrResult evaluateTailExcess(String datasetName, Dataset dataset, TrainStats stats,
            ContextSet context, TailCalibration calibration, double targetPrecision, int startBucket, int endBucket,
            double contextLineRecall, Config config, boolean conditional) {
        Metrics metrics = new Metrics(0, 0, 0, 0);
        for (int blockStart = startBucket; blockStart < endBucket; blockStart += config.tailBlockBuckets) {
            int blockEnd = min(endBucket, blockStart + config.tailBlockBuckets);
            ScoreBlock block = scoreBlock(dataset, stats, context, blockStart, blockEnd, config, conditional);
            boolean[] predictions = tailExcessPredictions(block, calibration, targetPrecision, config, conditional);
            for (int i = 0; i < block.scores.length; i++) {
                metrics.add(dataset.labels.values[block.lineIndexes[i]] != 0, predictions[i]);
            }
        }
        String method = (conditional ? "conditional_tail_excess_" : "tail_excess_") + config.scoreMode + "_b"
                + config.tailBlockBuckets;
        return new FdrResult(datasetName, method, context.name, context.anomalyRate, context.shingleSize,
                targetPrecision, calibration.totalCount, contextLineRecall, metrics);
    }

    private static FdrResult evaluateTailSeedExpand(String datasetName, Dataset dataset, TrainStats stats,
            ContextSet context, TailCalibration calibration, double targetPrecision, int startBucket, int endBucket,
            double contextLineRecall, Config config) {
        Metrics metrics = new Metrics(0, 0, 0, 0);
        for (int blockStart = startBucket; blockStart < endBucket; blockStart += config.tailBlockBuckets) {
            int blockEnd = min(endBucket, blockStart + config.tailBlockBuckets);
            ScoreBlock block = scoreBlock(dataset, stats, context, blockStart, blockEnd, config, false);
            boolean[] seeds = tailExcessPredictions(block, calibration, targetPrecision, config, false);
            Map<Long, List<Integer>> seedBuckets = new HashMap<>();
            for (int i = 0; i < block.scores.length; i++) {
                if (seeds[i]) {
                    int lineIndex = block.lineIndexes[i];
                    long key = entityEventKey(dataset.entityIds.values[lineIndex], dataset.eventIds.values[lineIndex]);
                    seedBuckets.computeIfAbsent(key, ignored -> new ArrayList<>())
                            .add(dataset.bucketIndexes.values[lineIndex]);
                }
            }
            for (int i = 0; i < block.scores.length; i++) {
                int lineIndex = block.lineIndexes[i];
                long key = entityEventKey(dataset.entityIds.values[lineIndex], dataset.eventIds.values[lineIndex]);
                boolean prediction = false;
                List<Integer> buckets = seedBuckets.get(key);
                if (buckets != null) {
                    int bucketIndex = dataset.bucketIndexes.values[lineIndex];
                    for (int seedBucket : buckets) {
                        if (Math.abs(bucketIndex - seedBucket) <= config.expandBuckets) {
                            prediction = true;
                            break;
                        }
                    }
                }
                metrics.add(dataset.labels.values[lineIndex] != 0, prediction);
            }
        }
        String method = "tail_seed_expand_" + config.scoreMode + "_b" + config.tailBlockBuckets + "_e"
                + config.expandBuckets;
        return new FdrResult(datasetName, method, context.name, context.anomalyRate, context.shingleSize,
                targetPrecision, calibration.totalCount, contextLineRecall, metrics);
    }

    private static ScoreBlock scoreBlock(Dataset dataset, TrainStats stats, ContextSet context, int startBucket,
            int endBucket, Config config, boolean conditional) {
        int count = dataset.countLines(startBucket, endBucket);
        double[] scores = new double[count];
        int[] classes = new int[count];
        int[] lineIndexes = new int[count];
        int index = 0;
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                scores[index] = lineScore(dataset, stats, context, i, config.scoreMode);
                classes[index] = conditional ? keywordClass(dataset.keywordScores.values[i]) : 0;
                lineIndexes[index] = i;
                ++index;
            }
        }
        return new ScoreBlock(scores, classes, lineIndexes);
    }

    private static boolean[] tailExcessPredictions(ScoreBlock block, TailCalibration calibration,
            double targetPrecision, Config config, boolean conditional) {
        boolean[] predictions = new boolean[block.scores.length];
        int classCount = conditional ? calibration.sortedScoresByClass.length : 1;
        for (int scoreClass = 0; scoreClass < classCount; scoreClass++) {
            double threshold = tailExcessThreshold(block.scores, block.classes, scoreClass,
                    calibration.scoresForClass(scoreClass), targetPrecision, config);
            if (!Double.isFinite(threshold)) {
                continue;
            }
            for (int i = 0; i < block.scores.length; i++) {
                if ((!conditional || block.classes[i] == scoreClass) && block.scores[i] >= threshold) {
                    predictions[i] = true;
                }
            }
        }
        return predictions;
    }

    private static double tailExcessThreshold(double[] scores, int[] classes, int scoreClass,
            double[] calibrationScores, double targetPrecision, Config config) {
        if (calibrationScores.length == 0) {
            return Double.POSITIVE_INFINITY;
        }
        double[] current = new double[scores.length];
        int count = 0;
        for (int i = 0; i < scores.length; i++) {
            if (classes[i] == scoreClass) {
                current[count++] = scores[i];
            }
        }
        if (count < config.tailMinSelected) {
            return Double.POSITIVE_INFINITY;
        }
        java.util.Arrays.sort(current, 0, count);
        double threshold = Double.POSITIVE_INFINITY;
        for (int i = count - 1; i >= 0; i--) {
            if (i < count - 1 && current[i] == current[i + 1]) {
                continue;
            }
            int selected = count - i;
            if (selected < config.tailMinSelected) {
                continue;
            }
            double tailProbability = empiricalUpperTailProbability(calibrationScores, current[i]);
            double expected = selected == 0 ? 0.0 : count * tailProbability;
            double excess = max(0.0, selected - expected);
            double precisionHat = excess / selected;
            double pValue = poissonSurvival(max(1.0e-12, expected), selected);
            if (precisionHat >= targetPrecision && pValue <= config.tailSignificance) {
                threshold = current[i];
            }
        }
        return threshold;
    }

    private static double empiricalUpperTailProbability(double[] sortedScores, double score) {
        if (sortedScores.length == 0) {
            return 1.0;
        }
        int index = lowerBound(sortedScores, score);
        return (sortedScores.length - index) / (double) sortedScores.length;
    }

    private static OracleResult oracleDiagnostic(String datasetName, String scoreName, Dataset dataset,
            TrainStats stats, ContextSet context, int startBucket, int endBucket, boolean combinedScore) {
        int count = dataset.countLines(startBucket, endBucket, context);
        long[] packed = new long[count];
        int index = 0;
        long positives = 0;
        LineScorer scorer = combinedScore ? null : scorerByName(scoreName);
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            if (!context.candidates[bucketIndex]) {
                continue;
            }
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                boolean label = dataset.labels.values[i] != 0;
                double score = combinedScore ? fdrLineScore(dataset, stats, context, i)
                        : scorer.score(dataset, stats, i);
                packed[index++] = pack(score, label);
                if (label) {
                    ++positives;
                }
            }
        }
        if (index == 0 || positives == 0) {
            return new OracleResult(datasetName, scoreName, context.name, context.anomalyRate, context.shingleSize,
                    index, positives, 0, 0, 0, 0, 0, 0, false);
        }
        java.util.Arrays.sort(packed, 0, index);
        long tp = 0;
        double bestF1 = 0;
        double bestF1Precision = 0;
        double bestF1Recall = 0;
        double bestPrecisionAtRecall50 = 0;
        double bestRecallAtPrecision50 = 0;
        double averagePrecisionNumerator = 0;
        boolean targetFeasible = false;
        for (int i = 0; i < index; i++) {
            if ((packed[i] & 1L) != 0) {
                ++tp;
                averagePrecisionNumerator += tp / (double) (i + 1L);
            }
            double precision = tp / (double) (i + 1L);
            double recall = tp / (double) positives;
            double f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
            if (f1 > bestF1) {
                bestF1 = f1;
                bestF1Precision = precision;
                bestF1Recall = recall;
            }
            if (recall >= 0.5) {
                bestPrecisionAtRecall50 = max(bestPrecisionAtRecall50, precision);
            }
            if (precision >= 0.5) {
                bestRecallAtPrecision50 = max(bestRecallAtPrecision50, recall);
            }
            if (precision >= 0.5 && recall >= 0.5) {
                targetFeasible = true;
            }
        }
        double averagePrecision = averagePrecisionNumerator / positives;
        return new OracleResult(datasetName, scoreName, context.name, context.anomalyRate, context.shingleSize, index,
                positives, bestPrecisionAtRecall50, bestRecallAtPrecision50, bestF1, bestF1Precision, bestF1Recall,
                averagePrecision, targetFeasible);
    }

    private static OracleResult oracleDiagnosticScoreMode(String datasetName, String scoreMode, Dataset dataset,
            TrainStats stats, ContextSet context, int startBucket, int endBucket) {
        if (scoreMode.startsWith("bgl_semantic_online")) {
            return OracleResult.oracleDiagnosticOnlineSemantic(datasetName, scoreMode, dataset, stats, context,
                    startBucket, endBucket);
        }
        int count = dataset.countLines(startBucket, endBucket, context);
        long[] packed = new long[count];
        int index = 0;
        long positives = 0;
        for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
            if (!context.candidates[bucketIndex]) {
                continue;
            }
            for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                boolean label = dataset.labels.values[i] != 0;
                double score = lineScore(dataset, stats, context, i, scoreMode);
                packed[index++] = pack(score, label);
                if (label) {
                    ++positives;
                }
            }
        }
        if (index == 0 || positives == 0) {
            return new OracleResult(datasetName, "score_mode_" + scoreMode, context.name, context.anomalyRate,
                    context.shingleSize, index, positives, 0, 0, 0, 0, 0, 0, false);
        }
        java.util.Arrays.sort(packed, 0, index);
        long tp = 0;
        double bestF1 = 0;
        double bestF1Precision = 0;
        double bestF1Recall = 0;
        double bestPrecisionAtRecall50 = 0;
        double bestRecallAtPrecision50 = 0;
        double averagePrecisionNumerator = 0;
        boolean targetFeasible = false;
        for (int i = 0; i < index; i++) {
            if ((packed[i] & 1L) != 0) {
                ++tp;
                averagePrecisionNumerator += tp / (double) (i + 1L);
            }
            double precision = tp / (double) (i + 1L);
            double recall = tp / (double) positives;
            double f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
            if (f1 > bestF1) {
                bestF1 = f1;
                bestF1Precision = precision;
                bestF1Recall = recall;
            }
            if (recall >= 0.5) {
                bestPrecisionAtRecall50 = max(bestPrecisionAtRecall50, precision);
            }
            if (precision >= 0.5) {
                bestRecallAtPrecision50 = max(bestRecallAtPrecision50, recall);
            }
            if (precision >= 0.5 && recall >= 0.5) {
                targetFeasible = true;
            }
        }
        double averagePrecision = averagePrecisionNumerator / positives;
        return new OracleResult(datasetName, "score_mode_" + scoreMode, context.name, context.anomalyRate,
                context.shingleSize, index, positives, bestPrecisionAtRecall50, bestRecallAtPrecision50, bestF1,
                bestF1Precision, bestF1Recall, averagePrecision, targetFeasible);
    }

    private static OracleResult oracleFromPacked(String datasetName, String scoreName, String contextName,
            double contextAnomalyRate, int contextShingle, long[] packed, int count, long positives) {
        if (count == 0 || positives == 0) {
            return new OracleResult(datasetName, scoreName, contextName, contextAnomalyRate, contextShingle, count,
                    positives, 0, 0, 0, 0, 0, 0, false);
        }
        java.util.Arrays.sort(packed, 0, count);
        long tp = 0;
        double bestF1 = 0;
        double bestF1Precision = 0;
        double bestF1Recall = 0;
        double bestPrecisionAtRecall50 = 0;
        double bestRecallAtPrecision50 = 0;
        double averagePrecisionNumerator = 0;
        boolean targetFeasible = false;
        for (int i = 0; i < count; i++) {
            if ((packed[i] & 1L) != 0) {
                ++tp;
                averagePrecisionNumerator += tp / (double) (i + 1L);
            }
            double precision = tp / (double) (i + 1L);
            double recall = tp / (double) positives;
            double f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
            if (f1 > bestF1) {
                bestF1 = f1;
                bestF1Precision = precision;
                bestF1Recall = recall;
            }
            if (recall >= 0.5) {
                bestPrecisionAtRecall50 = max(bestPrecisionAtRecall50, precision);
            }
            if (precision >= 0.5) {
                bestRecallAtPrecision50 = max(bestRecallAtPrecision50, recall);
            }
            if (precision >= 0.5 && recall >= 0.5) {
                targetFeasible = true;
            }
        }
        double averagePrecision = averagePrecisionNumerator / positives;
        return new OracleResult(datasetName, scoreName, contextName, contextAnomalyRate, contextShingle, count,
                positives, bestPrecisionAtRecall50, bestRecallAtPrecision50, bestF1, bestF1Precision, bestF1Recall,
                averagePrecision, targetFeasible);
    }

    private static LineScorer scorerByName(String name) {
        for (LineScorer scorer : LineScorer.suite()) {
            if (scorer.name().equals(name)) {
                return scorer;
            }
        }
        throw new IllegalArgumentException(name);
    }

    private static double fdrLineScore(Dataset dataset, TrainStats stats, ContextSet context, int lineIndex) {
        int bucketIndex = dataset.bucketIndexes.values[lineIndex];
        int eventId = dataset.eventIds.values[lineIndex];
        int entityId = dataset.entityIds.values[lineIndex];
        Bucket bucket = dataset.buckets.get(bucketIndex);
        double globalRarity = stats.globalRarity(eventId);
        double entityRarity = stats.entityRarity(entityId, eventId);
        double rarity = min(entityRarity, globalRarity + 2.0);
        double spike = Math.log1p(templateSpike(bucket, stats, eventId));
        double entitySpike = Math.log1p(entityTemplateSpike(bucket, stats, entityId, eventId));
        double parameter = stats.parameterRarity(entityId, eventId, dataset.parameterHashes.values[lineIndex]);
        double transition = min(12.0, stats.transitionSurprise(dataset.prevEventIds.values[lineIndex], eventId));
        int keyword = dataset.keywordScores.values[lineIndex];
        double keywordBoost = keyword >= 8 ? 12.0 : keyword >= 4 ? 3.0 : 0.0;
        if (keyword == 1 && (rarity >= 8.0 || spike >= 2.0 || entitySpike >= 2.0 || parameter >= 6.0
                || context.isPositive(bucketIndex))) {
            keywordBoost = 0.6;
        }
        double contextBoost = context.isPositive(bucketIndex) ? 1.5 : 0.0;
        return rarity + keywordBoost + 1.0 * spike + 1.2 * entitySpike + 0.35 * parameter + 0.25 * transition
                + contextBoost;
    }

    private static double lineScore(Dataset dataset, TrainStats stats, ContextSet context, int lineIndex,
            String scoreMode) {
        int eventId = dataset.eventIds.values[lineIndex];
        int entityId = dataset.entityIds.values[lineIndex];
        int keyword = dataset.keywordScores.values[lineIndex];
        switch (scoreMode) {
        case "rarity_keyword":
            return stats.entityRarity(entityId, eventId) + 3.0 * keyword;
        case "global_rarity_keyword":
            return stats.globalRarity(eventId) + 3.0 * keyword;
        case "global_rarity_keyword_tiebreak":
            return stats.globalRarity(eventId) + 3.0 * keyword
                    + 1.0e-3 * bglTieBreakScore(dataset, stats, context, lineIndex);
        case "rarity_keyword_tiebreak":
            return stats.entityRarity(entityId, eventId) + 3.0 * keyword
                    + 1.0e-3 * bglTieBreakScore(dataset, stats, context, lineIndex);
        case "global_rarity_keyword_gated":
            return stats.globalRarity(eventId) + gatedKeywordBoost(keyword, stats.globalRarity(eventId), 0.0, 0.0,
                    context.isPositive(dataset.bucketIndexes.values[lineIndex]));
        case "global_rarity_keyword_context":
            return stats.globalRarity(eventId) + 3.0 * keyword
                    + (context.isPositive(dataset.bucketIndexes.values[lineIndex]) ? 2.0 : 0.0);
        case "global_entity_param":
            return bglFeatureScore(dataset, stats, context, lineIndex, false, true, false);
        case "global_entity_spike":
            return bglFeatureScore(dataset, stats, context, lineIndex, false, false, true);
        case "bgl_rank":
            return bglFeatureScore(dataset, stats, context, lineIndex, false, true, true);
        case "bgl_rank_gated":
            return bglFeatureScore(dataset, stats, context, lineIndex, true, true, true);
        case "bgl_semantic":
            return bglSemanticScore(dataset, stats, context, lineIndex, true, true, false, null);
        case "bgl_semantic_no_suppress":
            return bglSemanticScore(dataset, stats, context, lineIndex, true, false, false, null);
        case "bgl_semantic_no_phrase":
            return bglSemanticScore(dataset, stats, context, lineIndex, false, true, false, null);
        case "bgl_semantic_online":
        case "bgl_semantic_online_no_suppress":
        case "bgl_semantic_online_no_phrase":
            return bglSemanticScore(dataset, stats, context, lineIndex, !scoreMode.endsWith("_no_phrase"),
                    !scoreMode.endsWith("_no_suppress"), false, null);
        case "keyword":
            return keyword;
        case "global_rarity":
            return stats.globalRarity(eventId);
        case "rarity":
            return stats.entityRarity(entityId, eventId);
        case "combined":
            return fdrLineScore(dataset, stats, context, lineIndex);
        default:
            throw new IllegalArgumentException("unknown score mode " + scoreMode);
        }
    }

    private static double bglSemanticScore(Dataset dataset, TrainStats stats, ContextSet context, int lineIndex,
            boolean includePhrase, boolean includeStableSuppression, boolean online, OnlineSemanticStats onlineStats) {
        int bucketIndex = dataset.bucketIndexes.values[lineIndex];
        int eventId = dataset.eventIds.values[lineIndex];
        int entityId = dataset.entityIds.values[lineIndex];
        int componentId = dataset.componentIds.values[lineIndex];
        int levelId = dataset.levelIds.values[lineIndex];
        int phraseHash = dataset.phraseHashes.values[lineIndex];
        int parameterHash = dataset.parameterHashes.values[lineIndex];
        Bucket bucket = dataset.buckets.get(bucketIndex);
        double templateRarity = stats.globalRarity(eventId);
        double signatureRarity = online ? onlineStats.semanticSignatureRarity(componentId, levelId, phraseHash)
                : stats.semanticSignatureRarity(componentId, levelId, phraseHash);
        double phraseRarity = includePhrase
                ? (online ? onlineStats.phraseRarity(phraseHash) : stats.phraseRarity(phraseHash))
                : 0.0;
        double parameter = stats.parameterRarity(entityId, eventId, parameterHash);
        double entitySpike = Math.log1p(entityTemplateSpike(bucket, stats, entityId, eventId));
        double stableSuppression = includeStableSuppression
                ? (online ? onlineStats.stableSignatureSuppression(componentId, levelId, phraseHash)
                        : stats.stableSignatureSuppression(componentId, levelId, phraseHash, 20, 5, 8.0))
                : 0.0;
        double noveltyGate = max(max(gate(signatureRarity, 3.0, 10.0), gate(phraseRarity, 3.0, 10.0)),
                max(gate(parameter, 3.0, 10.0), gate(entitySpike, 0.4, 3.0)));
        int keyword = dataset.keywordScores.values[lineIndex];
        double keywordBoost = keyword >= 8 ? 10.0 * noveltyGate
                : keyword >= 4 ? 4.0 * noveltyGate : keyword > 0 ? 0.5 * noveltyGate : 0.0;
        double contextBoost = context.isPositive(bucketIndex) ? 1.0 : 0.0;
        return 0.5 * templateRarity + 1.4 * signatureRarity + 1.2 * phraseRarity + 0.7 * parameter + 0.8 * entitySpike
                + keywordBoost + contextBoost - 1.8 * stableSuppression;
    }

    private static double onlineDecayedLineScore(Dataset dataset, OnlineBaselineState online, int lineIndex,
            Config config) {
        return onlineDecayedLineScore(dataset, online, lineIndex, config, config.scoreMode);
    }

    private static double onlineDecayedLineScore(Dataset dataset, OnlineBaselineState online, int lineIndex,
            Config config, String scoreMode) {
        int eventId = dataset.eventIds.values[lineIndex];
        int entityId = dataset.entityIds.values[lineIndex];
        int keyword = dataset.keywordScores.values[lineIndex];
        switch (scoreMode) {
        case "global_rarity_keyword":
            return online.slow.globalRarity(eventId) + 3.0 * keyword;
        case "global_rarity_keyword_tiebreak":
            return online.slow.globalRarity(eventId) + 3.0 * keyword
                    + config.onlineTiebreakWeight * onlineTieBreakScore(dataset, online, lineIndex);
        case "global_rarity_keyword_stable_tiebreak":
            return onlineStableKeywordScore(dataset, online, lineIndex, false, true, config);
        case "component_level_rarity_keyword":
            return online.slow.componentLevelRarity(dataset.componentIds.values[lineIndex],
                    dataset.levelIds.values[lineIndex], eventId) + 3.0 * keyword;
        case "component_level_rarity_keyword_tiebreak":
            return online.slow.componentLevelRarity(dataset.componentIds.values[lineIndex],
                    dataset.levelIds.values[lineIndex], eventId) + 3.0 * keyword
                    + config.onlineTiebreakWeight * onlineTieBreakScore(dataset, online, lineIndex);
        case "component_rarity_keyword":
            return online.slow.componentRarity(dataset.componentIds.values[lineIndex], eventId) + 3.0 * keyword;
        case "component_rarity_keyword_tiebreak":
            return online.slow.componentRarity(dataset.componentIds.values[lineIndex], eventId) + 3.0 * keyword
                    + config.onlineTiebreakWeight * onlineTieBreakScore(dataset, online, lineIndex);
        case "component_rarity_keyword_stable":
            return onlineStableKeywordScore(dataset, online, lineIndex, true, false, config);
        case "component_rarity_keyword_stable_tiebreak":
            return onlineStableKeywordScore(dataset, online, lineIndex, true, true, config);
        case "component_level_rarity_keyword_gated":
            return onlineComponentLevelGatedKeywordScore(dataset, online, lineIndex, false, config);
        case "component_level_rarity_keyword_gated_tiebreak":
            return onlineComponentLevelGatedKeywordScore(dataset, online, lineIndex, true, config);
        case "global_rarity_keyword_gated":
            return onlineGatedKeywordScore(dataset, online, lineIndex, false, config);
        case "global_rarity_keyword_gated_tiebreak":
            return onlineGatedKeywordScore(dataset, online, lineIndex, true, config);
        case "rarity_keyword":
            return online.slow.entityRarity(entityId, eventId) + 3.0 * keyword;
        case "rarity_keyword_tiebreak":
            return online.slow.entityRarity(entityId, eventId) + 3.0 * keyword
                    + config.onlineTiebreakWeight * onlineTieBreakScore(dataset, online, lineIndex);
        case "global_rarity":
            return online.slow.globalRarity(eventId);
        case "rarity":
            return online.slow.entityRarity(entityId, eventId);
        case "keyword":
            return keyword;
        default:
            return onlineCompositeLineScore(dataset, online, lineIndex);
        }
    }

    private static double onlineGatedKeywordScore(Dataset dataset, OnlineBaselineState online, int lineIndex,
            boolean includeTieBreak, Config config) {
        int eventId = dataset.eventIds.values[lineIndex];
        double rarity = online.slow.globalRarity(eventId);
        int keyword = dataset.keywordScores.values[lineIndex];
        double gate = min(1.0, rarity / max(1.0, online.keywordRarityGate));
        double score = rarity + 3.0 * keyword * gate;
        if (includeTieBreak) {
            score += config.onlineTiebreakWeight * onlineTieBreakScore(dataset, online, lineIndex);
        }
        return score;
    }

    private static double onlineStableKeywordScore(Dataset dataset, OnlineBaselineState online, int lineIndex,
            boolean componentRarity, boolean includeTieBreak, Config config) {
        int eventId = dataset.eventIds.values[lineIndex];
        int componentId = dataset.componentIds.values[lineIndex];
        int levelId = dataset.levelIds.values[lineIndex];
        int phraseHash = dataset.phraseHashes.values[lineIndex];
        double rarity = componentRarity ? online.slow.componentRarity(componentId, eventId)
                : online.slow.globalRarity(eventId);
        int keyword = dataset.keywordScores.values[lineIndex];
        double stablePenalty = online.stableSuppression(componentId, levelId, phraseHash, eventId);
        double score = rarity + 3.0 * keyword - 1.25 * stablePenalty;
        if (includeTieBreak && config.onlineTiebreakWeight > 0.0) {
            score += config.onlineTiebreakWeight * onlineTieBreakScore(dataset, online, lineIndex);
        }
        return max(0.0, score);
    }

    private static double onlineComponentLevelGatedKeywordScore(Dataset dataset, OnlineBaselineState online,
            int lineIndex, boolean includeTieBreak, Config config) {
        int eventId = dataset.eventIds.values[lineIndex];
        int componentId = dataset.componentIds.values[lineIndex];
        int levelId = dataset.levelIds.values[lineIndex];
        double rarity = online.slow.componentLevelRarity(componentId, levelId, eventId);
        int keyword = dataset.keywordScores.values[lineIndex];
        double gate = min(1.0, rarity / max(1.0, online.keywordRarityGate));
        double score = rarity + 3.0 * keyword * gate;
        if (includeTieBreak) {
            score += config.onlineTiebreakWeight * onlineTieBreakScore(dataset, online, lineIndex);
        }
        return score;
    }

    private static double onlineCompositeLineScore(Dataset dataset, OnlineBaselineState online, int lineIndex) {
        int eventId = dataset.eventIds.values[lineIndex];
        int entityId = dataset.entityIds.values[lineIndex];
        int componentId = dataset.componentIds.values[lineIndex];
        int levelId = dataset.levelIds.values[lineIndex];
        int phraseHash = dataset.phraseHashes.values[lineIndex];
        int parameterHash = dataset.parameterHashes.values[lineIndex];
        OnlineDecayedStats slow = online.slow;
        double globalRarity = slow.globalRarity(eventId);
        double entityRarity = slow.entityRarity(entityId, eventId);
        double componentRarity = slow.componentRarity(componentId, eventId);
        double componentLevelRarity = slow.componentLevelRarity(componentId, levelId, eventId);
        double rarity = min(entityRarity, globalRarity + 2.0);
        double signatureRarity = slow.semanticSignatureRarity(componentId, levelId, phraseHash);
        double phraseRarity = slow.phraseRarity(phraseHash);
        double parameter = slow.parameterRarity(entityId, eventId, parameterHash);
        double transition = min(12.0, slow.transitionSurprise(dataset.prevEventIds.values[lineIndex], eventId));
        double templateSpike = online.templateSpike(eventId);
        double entitySpike = online.entityTemplateSpike(entityId, eventId);
        double componentLevelSpike = online.componentLevelTemplateSpike(componentId, levelId, eventId);
        double spike = max(templateSpike, max(entitySpike, componentLevelSpike));
        double stableSuppression = online.stableSuppression(componentId, levelId, phraseHash, eventId);
        double noveltyGate = max(max(gate(rarity, 3.0, 10.0), gate(signatureRarity, 3.0, 10.0)),
                max(gate(parameter, 3.0, 10.0), gate(spike, 0.4, 3.0)));
        int keyword = dataset.keywordScores.values[lineIndex];
        double keywordBoost = keyword >= 8 ? 10.0 * noveltyGate
                : keyword >= 4 ? 4.0 * noveltyGate : keyword > 0 ? 0.6 * noveltyGate : 0.0;
        double entityExcess = max(0.0, rarity - globalRarity);
        double score = 0.55 * globalRarity + 0.6 * entityExcess + 0.35 * componentRarity + 0.35 * componentLevelRarity
                + 1.25 * signatureRarity + 0.9 * phraseRarity + 0.55 * parameter + 1.4 * spike + 0.2 * transition
                + keywordBoost - 1.5 * stableSuppression;
        return max(0.0, min(80.0, score));
    }

    private static double onlineTieBreakScore(Dataset dataset, OnlineBaselineState online, int lineIndex) {
        int eventId = dataset.eventIds.values[lineIndex];
        int entityId = dataset.entityIds.values[lineIndex];
        int componentId = dataset.componentIds.values[lineIndex];
        OnlineDecayedStats slow = online.slow;
        double entityRarity = slow.entityRarity(entityId, eventId);
        double componentRarity = slow.componentRarity(componentId, eventId);
        double entityGivenTemplate = slow.entityGivenTemplateRarity(entityId, eventId);
        double parameter = slow.parameterRarity(entityId, eventId, dataset.parameterHashes.values[lineIndex]);
        double entitySpike = online.entityTemplateSpike(entityId, eventId);
        double transition = min(12.0, slow.transitionSurprise(dataset.prevEventIds.values[lineIndex], eventId));
        return entityRarity + 0.5 * componentRarity + 0.5 * entityGivenTemplate + 0.5 * parameter + 2.0 * entitySpike
                + 0.25 * transition;
    }

    private static double gate(double value, double low, double high) {
        if (value <= low) {
            return 0.0;
        }
        if (value >= high) {
            return 1.0;
        }
        return (value - low) / (high - low);
    }

    private static double bglTieBreakScore(Dataset dataset, TrainStats stats, ContextSet context, int lineIndex) {
        int bucketIndex = dataset.bucketIndexes.values[lineIndex];
        int eventId = dataset.eventIds.values[lineIndex];
        int entityId = dataset.entityIds.values[lineIndex];
        Bucket bucket = dataset.buckets.get(bucketIndex);
        double entityRarity = stats.entityRarity(entityId, eventId);
        double componentRarity = stats.componentRarity(dataset.componentIds.values[lineIndex], eventId);
        double entityGivenTemplate = stats.entityGivenTemplateRarity(entityId, eventId);
        double parameter = stats.parameterRarity(entityId, eventId, dataset.parameterHashes.values[lineIndex]);
        double entitySpike = Math.log1p(entityTemplateSpike(bucket, stats, entityId, eventId));
        double transition = min(12.0, stats.transitionSurprise(dataset.prevEventIds.values[lineIndex], eventId));
        double contextBoost = context.isPositive(bucketIndex) ? 2.0 : 0.0;
        return entityRarity + 0.5 * componentRarity + 0.5 * entityGivenTemplate + 0.5 * parameter + 2.0 * entitySpike
                + 0.25 * transition + contextBoost;
    }

    private static double gatedKeywordBoost(int keyword, double rarity, double parameter, double spike,
            boolean contextPositive) {
        if (keyword >= 8) {
            return 18.0;
        }
        if (keyword >= 4) {
            return 8.0;
        }
        if (keyword > 0 && (rarity >= 7.0 || parameter >= 5.0 || spike >= 1.5 || contextPositive)) {
            return 1.0;
        }
        return 0.0;
    }

    private static double bglFeatureScore(Dataset dataset, TrainStats stats, ContextSet context, int lineIndex,
            boolean gateWeakKeywords, boolean includeParameter, boolean includeEntitySpike) {
        int bucketIndex = dataset.bucketIndexes.values[lineIndex];
        int eventId = dataset.eventIds.values[lineIndex];
        int entityId = dataset.entityIds.values[lineIndex];
        int componentId = dataset.componentIds.values[lineIndex];
        Bucket bucket = dataset.buckets.get(bucketIndex);
        double globalRarity = stats.globalRarity(eventId);
        double entityRarity = stats.entityRarity(entityId, eventId);
        double componentRarity = stats.componentRarity(componentId, eventId);
        double entityGivenTemplate = stats.entityGivenTemplateRarity(entityId, eventId);
        double parameter = includeParameter
                ? stats.parameterRarity(entityId, eventId, dataset.parameterHashes.values[lineIndex])
                : 0.0;
        double entitySpike = includeEntitySpike ? Math.log1p(entityTemplateSpike(bucket, stats, entityId, eventId))
                : 0.0;
        double globalSpike = Math.log1p(templateSpike(bucket, stats, eventId));
        double transition = min(12.0, stats.transitionSurprise(dataset.prevEventIds.values[lineIndex], eventId));
        double entityExcess = max(0.0, min(entityRarity, componentRarity) - globalRarity);
        double evidence = max(max(globalRarity, entityRarity), max(parameter, max(entitySpike, globalSpike)));
        int keyword = dataset.keywordScores.values[lineIndex];
        double keywordBoost;
        if (keyword >= 8) {
            keywordBoost = 9.0;
        } else if (keyword >= 4) {
            keywordBoost = gateWeakKeywords ? 4.0 : 8.0;
        } else if (keyword > 0) {
            keywordBoost = !gateWeakKeywords || evidence >= 7.0 || entityGivenTemplate >= 4.0
                    || context.isPositive(bucketIndex) ? 0.8 : 0.0;
        } else {
            keywordBoost = 0.0;
        }
        double contextBoost = context.isPositive(bucketIndex) ? 1.0 : 0.0;
        return globalRarity + 0.55 * entityExcess + 0.35 * min(12.0, entityGivenTemplate) + 0.55 * parameter
                + 1.7 * entitySpike + 0.8 * globalSpike + 0.15 * transition + keywordBoost + contextBoost;
    }

    private static double fdrCellScore(Bucket bucket, TrainStats stats, ContextSet context, int bucketIndex,
            int eventId) {
        double rarity = stats.globalRarity(eventId);
        double spike = Math.log1p(templateSpike(bucket, stats, eventId));
        int keyword = bucket.maxKeywordByEvent.getOrDefault(eventId, (byte) 0);
        double keywordBoost = keyword >= 8 ? 10.0
                : keyword >= 4 ? 3.0 : keyword == 1 && (rarity >= 8.0 || spike >= 2.0) ? 0.6 : 0.0;
        double contextBoost = context.isPositive(bucketIndex) ? 1.5 : 0.0;
        return rarity + keywordBoost + 2.0 * spike + contextBoost;
    }

    private static double templateSpike(Bucket bucket, TrainStats stats, int eventId) {
        double expected = bucket.totalCount * stats.globalRate(eventId);
        double observed = bucket.eventCounts.getOrDefault(eventId, 0);
        return max(0.0, (observed - expected) / Math.sqrt(expected + 1.0));
    }

    private static double entityTemplateSpike(Bucket bucket, TrainStats stats, int entityId, int eventId) {
        if (bucket.entityCounts == null || bucket.entityEventCounts == null) {
            return 0.0;
        }
        int entityBucketTotal = bucket.entityCounts.getOrDefault(entityId, 0);
        if (entityBucketTotal <= 0) {
            return 0.0;
        }
        double entityTotal = stats.entityTotals.getOrDefault(entityId, 0);
        double probability = entityTotal <= 0.0 ? stats.globalRate(eventId)
                : stats.entityEventCount(entityId, eventId) / entityTotal;
        double expected = entityBucketTotal * probability;
        int observed = bucket.entityEventCounts.getOrDefault(entityEventKey(entityId, eventId), 0);
        return max(0.0, (observed - expected) / Math.sqrt(expected + 1.0));
    }

    private static double empiricalUpperTailPValue(double[] sortedScores, double score) {
        if (sortedScores.length == 0) {
            return 1.0;
        }
        int index = lowerBound(sortedScores, score);
        return (sortedScores.length - index + 1.0) / (sortedScores.length + 1.0);
    }

    private static double quantile(double[] sortedScores, double quantile) {
        if (sortedScores.length == 0) {
            return Double.POSITIVE_INFINITY;
        }
        double clean = max(0.0, min(1.0, quantile));
        int index = (int) Math.ceil(clean * sortedScores.length) - 1;
        return sortedScores[min(max(index, 0), sortedScores.length - 1)];
    }

    private static double listQuantile(List<Double> scores, double quantile) {
        if (scores == null || scores.isEmpty()) {
            return Double.POSITIVE_INFINITY;
        }
        Collections.sort(scores);
        double clean = max(0.0, min(1.0, quantile));
        int index = (int) Math.ceil(clean * scores.size()) - 1;
        return scores.get(min(max(index, 0), scores.size() - 1));
    }

    private static int lowerBound(double[] values, double target) {
        int low = 0;
        int high = values.length;
        while (low < high) {
            int middle = (low + high) >>> 1;
            if (values[middle] < target) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        return low;
    }

    private static int lowerBound(List<Double> values, double target) {
        int low = 0;
        int high = values.size();
        while (low < high) {
            int middle = (low + high) >>> 1;
            if (values.get(middle) < target) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        return low;
    }

    private static int upperBound(List<Double> values, double target) {
        int low = 0;
        int high = values.size();
        while (low < high) {
            int middle = (low + high) >>> 1;
            if (values.get(middle) <= target) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        return low;
    }

    private static double bhCutoff(double[] pValues, double q) {
        double[] sorted = java.util.Arrays.copyOf(pValues, pValues.length);
        java.util.Arrays.sort(sorted);
        double cutoff = -1.0;
        for (int i = 0; i < sorted.length; i++) {
            if (sorted[i] <= q * (i + 1.0) / sorted.length) {
                cutoff = sorted[i];
            }
        }
        return cutoff;
    }

    private static long anomalousLines(Dataset dataset, int startBucket, int endBucket) {
        long count = 0;
        for (int i = 0; i < dataset.labels.size; i++) {
            int bucketIndex = dataset.bucketIndexes.values[i];
            if (bucketIndex >= startBucket && bucketIndex < endBucket && dataset.labels.values[i] != 0) {
                ++count;
            }
        }
        return count;
    }

    private static long coveredAnomalousLines(Dataset dataset, ContextSet context, int startBucket, int endBucket) {
        long count = 0;
        for (int i = 0; i < dataset.labels.size; i++) {
            int bucketIndex = dataset.bucketIndexes.values[i];
            if (bucketIndex >= startBucket && bucketIndex < endBucket && context.candidates[bucketIndex]
                    && dataset.labels.values[i] != 0) {
                ++count;
            }
        }
        return count;
    }

    private static double contextLineRecall(Dataset dataset, ContextSet context, int startBucket, int endBucket) {
        if (context.isUngated()) {
            return 1.0;
        }
        long anomalies = anomalousLines(dataset, startBucket, endBucket);
        return anomalies == 0 ? 0.0
                : coveredAnomalousLines(dataset, context, startBucket, endBucket) / (double) anomalies;
    }

    private static long pack(double score, boolean label) {
        double clean = Double.isFinite(score) ? max(0.0, score) : Double.MAX_VALUE;
        long bits = Double.doubleToRawLongBits(clean);
        long key = Long.MAX_VALUE - bits;
        return (key & ~1L) | (label ? 1L : 0L);
    }

    private static double unpackScore(long packed) {
        long key = packed & ~1L;
        long bits = Long.MAX_VALUE - key;
        return Double.longBitsToDouble(bits);
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

    private static void writeFdrResults(String output, List<FdrResult> results) throws IOException {
        Path path = Paths.get(output);
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write(FdrResult.header());
            writer.newLine();
            for (FdrResult result : results) {
                writer.write(result.toCsv());
                writer.newLine();
            }
        }
    }

    private static void writeOracleResults(String output, List<OracleResult> results) throws IOException {
        Path path = Paths.get(output);
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write(OracleResult.header());
            writer.newLine();
            for (OracleResult result : results) {
                writer.write(result.toCsv());
                writer.newLine();
            }
        }
    }

    private static void writeTemplateDiagnostics(String output, List<TemplateDiagnostic> results) throws IOException {
        Path path = Paths.get(output);
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write(TemplateDiagnostic.header());
            writer.newLine();
            for (TemplateDiagnostic result : results) {
                writer.write(result.toCsv());
                writer.newLine();
            }
        }
    }

    private static void printSummary(List<Result> results, int top) {
        List<Result> sorted = new ArrayList<>(results);
        sorted.sort((a, b) -> Double.compare(b.metrics.f1(), a.metrics.f1()));
        System.out.println(Result.header());
        for (int i = 0; i < min(top, sorted.size()); i++) {
            System.out.println(sorted.get(i).toCsv());
        }
    }

    private static void printFdrSummary(List<FdrResult> results, int top) {
        List<FdrResult> sorted = new ArrayList<>(results);
        sorted.sort((a, b) -> Double.compare(b.metrics.f1(), a.metrics.f1()));
        System.out.println(FdrResult.header());
        for (int i = 0; i < min(top, sorted.size()); i++) {
            System.out.println(sorted.get(i).toCsv());
        }
    }

    private static void printOracleSummary(List<OracleResult> results, int top) {
        List<OracleResult> sorted = new ArrayList<>(results);
        sorted.sort((a, b) -> Double.compare(b.bestF1, a.bestF1));
        System.out.println(OracleResult.header());
        for (int i = 0; i < min(top, sorted.size()); i++) {
            System.out.println(sorted.get(i).toCsv());
        }
    }

    private static void printTemplateDiagnostics(List<TemplateDiagnostic> results, int top) {
        System.out.println(TemplateDiagnostic.header());
        for (int i = 0; i < min(top, results.size()); i++) {
            System.out.println(results.get(i).toCsv());
        }
    }

    private interface LineScorer {
        String name();

        double score(Dataset dataset, TrainStats stats, int lineIndex);

        static List<LineScorer> suite() {
            List<LineScorer> scorers = new ArrayList<>();
            scorers.add(new BaseScorer("rarity") {
                @Override
                public double score(Dataset dataset, TrainStats stats, int lineIndex) {
                    return stats.entityRarity(dataset.entityIds.values[lineIndex], dataset.eventIds.values[lineIndex]);
                }
            });
            scorers.add(new BaseScorer("global_rarity") {
                @Override
                public double score(Dataset dataset, TrainStats stats, int lineIndex) {
                    return stats.globalRarity(dataset.eventIds.values[lineIndex]);
                }
            });
            scorers.add(new BaseScorer("oov") {
                @Override
                public double score(Dataset dataset, TrainStats stats, int lineIndex) {
                    return stats.entityEventCount(dataset.entityIds.values[lineIndex],
                            dataset.eventIds.values[lineIndex]) == 0 ? 1.0 : 0.0;
                }
            });
            scorers.add(new BaseScorer("spike") {
                @Override
                public double score(Dataset dataset, TrainStats stats, int lineIndex) {
                    int bucketIndex = dataset.bucketIndexes.values[lineIndex];
                    int eventId = dataset.eventIds.values[lineIndex];
                    Bucket bucket = dataset.buckets.get(bucketIndex);
                    double expected = bucket.totalCount * stats.globalRate(eventId);
                    double observed = bucket.eventCounts.getOrDefault(eventId, 0);
                    return max(0.0, (observed - expected) / Math.sqrt(expected + 1.0));
                }
            });
            scorers.add(new BaseScorer("transition") {
                @Override
                public double score(Dataset dataset, TrainStats stats, int lineIndex) {
                    return stats.transitionSurprise(dataset.prevEventIds.values[lineIndex],
                            dataset.eventIds.values[lineIndex]);
                }
            });
            scorers.add(new BaseScorer("keyword") {
                @Override
                public double score(Dataset dataset, TrainStats stats, int lineIndex) {
                    return dataset.keywordScores.values[lineIndex];
                }
            });
            scorers.add(new BaseScorer("rarity_keyword") {
                @Override
                public double score(Dataset dataset, TrainStats stats, int lineIndex) {
                    return stats.entityRarity(dataset.entityIds.values[lineIndex], dataset.eventIds.values[lineIndex])
                            + 3.0 * dataset.keywordScores.values[lineIndex];
                }
            });
            scorers.add(new BaseScorer("global_rarity_keyword") {
                @Override
                public double score(Dataset dataset, TrainStats stats, int lineIndex) {
                    return stats.globalRarity(dataset.eventIds.values[lineIndex])
                            + 3.0 * dataset.keywordScores.values[lineIndex];
                }
            });
            scorers.add(new BaseScorer("rarity_spike") {
                @Override
                public double score(Dataset dataset, TrainStats stats, int lineIndex) {
                    int bucketIndex = dataset.bucketIndexes.values[lineIndex];
                    int eventId = dataset.eventIds.values[lineIndex];
                    Bucket bucket = dataset.buckets.get(bucketIndex);
                    double expected = bucket.totalCount * stats.globalRate(eventId);
                    double observed = bucket.eventCounts.getOrDefault(eventId, 0);
                    double spike = max(0.0, (observed - expected) / Math.sqrt(expected + 1.0));
                    return stats.entityRarity(dataset.entityIds.values[lineIndex], eventId) + spike;
                }
            });
            scorers.add(new BaseScorer("ensemble") {
                @Override
                public double score(Dataset dataset, TrainStats stats, int lineIndex) {
                    double rarity = stats.entityRarity(dataset.entityIds.values[lineIndex],
                            dataset.eventIds.values[lineIndex]);
                    double transition = stats.transitionSurprise(dataset.prevEventIds.values[lineIndex],
                            dataset.eventIds.values[lineIndex]);
                    int bucketIndex = dataset.bucketIndexes.values[lineIndex];
                    int eventId = dataset.eventIds.values[lineIndex];
                    Bucket bucket = dataset.buckets.get(bucketIndex);
                    double expected = bucket.totalCount * stats.globalRate(eventId);
                    double spike = max(0.0,
                            (bucket.eventCounts.getOrDefault(eventId, 0) - expected) / Math.sqrt(expected + 1.0));
                    double oov = stats.entityEventCount(dataset.entityIds.values[lineIndex], eventId) == 0 ? 10.0 : 0.0;
                    double keyword = 3.0 * dataset.keywordScores.values[lineIndex];
                    return max(max(rarity, transition), max(max(spike, oov), keyword));
                }
            });
            return scorers;
        }
    }

    private abstract static class BaseScorer implements LineScorer {
        private final String name;

        private BaseScorer(String name) {
            this.name = name;
        }

        @Override
        public String name() {
            return name;
        }
    }

    private static final class ContextSet {
        private final String name;
        private final boolean[] candidates;
        private final double anomalyRate;
        private final int shingleSize;

        private ContextSet(String name, boolean[] candidates, double anomalyRate, int shingleSize) {
            this.name = name;
            this.candidates = candidates;
            this.anomalyRate = anomalyRate;
            this.shingleSize = shingleSize;
        }

        private static ContextSet all(String name, int size) {
            boolean[] candidates = new boolean[size];
            java.util.Arrays.fill(candidates, true);
            return new ContextSet(name, candidates, 1.0, 1);
        }

        private boolean isUngated() {
            return anomalyRate >= 1.0;
        }

        private boolean isPositive(int bucketIndex) {
            return !isUngated() && candidates[bucketIndex];
        }

        private static ContextSet trcf(Dataset dataset, TrainStats stats, int trainEnd, double anomalyRate,
                int shingleSize, Config config) {
            double[][] vectors = new double[dataset.buckets.size()][];
            for (int i = 0; i < dataset.buckets.size(); i++) {
                vectors[i] = vectorize(dataset.buckets.get(i), stats, config.topK);
            }
            ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder()
                    .dimensions(vectors[0].length * shingleSize).shingleSize(shingleSize)
                    .numberOfTrees(config.numberOfTrees).sampleSize(config.sampleSize).outputAfter(config.outputAfter)
                    .randomSeed(config.seed).anomalyRate(anomalyRate).autoAdjust(true).zFactor(config.zFactor)
                    .transformMethod(TransformMethod.NONE).scoringStrategy(ScoringStrategy.EXPECTED_INVERSE_DEPTH)
                    .build();
            boolean[] candidates = new boolean[dataset.buckets.size()];
            for (int i = 0; i < vectors.length; i++) {
                AnomalyDescriptor result = forest.process(vectors[i], i);
                if (i >= trainEnd && result.getAnomalyGrade() > 0.0) {
                    candidates[i] = true;
                }
            }
            return new ContextSet(
                    String.format(Locale.ROOT, "trcf_global_rate%.3f_shingle%d", anomalyRate, shingleSize), candidates,
                    anomalyRate, shingleSize);
        }
    }

    private static double[] vectorize(Bucket bucket, TrainStats stats, int topK) {
        double[] evidence = evidenceVector(bucket, stats);
        double[] vector = new double[topK + 1 + evidence.length];
        int other = 0;
        for (Map.Entry<Integer, Integer> entry : bucket.eventCounts.entrySet()) {
            Integer index = stats.topEventIndex.get(entry.getKey());
            if (index != null && index < topK) {
                vector[index] = Math.log1p(entry.getValue());
            } else {
                other += entry.getValue();
            }
        }
        vector[topK] = Math.log1p(other);
        System.arraycopy(evidence, 0, vector, topK + 1, evidence.length);
        return vector;
    }

    private static double[] evidenceVector(Bucket bucket, TrainStats stats) {
        int total = bucket.totalCount;
        int rare = 0;
        int oov = 0;
        int dominant = 0;
        double maxRarity = 0.0;
        double sumRarity = 0.0;
        double clippedRarity = 0.0;
        double entropy = 0.0;
        double topShift = 0.0;
        List<RarityCount> rarityCounts = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : bucket.eventCounts.entrySet()) {
            int eventId = entry.getKey();
            int count = entry.getValue();
            dominant = max(dominant, count);
            if (!stats.globalCounts.containsKey(eventId)) {
                oov += count;
            }
            if (stats.rareEvents.contains(eventId)) {
                rare += count;
            }
            double rarity = stats.globalRarity(eventId);
            rarityCounts.add(new RarityCount(rarity, count));
            maxRarity = max(maxRarity, rarity);
            sumRarity += rarity * count;
            clippedRarity += min(20.0, rarity) * count;
            double p = total == 0 ? 0.0 : count / (double) total;
            if (p > 0.0) {
                entropy -= p * Math.log(p);
            }
        }
        for (Map.Entry<Integer, Integer> entry : stats.topEventIndex.entrySet()) {
            if (entry.getValue() < 50) {
                int eventId = entry.getKey();
                double bucketRate = total == 0 ? 0.0 : bucket.eventCounts.getOrDefault(eventId, 0) / (double) total;
                topShift += Math.abs(bucketRate - stats.globalRate(eventId));
            }
        }
        return new double[] { Math.log1p(total), Math.log1p(bucket.eventCounts.size()), Math.log1p(rare),
                Math.log1p(oov), total == 0 ? 0.0 : rare / (double) total, total == 0 ? 0.0 : oov / (double) total,
                maxRarity, total == 0 ? 0.0 : sumRarity / total, weightedQuantile(rarityCounts, total, 0.95),
                weightedQuantile(rarityCounts, total, 0.99), Math.log1p(clippedRarity), entropy,
                total == 0 ? 0.0 : dominant / (double) total, topShift };
    }

    private static double weightedQuantile(List<RarityCount> values, int total, double quantile) {
        if (total <= 0 || values.isEmpty()) {
            return 0.0;
        }
        values.sort(Comparator.comparingDouble(value -> value.rarity));
        int target = max(1, (int) Math.ceil(total * quantile));
        int seen = 0;
        for (RarityCount value : values) {
            seen += value.count;
            if (seen >= target) {
                return value.rarity;
            }
        }
        return values.get(values.size() - 1).rarity;
    }

    private static final class Dataset {
        private final List<Bucket> buckets = new ArrayList<>();
        private final IntList bucketIndexes = new IntList();
        private final IntList eventIds = new IntList();
        private final IntList entityIds = new IntList();
        private final IntList componentIds = new IntList();
        private final IntList levelIds = new IntList();
        private final IntList prevEventIds = new IntList();
        private final IntList parameterHashes = new IntList();
        private final IntList phraseHashes = new IntList();
        private final ByteList keywordScores = new ByteList();
        private final ByteList labels = new ByteList();
        private final List<String> lineMessages = new ArrayList<>();
        private final List<String> entityNames = new ArrayList<>();
        private final List<String> componentNames = new ArrayList<>();
        private final List<String> levelNames = new ArrayList<>();
        private String[] templateTexts = new String[0];
        private int[] lineStarts;
        private int templateCount;

        private static Dataset load(String datasetName, String input, int sampleModulo, Config config)
                throws IOException {
            System.out.printf(Locale.ROOT, "loading dataset=%s input=%s bucket_seconds=%d sample_modulo=%d%n",
                    datasetName, input, config.bucketSeconds, sampleModulo);
            Dataset dataset = new Dataset();
            TemplateParser parser = newTemplateParser(config);
            Map<String, Integer> entityIds = new HashMap<>();
            Map<String, Integer> componentIds = new HashMap<>();
            Map<String, Integer> levelIds = new HashMap<>();
            Map<Integer, Integer> lastEventByEntity = new HashMap<>();
            Bucket current = null;
            long originalLine = 0;
            try (BufferedReader reader = Files.newBufferedReader(Paths.get(input), RAW_LOG_CHARSET)) {
                String line;
                while ((line = reader.readLine()) != null) {
                    ++originalLine;
                    if (sampleModulo > 1 && positiveHash(Long.toString(originalLine)) % sampleModulo != 0) {
                        continue;
                    }
                    ParsedLine parsed = parseBglLikeLine(line);
                    if (parsed == null || parsed.message.isEmpty()) {
                        continue;
                    }
                    long bucketKey = Math.floorDiv(parsed.epochSeconds, config.bucketSeconds);
                    if (current == null || current.bucketKey != bucketKey) {
                        current = new Bucket(bucketKey);
                        dataset.buckets.add(current);
                    }
                    int bucketIndex = dataset.buckets.size() - 1;
                    int entityId = config.memoryLimits.bounded
                            ? internBoundedId(parsed.entity, entityIds, dataset.entityNames,
                                    config.memoryLimits.entityEnabled() ? config.memoryLimits.maxEntities : 0)
                            : internId(parsed.entity, entityIds, dataset.entityNames);
                    int componentId = config.memoryLimits.bounded
                            ? internBoundedId(parsed.component, componentIds, dataset.componentNames,
                                    config.memoryLimits.maxGroups)
                            : internId(parsed.component, componentIds, dataset.componentNames);
                    int levelId = config.memoryLimits.bounded
                            ? internBoundedId(parsed.level, levelIds, dataset.levelNames, config.memoryLimits.maxGroups)
                            : internId(parsed.level, levelIds, dataset.levelNames);
                    int eventId = parser.parse(normalize(parsed.message));
                    int parameterHash = parameterHash(parsed.message);
                    int phraseHash = semanticPhraseHash(parsed.message);
                    int prevEventId = config.memoryLimits.entityEnabled() ? lastEventByEntity.getOrDefault(entityId, -1)
                            : -1;
                    if (config.memoryLimits.entityEnabled()) {
                        lastEventByEntity.put(entityId, eventId);
                    }
                    current.add(eventId, entityId, parsed.label, parsed.keywordScore, config.useEntityBucketCounts());
                    dataset.bucketIndexes.add(bucketIndex);
                    dataset.eventIds.add(eventId);
                    dataset.entityIds.add(entityId);
                    dataset.componentIds.add(componentId);
                    dataset.levelIds.add(levelId);
                    dataset.prevEventIds.add(prevEventId);
                    dataset.parameterHashes.add(parameterHash);
                    dataset.phraseHashes.add(phraseHash);
                    dataset.keywordScores.add(parsed.keywordScore);
                    dataset.labels.add((byte) (parsed.label ? 1 : 0));
                    if (config.storeLineMessages()) {
                        dataset.lineMessages.add(parsed.message);
                    }
                    if (dataset.labels.size % config.progressInterval == 0) {
                        System.out.printf(Locale.ROOT, "  %s sampled_lines=%d original=%d buckets=%d templates=%d%n",
                                datasetName, dataset.labels.size, originalLine, dataset.buckets.size(),
                                parser.templateCount());
                    }
                }
            }
            dataset.templateCount = parser.templateCount();
            dataset.templateTexts = parser.templates();
            dataset.buildLineStarts();
            System.out.printf(Locale.ROOT,
                    "loaded dataset=%s lines=%d buckets=%d templates=%d anomalous_lines=%d entities=%d components=%d levels=%d%n",
                    datasetName, dataset.labels.size, dataset.buckets.size(), dataset.templateCount,
                    dataset.labels.sum(), entityIds.size(), componentIds.size(), levelIds.size());
            return dataset;
        }

        private static int internId(String value, Map<String, Integer> ids, List<String> names) {
            Integer existing = ids.get(value);
            if (existing != null) {
                return existing;
            }
            int id = ids.size();
            ids.put(value, id);
            names.add(value);
            return id;
        }

        private static int internBoundedId(String value, Map<String, Integer> ids, List<String> names, int cap) {
            if (cap <= 0) {
                return 0;
            }
            if (names.isEmpty()) {
                names.add("<unknown>");
            }
            Integer existing = ids.get(value);
            if (existing != null) {
                return existing;
            }
            if (ids.size() >= cap) {
                return 0;
            }
            int id = ids.size() + 1;
            ids.put(value, id);
            names.add(value);
            return id;
        }

        private void buildLineStarts() {
            lineStarts = new int[buckets.size() + 1];
            int line = 0;
            for (int bucketIndex = 0; bucketIndex < buckets.size(); bucketIndex++) {
                lineStarts[bucketIndex] = line;
                while (line < labels.size && bucketIndexes.values[line] == bucketIndex) {
                    ++line;
                }
            }
            lineStarts[buckets.size()] = labels.size;
        }

        private int countLines(int startBucket, int endBucket) {
            if (lineStarts == null || startBucket < 0 || endBucket > buckets.size() || endBucket < startBucket) {
                return 0;
            }
            return lineStarts[endBucket] - lineStarts[startBucket];
        }

        private int countLines(int startBucket, int endBucket, ContextSet context) {
            int count = 0;
            for (int i = 0; i < labels.size; i++) {
                int bucketIndex = bucketIndexes.values[i];
                if (bucketIndex >= startBucket && bucketIndex < endBucket && context.candidates[bucketIndex]) {
                    ++count;
                }
            }
            return count;
        }
    }

    private static final class TrainStats {
        private final Map<Integer, Integer> globalCounts;
        private final Map<Long, Integer> entityEventCounts;
        private final Map<Integer, Integer> entityTotals;
        private final Map<Long, Integer> componentEventCounts;
        private final Map<Integer, Integer> componentTotals;
        private final Map<Long, Integer> templateParameterCounts;
        private final Map<Long, Integer> entityParameterCounts;
        private final Map<Integer, Integer> parameterCounts;
        private final Map<Long, Integer> semanticSignatureCounts;
        private final Map<Long, Integer> semanticSignatureBucketCounts;
        private final Map<Long, Integer> componentLevelTotals;
        private final Map<Integer, Integer> phraseCounts;
        private final Map<Long, Integer> transitionCounts;
        private final Map<Integer, Integer> prevTotals;
        private final Set<Integer> rareEvents;
        private final Map<Integer, Integer> topEventIndex;
        private final int globalTotal;
        private final int vocabularySize;
        private final int entityCount;
        private final int componentCount;
        private final int parameterVocabularySize;
        private final int semanticSignatureVocabularySize;
        private final int phraseVocabularySize;

        private TrainStats(Map<Integer, Integer> globalCounts, Map<Long, Integer> entityEventCounts,
                Map<Integer, Integer> entityTotals, Map<Long, Integer> componentEventCounts,
                Map<Integer, Integer> componentTotals, Map<Long, Integer> templateParameterCounts,
                Map<Long, Integer> entityParameterCounts, Map<Integer, Integer> parameterCounts,
                Map<Long, Integer> semanticSignatureCounts, Map<Long, Integer> semanticSignatureBucketCounts,
                Map<Long, Integer> componentLevelTotals, Map<Integer, Integer> phraseCounts,
                Map<Long, Integer> transitionCounts, Map<Integer, Integer> prevTotals, Set<Integer> rareEvents,
                Map<Integer, Integer> topEventIndex, int globalTotal, int vocabularySize, int entityCount,
                int componentCount, int parameterVocabularySize, int semanticSignatureVocabularySize,
                int phraseVocabularySize) {
            this.globalCounts = globalCounts;
            this.entityEventCounts = entityEventCounts;
            this.entityTotals = entityTotals;
            this.componentEventCounts = componentEventCounts;
            this.componentTotals = componentTotals;
            this.templateParameterCounts = templateParameterCounts;
            this.entityParameterCounts = entityParameterCounts;
            this.parameterCounts = parameterCounts;
            this.semanticSignatureCounts = semanticSignatureCounts;
            this.semanticSignatureBucketCounts = semanticSignatureBucketCounts;
            this.componentLevelTotals = componentLevelTotals;
            this.phraseCounts = phraseCounts;
            this.transitionCounts = transitionCounts;
            this.prevTotals = prevTotals;
            this.rareEvents = rareEvents;
            this.topEventIndex = topEventIndex;
            this.globalTotal = globalTotal;
            this.vocabularySize = vocabularySize;
            this.entityCount = entityCount;
            this.componentCount = componentCount;
            this.parameterVocabularySize = parameterVocabularySize;
            this.semanticSignatureVocabularySize = semanticSignatureVocabularySize;
            this.phraseVocabularySize = phraseVocabularySize;
        }

        private static TrainStats from(Dataset dataset, int trainEnd, Config config) {
            return fromRange(dataset, 0, trainEnd, config);
        }

        private static TrainStats fromRange(Dataset dataset, int startBucket, int endBucket, Config config) {
            Map<Integer, Integer> globalCounts = new HashMap<>();
            Map<Long, Integer> entityEventCounts = new HashMap<>();
            Map<Integer, Integer> entityTotals = new HashMap<>();
            Map<Long, Integer> componentEventCounts = new HashMap<>();
            Map<Integer, Integer> componentTotals = new HashMap<>();
            Map<Long, Integer> templateParameterCounts = config.useParameterFeatures() ? new HashMap<>()
                    : Collections.emptyMap();
            Map<Long, Integer> entityParameterCounts = config.useParameterFeatures() ? new HashMap<>()
                    : Collections.emptyMap();
            Map<Integer, Integer> parameterCounts = config.useParameterFeatures() ? new HashMap<>()
                    : Collections.emptyMap();
            Map<Long, Integer> semanticSignatureCounts = config.useSemanticFeatures() ? new HashMap<>()
                    : Collections.emptyMap();
            Map<Long, Integer> semanticSignatureBucketCounts = config.useSemanticFeatures() ? new HashMap<>()
                    : Collections.emptyMap();
            Map<Long, Integer> lastSemanticSignatureBucket = config.useSemanticFeatures() ? new HashMap<>()
                    : Collections.emptyMap();
            Map<Long, Integer> componentLevelTotals = config.useSemanticFeatures() ? new HashMap<>()
                    : Collections.emptyMap();
            Map<Integer, Integer> phraseCounts = config.useSemanticFeatures() ? new HashMap<>()
                    : Collections.emptyMap();
            Map<Long, Integer> transitionCounts = new HashMap<>();
            Map<Integer, Integer> prevTotals = new HashMap<>();
            int total = 0;
            int maxEntityId = -1;
            int maxComponentId = -1;
            for (int i = 0; i < dataset.labels.size; i++) {
                int bucketIndex = dataset.bucketIndexes.values[i];
                if (bucketIndex < startBucket || bucketIndex >= endBucket) {
                    continue;
                }
                int eventId = dataset.eventIds.values[i];
                int entityId = dataset.entityIds.values[i];
                int componentId = dataset.componentIds.values[i];
                int levelId = dataset.levelIds.values[i];
                int prevEventId = dataset.prevEventIds.values[i];
                maxEntityId = max(maxEntityId, entityId);
                maxComponentId = max(maxComponentId, componentId);
                globalCounts.put(eventId, globalCounts.getOrDefault(eventId, 0) + 1);
                entityEventCounts.put(entityEventKey(entityId, eventId),
                        entityEventCounts.getOrDefault(entityEventKey(entityId, eventId), 0) + 1);
                entityTotals.put(entityId, entityTotals.getOrDefault(entityId, 0) + 1);
                componentEventCounts.put(entityEventKey(componentId, eventId),
                        componentEventCounts.getOrDefault(entityEventKey(componentId, eventId), 0) + 1);
                componentTotals.put(componentId, componentTotals.getOrDefault(componentId, 0) + 1);
                if (config.useParameterFeatures()) {
                    int parameterHash = dataset.parameterHashes.values[i];
                    if (parameterHash != 0) {
                        templateParameterCounts.put(entityEventKey(eventId, parameterHash),
                                templateParameterCounts.getOrDefault(entityEventKey(eventId, parameterHash), 0) + 1);
                        entityParameterCounts.put(entityEventKey(entityId, parameterHash),
                                entityParameterCounts.getOrDefault(entityEventKey(entityId, parameterHash), 0) + 1);
                        parameterCounts.put(parameterHash, parameterCounts.getOrDefault(parameterHash, 0) + 1);
                    }
                }
                if (config.useSemanticFeatures()) {
                    int phraseHash = dataset.phraseHashes.values[i];
                    if (phraseHash != 0) {
                        long signatureKey = semanticSignatureKey(componentId, levelId, phraseHash);
                        long componentLevelKey = componentLevelKey(componentId, levelId);
                        semanticSignatureCounts.put(signatureKey,
                                semanticSignatureCounts.getOrDefault(signatureKey, 0) + 1);
                        componentLevelTotals.put(componentLevelKey,
                                componentLevelTotals.getOrDefault(componentLevelKey, 0) + 1);
                        phraseCounts.put(phraseHash, phraseCounts.getOrDefault(phraseHash, 0) + 1);
                        Integer previousBucket = lastSemanticSignatureBucket.put(signatureKey, bucketIndex);
                        if (previousBucket == null || previousBucket != bucketIndex) {
                            semanticSignatureBucketCounts.put(signatureKey,
                                    semanticSignatureBucketCounts.getOrDefault(signatureKey, 0) + 1);
                        }
                    }
                }
                transitionCounts.put(transitionKey(prevEventId, eventId),
                        transitionCounts.getOrDefault(transitionKey(prevEventId, eventId), 0) + 1);
                prevTotals.put(prevEventId, prevTotals.getOrDefault(prevEventId, 0) + 1);
                ++total;
            }
            Set<Integer> rare = new HashSet<>();
            for (Map.Entry<Integer, Integer> entry : globalCounts.entrySet()) {
                if (entry.getValue() <= config.rareEventMaxTrainCount) {
                    rare.add(entry.getKey());
                }
            }
            List<Map.Entry<Integer, Integer>> sorted = new ArrayList<>(globalCounts.entrySet());
            sorted.sort((a, b) -> Integer.compare(b.getValue(), a.getValue()));
            Map<Integer, Integer> topIndex = new HashMap<>();
            for (int i = 0; i < min(config.topK, sorted.size()); i++) {
                topIndex.put(sorted.get(i).getKey(), i);
            }
            return new TrainStats(globalCounts, entityEventCounts, entityTotals, componentEventCounts, componentTotals,
                    templateParameterCounts, entityParameterCounts, parameterCounts, semanticSignatureCounts,
                    semanticSignatureBucketCounts, componentLevelTotals, phraseCounts, transitionCounts, prevTotals,
                    rare, topIndex, total, max(1, globalCounts.size()), max(1, maxEntityId + 1),
                    max(1, maxComponentId + 1), max(1, parameterCounts.size()), max(1, semanticSignatureCounts.size()),
                    max(1, phraseCounts.size()));
        }

        private int entityEventCount(int entityId, int eventId) {
            return entityEventCounts.getOrDefault(entityEventKey(entityId, eventId), 0);
        }

        private double entityRarity(int entityId, int eventId) {
            int entityTotal = entityTotals.getOrDefault(entityId, 0);
            if (entityTotal < 10) {
                return globalRarity(eventId);
            }
            int count = entityEventCount(entityId, eventId);
            double probability = (count + 1.0) / (entityTotal + vocabularySize + 1.0);
            return -Math.log(probability);
        }

        private double componentRarity(int componentId, int eventId) {
            int componentTotal = componentTotals.getOrDefault(componentId, 0);
            if (componentTotal < 10) {
                return globalRarity(eventId);
            }
            int count = componentEventCounts.getOrDefault(entityEventKey(componentId, eventId), 0);
            double probability = (count + 1.0) / (componentTotal + vocabularySize + 1.0);
            return -Math.log(probability);
        }

        private double entityGivenTemplateRarity(int entityId, int eventId) {
            int eventTotal = globalCounts.getOrDefault(eventId, 0);
            if (eventTotal < 10) {
                return 0.0;
            }
            int count = entityEventCount(entityId, eventId);
            double probability = (count + 1.0) / (eventTotal + entityCount + 1.0);
            return -Math.log(probability);
        }

        private double parameterRarity(int entityId, int eventId, int parameterHash) {
            if (parameterHash == 0 || parameterVocabularySize <= 1) {
                return 0.0;
            }
            double score = 0.0;
            int templateTotal = globalCounts.getOrDefault(eventId, 0);
            if (templateTotal >= 20) {
                int count = templateParameterCounts.getOrDefault(entityEventKey(eventId, parameterHash), 0);
                double probability = (count + 1.0) / (templateTotal + min(parameterVocabularySize, 10000) + 1.0);
                score += min(8.0, -Math.log(probability));
                if (count == 0) {
                    score += 2.0;
                }
            }
            int entityTotal = entityTotals.getOrDefault(entityId, 0);
            if (entityTotal >= 20) {
                int count = entityParameterCounts.getOrDefault(entityEventKey(entityId, parameterHash), 0);
                double probability = (count + 1.0) / (entityTotal + min(parameterVocabularySize, 10000) + 1.0);
                score += 0.35 * min(8.0, -Math.log(probability));
            }
            return min(12.0, score);
        }

        private int semanticSignatureCount(int componentId, int levelId, int phraseHash) {
            return semanticSignatureCounts.getOrDefault(semanticSignatureKey(componentId, levelId, phraseHash), 0);
        }

        private int semanticSignatureBucketCount(int componentId, int levelId, int phraseHash) {
            return semanticSignatureBucketCounts.getOrDefault(semanticSignatureKey(componentId, levelId, phraseHash),
                    0);
        }

        private double semanticSignatureRarity(int componentId, int levelId, int phraseHash) {
            if (phraseHash == 0 || semanticSignatureVocabularySize <= 1) {
                return 0.0;
            }
            long componentLevelKey = componentLevelKey(componentId, levelId);
            int total = componentLevelTotals.getOrDefault(componentLevelKey, 0);
            if (total < 20) {
                return phraseRarity(phraseHash);
            }
            int count = semanticSignatureCount(componentId, levelId, phraseHash);
            double probability = (count + 1.0) / (total + semanticSignatureVocabularySize + 1.0);
            return -Math.log(probability);
        }

        private double phraseRarity(int phraseHash) {
            if (phraseHash == 0 || phraseVocabularySize <= 1) {
                return 0.0;
            }
            int count = phraseCounts.getOrDefault(phraseHash, 0);
            double probability = (count + 1.0) / (globalTotal + phraseVocabularySize + 1.0);
            return -Math.log(probability);
        }

        private double stableSignatureSuppression(int componentId, int levelId, int phraseHash, int minCount,
                int minBuckets, double maxSuppression) {
            int count = semanticSignatureCount(componentId, levelId, phraseHash);
            int buckets = semanticSignatureBucketCount(componentId, levelId, phraseHash);
            if (count < minCount || buckets < minBuckets) {
                return 0.0;
            }
            return min(maxSuppression, Math.log1p(count) + 0.5 * Math.log1p(buckets));
        }

        private double globalRarity(int eventId) {
            int count = globalCounts.getOrDefault(eventId, 0);
            double probability = (count + 1.0) / (globalTotal + vocabularySize + 1.0);
            return -Math.log(probability);
        }

        private double globalRate(int eventId) {
            return globalTotal == 0 ? 0.0 : globalCounts.getOrDefault(eventId, 0) / (double) globalTotal;
        }

        private double transitionSurprise(int prevEventId, int eventId) {
            int prevTotal = prevTotals.getOrDefault(prevEventId, 0);
            if (prevTotal < 10) {
                return globalRarity(eventId);
            }
            int count = transitionCounts.getOrDefault(transitionKey(prevEventId, eventId), 0);
            double probability = (count + 1.0) / (prevTotal + vocabularySize + 1.0);
            return -Math.log(probability);
        }
    }

    private static final class OnlineSemanticStats {
        private final Map<Long, Integer> semanticSignatureCounts;
        private final Map<Long, Integer> semanticSignatureBucketCounts;
        private final Map<Long, Integer> componentLevelTotals;
        private final Map<Integer, Integer> phraseCounts;
        private final Map<Long, Integer> lastSemanticSignatureBucket = new HashMap<>();
        private int total;
        private int semanticSignatureVocabularySize;
        private int phraseVocabularySize;

        private OnlineSemanticStats(TrainStats stats) {
            semanticSignatureCounts = new HashMap<>(stats.semanticSignatureCounts);
            semanticSignatureBucketCounts = new HashMap<>(stats.semanticSignatureBucketCounts);
            componentLevelTotals = new HashMap<>(stats.componentLevelTotals);
            phraseCounts = new HashMap<>(stats.phraseCounts);
            total = stats.globalTotal;
            semanticSignatureVocabularySize = stats.semanticSignatureVocabularySize;
            phraseVocabularySize = stats.phraseVocabularySize;
        }

        private double semanticSignatureRarity(int componentId, int levelId, int phraseHash) {
            if (phraseHash == 0 || semanticSignatureVocabularySize <= 1) {
                return 0.0;
            }
            long componentLevelKey = componentLevelKey(componentId, levelId);
            int componentLevelTotal = componentLevelTotals.getOrDefault(componentLevelKey, 0);
            if (componentLevelTotal < 20) {
                return phraseRarity(phraseHash);
            }
            long signatureKey = semanticSignatureKey(componentId, levelId, phraseHash);
            int count = semanticSignatureCounts.getOrDefault(signatureKey, 0);
            double probability = (count + 1.0) / (componentLevelTotal + semanticSignatureVocabularySize + 1.0);
            return -Math.log(probability);
        }

        private double phraseRarity(int phraseHash) {
            if (phraseHash == 0 || phraseVocabularySize <= 1) {
                return 0.0;
            }
            int count = phraseCounts.getOrDefault(phraseHash, 0);
            double probability = (count + 1.0) / (total + phraseVocabularySize + 1.0);
            return -Math.log(probability);
        }

        private double stableSignatureSuppression(int componentId, int levelId, int phraseHash) {
            long signatureKey = semanticSignatureKey(componentId, levelId, phraseHash);
            int count = semanticSignatureCounts.getOrDefault(signatureKey, 0);
            int buckets = semanticSignatureBucketCounts.getOrDefault(signatureKey, 0);
            if (count < 20 || buckets < 5) {
                return 0.0;
            }
            return min(8.0, Math.log1p(count) + 0.5 * Math.log1p(buckets));
        }

        private void update(Dataset dataset, int lineIndex) {
            int phraseHash = dataset.phraseHashes.values[lineIndex];
            if (phraseHash == 0) {
                return;
            }
            int componentId = dataset.componentIds.values[lineIndex];
            int levelId = dataset.levelIds.values[lineIndex];
            int bucketIndex = dataset.bucketIndexes.values[lineIndex];
            long signatureKey = semanticSignatureKey(componentId, levelId, phraseHash);
            long componentLevelKey = componentLevelKey(componentId, levelId);
            if (!semanticSignatureCounts.containsKey(signatureKey)) {
                ++semanticSignatureVocabularySize;
            }
            if (!phraseCounts.containsKey(phraseHash)) {
                ++phraseVocabularySize;
            }
            semanticSignatureCounts.put(signatureKey, semanticSignatureCounts.getOrDefault(signatureKey, 0) + 1);
            componentLevelTotals.put(componentLevelKey, componentLevelTotals.getOrDefault(componentLevelKey, 0) + 1);
            phraseCounts.put(phraseHash, phraseCounts.getOrDefault(phraseHash, 0) + 1);
            Integer previousBucket = lastSemanticSignatureBucket.put(signatureKey, bucketIndex);
            if (previousBucket == null || previousBucket != bucketIndex) {
                semanticSignatureBucketCounts.put(signatureKey,
                        semanticSignatureBucketCounts.getOrDefault(signatureKey, 0) + 1);
            }
            ++total;
        }
    }

    private static final class OnlineFeatureSet {
        private static final OnlineFeatureSet NONE = new OnlineFeatureSet(false, false, false, false, false, false,
                false, false, false);

        private final boolean global;
        private final boolean templateBuckets;
        private final boolean entity;
        private final boolean component;
        private final boolean componentLevel;
        private final boolean parameters;
        private final boolean semanticSignature;
        private final boolean phrase;
        private final boolean transition;

        private OnlineFeatureSet(boolean global, boolean templateBuckets, boolean entity, boolean component,
                boolean componentLevel, boolean parameters, boolean semanticSignature, boolean phrase,
                boolean transition) {
            this.global = global;
            this.templateBuckets = templateBuckets;
            this.entity = entity;
            this.component = component;
            this.componentLevel = componentLevel;
            this.parameters = parameters;
            this.semanticSignature = semanticSignature;
            this.phrase = phrase;
            this.transition = transition;
        }

        private static OnlineFeatureSet all(boolean parameters) {
            return new OnlineFeatureSet(true, true, true, true, true, parameters, true, true, true);
        }

        private static OnlineFeatureSet forBaseline(Config config, String requestedScoreMode, String baseline) {
            if (!config.useCompactOnlineState()) {
                return all(config.useParameterFeatures());
            }
            String scoreMode = requestedScoreMode == null ? config.scoreMode : requestedScoreMode;
            if ("auto".equals(scoreMode)) {
                // Selection needs the union of compact candidate features: global rarity and
                // component/stable rarity, but not entity, transition, parameter, or spike
                // state.
                if ("slow".equals(baseline)) {
                    return new OnlineFeatureSet(true, false, false, true, false, false, false, false, false);
                }
                if ("stable".equals(baseline)) {
                    return new OnlineFeatureSet(true, true, false, false, false, false, true, false, false);
                }
                return NONE;
            }
            if (usesCompositeOnlineScore(scoreMode)) {
                return all(config.useParameterFeatures());
            }
            boolean stableScore = scoreMode.contains("_stable");
            boolean componentLevelScore = scoreMode.startsWith("component_level_");
            boolean componentScore = scoreMode.startsWith("component_rarity_") || componentLevelScore;
            boolean entityScore = "rarity".equals(scoreMode) || scoreMode.startsWith("rarity_");
            boolean spikeScore = scoreMode.contains("spike");

            if ("fast".equals(baseline)) {
                if (!spikeScore) {
                    return NONE;
                }
                return new OnlineFeatureSet(true, false, entityScore, componentScore, componentLevelScore, false, false,
                        false, false);
            }
            if ("stable".equals(baseline)) {
                if (!stableScore) {
                    return NONE;
                }
                return new OnlineFeatureSet(true, true, false, false, false, false, true, false, false);
            }
            boolean global = true;
            boolean parameters = config.useParameterFeatures() && scoreMode.contains("param");
            return new OnlineFeatureSet(global, false, entityScore, componentScore, componentLevelScore, parameters,
                    false, false, false);
        }

        private static boolean usesCompositeOnlineScore(String scoreMode) {
            return "combined".equals(scoreMode) || scoreMode.startsWith("bgl_") || scoreMode.contains("semantic")
                    || "online_composite".equals(scoreMode);
        }

        private boolean any() {
            return global || templateBuckets || entity || component || componentLevel || parameters || semanticSignature
                    || phrase || transition;
        }

        @Override
        public String toString() {
            StringBuilder builder = new StringBuilder();
            if (global) {
                builder.append('g');
            }
            if (templateBuckets) {
                builder.append('b');
            }
            if (entity) {
                builder.append('e');
            }
            if (component) {
                builder.append('c');
            }
            if (componentLevel) {
                builder.append('l');
            }
            if (parameters) {
                builder.append('p');
            }
            if (semanticSignature) {
                builder.append('s');
            }
            if (phrase) {
                builder.append('r');
            }
            if (transition) {
                builder.append('t');
            }
            return builder.length() == 0 ? "-" : builder.toString();
        }
    }

    private static final class OnlineBaselineState {
        private final OnlineDecayedStats slow;
        private final OnlineDecayedStats fast;
        private final OnlineDecayedStats stable;
        private double keywordRarityGate = 1.0;

        private OnlineBaselineState(Config config) {
            this(config, config.scoreMode);
        }

        private OnlineBaselineState(Config config, String scoreMode) {
            int fastDecayInterval = max(1, min(config.onlineDecayIntervalBuckets, (int) Math
                    .round(hoursToSeconds(config.onlineFastHalfLifeHours) / max(1.0, config.bucketSeconds) / 12.0)));
            OnlineFeatureSet slowFeatures = OnlineFeatureSet.forBaseline(config, scoreMode, "slow");
            OnlineFeatureSet fastFeatures = OnlineFeatureSet.forBaseline(config, scoreMode, "fast");
            OnlineFeatureSet stableFeatures = OnlineFeatureSet.forBaseline(config, scoreMode, "stable");
            slow = new OnlineDecayedStats(daysToSeconds(config.onlineTemplateHalfLifeDays), config.bucketSeconds,
                    config.onlineDecayIntervalBuckets, config.useParameterFeatures(), config.onlineLazyDecay,
                    config.onlineCountSketchParameters, config.onlineCountSketchDepth, config.onlineCountSketchWidth,
                    slowFeatures, config.memoryLimits);
            fast = new OnlineDecayedStats(hoursToSeconds(config.onlineFastHalfLifeHours), config.bucketSeconds,
                    fastDecayInterval, config.useParameterFeatures(), config.onlineLazyDecay,
                    config.onlineCountSketchParameters, config.onlineCountSketchDepth, config.onlineCountSketchWidth,
                    fastFeatures, config.memoryLimits);
            stable = new OnlineDecayedStats(daysToSeconds(config.onlineStableHalfLifeDays), config.bucketSeconds,
                    config.onlineDecayIntervalBuckets, config.useParameterFeatures(), config.onlineLazyDecay,
                    config.onlineCountSketchParameters, config.onlineCountSketchDepth, config.onlineCountSketchWidth,
                    stableFeatures, config.memoryLimits);
        }

        private OnlineBaselineState(OnlineBaselineState other) {
            slow = new OnlineDecayedStats(other.slow);
            fast = new OnlineDecayedStats(other.fast);
            stable = new OnlineDecayedStats(other.stable);
            keywordRarityGate = other.keywordRarityGate;
        }

        private static OnlineBaselineState fromWarmup(Dataset dataset, int startBucket, int endBucket, Config config) {
            return fromWarmup(dataset, startBucket, endBucket, config, config.scoreMode);
        }

        private static OnlineBaselineState fromWarmup(Dataset dataset, int startBucket, int endBucket, Config config,
                String scoreMode) {
            OnlineBaselineState state = new OnlineBaselineState(config, scoreMode);
            for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
                long bucketKey = dataset.buckets.get(bucketIndex).bucketKey;
                state.advanceTo(bucketKey);
                for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                    state.update(dataset, i, 1.0, 1.0, 1.0);
                }
            }
            state.keywordRarityGate = keywordRarityGate(dataset, state.slow, startBucket, endBucket, config);
            return state;
        }

        private static double keywordRarityGate(Dataset dataset, OnlineDecayedStats stats, int startBucket,
                int endBucket, Config config) {
            List<Double> rarities = new ArrayList<>();
            for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
                for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                    if (dataset.keywordScores.values[i] > 0) {
                        rarities.add(stats.globalRarity(dataset.eventIds.values[i]));
                    }
                }
            }
            if (rarities.isEmpty()) {
                return 8.0;
            }
            Collections.sort(rarities);
            int index = (int) Math.ceil(max(0.0, min(1.0, config.onlineKeywordGateQuantile)) * rarities.size()) - 1;
            return max(1.0, rarities.get(min(max(index, 0), rarities.size() - 1)));
        }

        private void advanceTo(long bucketKey) {
            slow.advanceTo(bucketKey);
            fast.advanceTo(bucketKey);
            stable.advanceTo(bucketKey);
        }

        private void updateAfterDecision(Dataset dataset, int lineIndex, boolean prediction, Config config) {
            double guardedWeight = prediction ? config.onlineAlertUpdateWeight : 1.0;
            update(dataset, lineIndex, guardedWeight, 1.0, guardedWeight);
        }

        private void updateAfterDecision(StreamLine line, boolean prediction, Config config) {
            double guardedWeight = prediction ? config.onlineAlertUpdateWeight : 1.0;
            update(line, guardedWeight, 1.0, guardedWeight);
        }

        private void update(Dataset dataset, int lineIndex, double slowWeight, double fastWeight, double stableWeight) {
            slow.update(dataset, lineIndex, slowWeight);
            fast.update(dataset, lineIndex, fastWeight);
            stable.update(dataset, lineIndex, stableWeight);
        }

        private void update(StreamLine line, double slowWeight, double fastWeight, double stableWeight) {
            slow.update(line, slowWeight);
            fast.update(line, fastWeight);
            stable.update(line, stableWeight);
        }

        private double templateSpike(int eventId) {
            return ratioSpike(fast.globalProbability(eventId), slow.globalProbability(eventId), fast.globalTotal,
                    slow.globalTotal);
        }

        private double entityTemplateSpike(int entityId, int eventId) {
            return ratioSpike(fast.entityProbability(entityId, eventId), slow.entityProbability(entityId, eventId),
                    fast.entityTotal(entityId), slow.entityTotal(entityId));
        }

        private double componentLevelTemplateSpike(int componentId, int levelId, int eventId) {
            return ratioSpike(fast.componentLevelProbability(componentId, levelId, eventId),
                    slow.componentLevelProbability(componentId, levelId, eventId),
                    fast.componentLevelTotal(componentId, levelId), slow.componentLevelTotal(componentId, levelId));
        }

        private double stableSuppression(int componentId, int levelId, int phraseHash, int eventId) {
            double eventCount = stable.globalCount(eventId);
            double eventBuckets = stable.templateBucketCount(eventId);
            double templateSuppression = eventCount >= 20.0 && eventBuckets >= 5.0
                    ? min(4.0, 0.4 * Math.log1p(eventCount) + 0.6 * Math.log1p(eventBuckets))
                    : 0.0;
            return min(8.0, templateSuppression
                    + stable.stableSignatureSuppression(componentId, levelId, phraseHash, 20.0, 5.0, 6.0));
        }

        private static double ratioSpike(double fastProbability, double slowProbability, double fastTotal,
                double slowTotal) {
            if (fastTotal < 5.0 || slowTotal < 20.0 || fastProbability <= slowProbability) {
                return 0.0;
            }
            return Math.log1p(max(0.0, fastProbability / max(1.0e-12, slowProbability) - 1.0));
        }

        private String entrySummary() {
            return String.format(Locale.ROOT, "slow[%s];fast[%s];stable[%s]", slow.entrySummary(), fast.entrySummary(),
                    stable.entrySummary());
        }

        private String capSummary() {
            return String.format(Locale.ROOT, "slow[%s];fast[%s];stable[%s]", slow.capSummary(), fast.capSummary(),
                    stable.capSummary());
        }

        private long estimatedModelBytes() {
            return 64L + slow.estimatedModelBytes() + fast.estimatedModelBytes() + stable.estimatedModelBytes();
        }
    }

    private static final class OnlineDecayedStats {
        private static final double PRUNE_BELOW = 1.0e-6;
        private static final int PARAMETER_VOCABULARY_CAP = 10000;

        private final Map<Integer, Double> globalCounts = new HashMap<>();
        private final Map<Integer, Double> templateBucketCounts = new HashMap<>();
        private final Map<Integer, Long> lastTemplateBucket = new HashMap<>();
        private final Map<Long, Double> entityEventCounts = new HashMap<>();
        private final Map<Integer, Double> entityTotals = new HashMap<>();
        private final Map<Long, Double> componentEventCounts = new HashMap<>();
        private final Map<Integer, Double> componentTotals = new HashMap<>();
        private final Map<Long, Double> componentLevelEventCounts = new HashMap<>();
        private final Map<Long, Double> componentLevelTotals = new HashMap<>();
        private final Map<Long, Double> templateParameterCounts = new HashMap<>();
        private final Map<Long, Double> entityParameterCounts = new HashMap<>();
        private final Map<Integer, Double> parameterCounts = new HashMap<>();
        private final Map<Long, Double> semanticSignatureCounts = new HashMap<>();
        private final Map<Long, Double> semanticSignatureBucketCounts = new HashMap<>();
        private final Map<Long, Long> lastSemanticSignatureBucket = new HashMap<>();
        private final Map<Integer, Double> phraseCounts = new HashMap<>();
        private final Map<Long, Double> transitionCounts = new HashMap<>();
        private final Map<Integer, Double> prevTotals = new HashMap<>();
        private final double halfLifeSeconds;
        private final int bucketSeconds;
        private final int decayIntervalBuckets;
        private final boolean useParameterFeatures;
        private final boolean lazyDecay;
        private final boolean sketchParameterPairs;
        private final OnlineFeatureSet features;
        private final MemoryLimits memoryLimits;
        private final CountMinSketchLong templateParameterSketch;
        private final CountMinSketchLong entityParameterSketch;
        private double globalTotal;
        private double scale = 1.0;
        private int vocabularySize = 1;
        private int parameterVocabularySize = 1;
        private int semanticSignatureVocabularySize = 1;
        private int phraseVocabularySize = 1;
        private long lastBucketKey = Long.MIN_VALUE;
        private long droppedNewKeys;
        private long newSignatureAttempts;
        private long droppedNewSignatures;

        private OnlineDecayedStats(double halfLifeSeconds, int bucketSeconds, int decayIntervalBuckets,
                boolean useParameterFeatures, boolean lazyDecay, boolean sketchParameterPairs, int sketchDepth,
                int sketchWidth, OnlineFeatureSet features, MemoryLimits memoryLimits) {
            this.halfLifeSeconds = max(1.0, halfLifeSeconds);
            this.bucketSeconds = bucketSeconds;
            this.decayIntervalBuckets = max(1, decayIntervalBuckets);
            this.features = features;
            this.memoryLimits = memoryLimits;
            this.useParameterFeatures = useParameterFeatures && features.parameters;
            this.lazyDecay = lazyDecay;
            this.sketchParameterPairs = this.useParameterFeatures && sketchParameterPairs;
            templateParameterSketch = this.sketchParameterPairs ? new CountMinSketchLong(sketchDepth, sketchWidth)
                    : null;
            entityParameterSketch = this.sketchParameterPairs ? new CountMinSketchLong(sketchDepth, sketchWidth) : null;
        }

        private OnlineDecayedStats(OnlineDecayedStats other) {
            globalCounts.putAll(other.globalCounts);
            templateBucketCounts.putAll(other.templateBucketCounts);
            lastTemplateBucket.putAll(other.lastTemplateBucket);
            entityEventCounts.putAll(other.entityEventCounts);
            entityTotals.putAll(other.entityTotals);
            componentEventCounts.putAll(other.componentEventCounts);
            componentTotals.putAll(other.componentTotals);
            componentLevelEventCounts.putAll(other.componentLevelEventCounts);
            componentLevelTotals.putAll(other.componentLevelTotals);
            templateParameterCounts.putAll(other.templateParameterCounts);
            entityParameterCounts.putAll(other.entityParameterCounts);
            parameterCounts.putAll(other.parameterCounts);
            semanticSignatureCounts.putAll(other.semanticSignatureCounts);
            semanticSignatureBucketCounts.putAll(other.semanticSignatureBucketCounts);
            lastSemanticSignatureBucket.putAll(other.lastSemanticSignatureBucket);
            phraseCounts.putAll(other.phraseCounts);
            transitionCounts.putAll(other.transitionCounts);
            prevTotals.putAll(other.prevTotals);
            halfLifeSeconds = other.halfLifeSeconds;
            bucketSeconds = other.bucketSeconds;
            decayIntervalBuckets = other.decayIntervalBuckets;
            useParameterFeatures = other.useParameterFeatures;
            lazyDecay = other.lazyDecay;
            sketchParameterPairs = other.sketchParameterPairs;
            features = other.features;
            memoryLimits = other.memoryLimits;
            templateParameterSketch = other.templateParameterSketch == null ? null
                    : new CountMinSketchLong(other.templateParameterSketch);
            entityParameterSketch = other.entityParameterSketch == null ? null
                    : new CountMinSketchLong(other.entityParameterSketch);
            globalTotal = other.globalTotal;
            scale = other.scale;
            vocabularySize = other.vocabularySize;
            parameterVocabularySize = other.parameterVocabularySize;
            semanticSignatureVocabularySize = other.semanticSignatureVocabularySize;
            phraseVocabularySize = other.phraseVocabularySize;
            lastBucketKey = other.lastBucketKey;
            droppedNewKeys = other.droppedNewKeys;
            newSignatureAttempts = other.newSignatureAttempts;
            droppedNewSignatures = other.droppedNewSignatures;
        }

        private void advanceTo(long bucketKey) {
            if (lastBucketKey == Long.MIN_VALUE) {
                lastBucketKey = bucketKey;
                return;
            }
            long deltaBuckets = bucketKey - lastBucketKey;
            if (deltaBuckets <= 0 || deltaBuckets < decayIntervalBuckets) {
                return;
            }
            double decay = Math.exp(-(deltaBuckets * (double) bucketSeconds) / halfLifeSeconds);
            globalTotal *= decay;
            if (lazyDecay) {
                scale *= decay;
                if (scale < 1.0e-100) {
                    rescale();
                }
                lastBucketKey = bucketKey;
                return;
            }
            decayIntMap(globalCounts, decay);
            decayIntMap(templateBucketCounts, decay);
            decayLongMap(entityEventCounts, decay);
            decayIntMap(entityTotals, decay);
            decayLongMap(componentEventCounts, decay);
            decayIntMap(componentTotals, decay);
            decayLongMap(componentLevelEventCounts, decay);
            decayLongMap(componentLevelTotals, decay);
            decayLongMap(templateParameterCounts, decay);
            decayLongMap(entityParameterCounts, decay);
            decayIntMap(parameterCounts, decay);
            decayLongMap(semanticSignatureCounts, decay);
            decayLongMap(semanticSignatureBucketCounts, decay);
            decayIntMap(phraseCounts, decay);
            decayLongMap(transitionCounts, decay);
            decayIntMap(prevTotals, decay);
            pruneBucketMarkers();
            lastBucketKey = bucketKey;
        }

        private void update(Dataset dataset, int lineIndex, double weight) {
            int bucketIndex = dataset.bucketIndexes.values[lineIndex];
            long bucketKey = dataset.buckets.get(bucketIndex).bucketKey;
            advanceTo(bucketKey);
            if (weight <= 0.0 || !features.any()) {
                return;
            }
            int eventId = dataset.eventIds.values[lineIndex];
            int entityId = dataset.entityIds.values[lineIndex];
            int componentId = dataset.componentIds.values[lineIndex];
            int levelId = dataset.levelIds.values[lineIndex];
            int parameterHash = dataset.parameterHashes.values[lineIndex];
            int phraseHash = dataset.phraseHashes.values[lineIndex];
            int prevEventId = dataset.prevEventIds.values[lineIndex];
            if (features.global) {
                boolean newTemplate = !globalCounts.containsKey(eventId);
                if (add(globalCounts, eventId, weight, memoryLimits.maxTemplates) && newTemplate) {
                    ++vocabularySize;
                }
                globalTotal += weight;
            }
            if (features.templateBuckets) {
                if (templateBucketChanged(eventId, bucketKey)) {
                    add(templateBucketCounts, eventId, weight, memoryLimits.maxTemplates);
                }
            }
            long entityEventKey = features.entity || features.parameters ? entityEventKey(entityId, eventId) : 0L;
            if (features.entity) {
                add(entityEventCounts, entityEventKey, weight, memoryLimits.maxGroupTemplatePairs);
                add(entityTotals, entityId, weight, memoryLimits.maxEntities);
            }
            if (features.component) {
                add(componentEventCounts, entityEventKey(componentId, eventId), weight,
                        memoryLimits.maxGroupTemplatePairs);
                add(componentTotals, componentId, weight, memoryLimits.maxGroups);
            }
            if (features.componentLevel) {
                long componentLevelKey = componentLevelKey(componentId, levelId);
                add(componentLevelEventCounts, componentLevelEventKey(componentId, levelId, eventId), weight,
                        memoryLimits.maxGroupTemplatePairs);
                add(componentLevelTotals, componentLevelKey, weight, memoryLimits.maxGroups);
            }
            if (useParameterFeatures && parameterHash != 0) {
                if (parameterVocabularySize < PARAMETER_VOCABULARY_CAP) {
                    boolean newParameter = !parameterCounts.containsKey(parameterHash);
                    if (add(parameterCounts, parameterHash, weight, PARAMETER_VOCABULARY_CAP) && newParameter) {
                        ++parameterVocabularySize;
                    }
                }
                addTemplateParameter(entityEventKey(eventId, parameterHash), weight);
                addEntityParameter(entityEventKey(entityId, parameterHash), weight);
            }
            if ((features.semanticSignature || features.phrase) && phraseHash != 0) {
                long signatureKey = semanticSignatureKey(componentId, levelId, phraseHash);
                if (features.semanticSignature) {
                    boolean newSignature = !semanticSignatureCounts.containsKey(signatureKey);
                    if (newSignature) {
                        ++newSignatureAttempts;
                    }
                    if (add(semanticSignatureCounts, signatureKey, weight, memoryLimits.maxStableSignatures)) {
                        if (newSignature) {
                            ++semanticSignatureVocabularySize;
                        }
                    } else if (newSignature) {
                        ++droppedNewSignatures;
                    }
                }
                if (features.phrase) {
                    boolean newPhrase = !phraseCounts.containsKey(phraseHash);
                    if (add(phraseCounts, phraseHash, weight, memoryLimits.maxStableSignatures) && newPhrase) {
                        ++phraseVocabularySize;
                    }
                }
                if (features.semanticSignature) {
                    if (semanticSignatureBucketChanged(signatureKey, bucketKey)) {
                        add(semanticSignatureBucketCounts, signatureKey, weight, memoryLimits.maxStableSignatures);
                    }
                }
            }
            if (features.transition) {
                add(transitionCounts, transitionKey(prevEventId, eventId), weight, memoryLimits.maxGroupTemplatePairs);
                add(prevTotals, prevEventId, weight, memoryLimits.maxTemplates);
            }
        }

        private void update(StreamLine line, double weight) {
            advanceTo(line.bucketKey);
            if (weight <= 0.0 || !features.any()) {
                return;
            }
            int eventId = line.eventId;
            int entityId = line.entityId;
            int componentId = line.componentId;
            int levelId = line.levelId;
            int parameterHash = line.parameterHash;
            int phraseHash = line.phraseHash;
            int prevEventId = line.prevEventId;
            if (features.global) {
                boolean newTemplate = !globalCounts.containsKey(eventId);
                if (add(globalCounts, eventId, weight, memoryLimits.maxTemplates) && newTemplate) {
                    ++vocabularySize;
                }
                globalTotal += weight;
            }
            if (features.templateBuckets) {
                if (templateBucketChanged(eventId, line.bucketKey)) {
                    add(templateBucketCounts, eventId, weight, memoryLimits.maxTemplates);
                }
            }
            long entityEventKey = features.entity || features.parameters ? entityEventKey(entityId, eventId) : 0L;
            if (features.entity) {
                add(entityEventCounts, entityEventKey, weight, memoryLimits.maxGroupTemplatePairs);
                add(entityTotals, entityId, weight, memoryLimits.maxEntities);
            }
            if (features.component) {
                add(componentEventCounts, entityEventKey(componentId, eventId), weight,
                        memoryLimits.maxGroupTemplatePairs);
                add(componentTotals, componentId, weight, memoryLimits.maxGroups);
            }
            if (features.componentLevel) {
                long componentLevelKey = componentLevelKey(componentId, levelId);
                add(componentLevelEventCounts, componentLevelEventKey(componentId, levelId, eventId), weight,
                        memoryLimits.maxGroupTemplatePairs);
                add(componentLevelTotals, componentLevelKey, weight, memoryLimits.maxGroups);
            }
            if (useParameterFeatures && parameterHash != 0) {
                if (parameterVocabularySize < PARAMETER_VOCABULARY_CAP) {
                    boolean newParameter = !parameterCounts.containsKey(parameterHash);
                    if (add(parameterCounts, parameterHash, weight, PARAMETER_VOCABULARY_CAP) && newParameter) {
                        ++parameterVocabularySize;
                    }
                }
                addTemplateParameter(entityEventKey(eventId, parameterHash), weight);
                addEntityParameter(entityEventKey(entityId, parameterHash), weight);
            }
            if ((features.semanticSignature || features.phrase) && phraseHash != 0) {
                long signatureKey = semanticSignatureKey(componentId, levelId, phraseHash);
                if (features.semanticSignature) {
                    boolean newSignature = !semanticSignatureCounts.containsKey(signatureKey);
                    if (newSignature) {
                        ++newSignatureAttempts;
                    }
                    if (add(semanticSignatureCounts, signatureKey, weight, memoryLimits.maxStableSignatures)) {
                        if (newSignature) {
                            ++semanticSignatureVocabularySize;
                        }
                    } else if (newSignature) {
                        ++droppedNewSignatures;
                    }
                }
                if (features.phrase) {
                    boolean newPhrase = !phraseCounts.containsKey(phraseHash);
                    if (add(phraseCounts, phraseHash, weight, memoryLimits.maxStableSignatures) && newPhrase) {
                        ++phraseVocabularySize;
                    }
                }
                if (features.semanticSignature) {
                    if (semanticSignatureBucketChanged(signatureKey, line.bucketKey)) {
                        add(semanticSignatureBucketCounts, signatureKey, weight, memoryLimits.maxStableSignatures);
                    }
                }
            }
            if (features.transition) {
                add(transitionCounts, transitionKey(prevEventId, eventId), weight, memoryLimits.maxGroupTemplatePairs);
                add(prevTotals, prevEventId, weight, memoryLimits.maxTemplates);
            }
        }

        private double globalCount(int eventId) {
            return count(globalCounts, eventId);
        }

        private double templateBucketCount(int eventId) {
            return count(templateBucketCounts, eventId);
        }

        private double entityTotal(int entityId) {
            return count(entityTotals, entityId);
        }

        private double componentLevelTotal(int componentId, int levelId) {
            return count(componentLevelTotals, componentLevelKey(componentId, levelId));
        }

        private double globalProbability(int eventId) {
            return smoothedProbability(globalCount(eventId), globalTotal, vocabularySize);
        }

        private double entityProbability(int entityId, int eventId) {
            double total = entityTotal(entityId);
            if (total < 1.0) {
                return globalProbability(eventId);
            }
            return smoothedProbability(count(entityEventCounts, entityEventKey(entityId, eventId)), total,
                    vocabularySize);
        }

        private double componentLevelProbability(int componentId, int levelId, int eventId) {
            double total = componentLevelTotal(componentId, levelId);
            if (total < 1.0) {
                return globalProbability(eventId);
            }
            return smoothedProbability(
                    count(componentLevelEventCounts, componentLevelEventKey(componentId, levelId, eventId)), total,
                    vocabularySize);
        }

        private double globalRarity(int eventId) {
            return -Math.log(globalProbability(eventId));
        }

        private double entityRarity(int entityId, int eventId) {
            double total = entityTotal(entityId);
            if (total < 10.0) {
                return globalRarity(eventId);
            }
            return -Math.log(entityProbability(entityId, eventId));
        }

        private double entityGivenTemplateRarity(int entityId, int eventId) {
            double eventTotal = globalCount(eventId);
            if (eventTotal < 10.0) {
                return 0.0;
            }
            double count = count(entityEventCounts, entityEventKey(entityId, eventId));
            return -Math.log(smoothedProbability(count, eventTotal, max(1, entityTotals.size())));
        }

        private double componentRarity(int componentId, int eventId) {
            double total = count(componentTotals, componentId);
            if (total < 10.0) {
                return globalRarity(eventId);
            }
            return -Math.log(smoothedProbability(count(componentEventCounts, entityEventKey(componentId, eventId)),
                    total, vocabularySize));
        }

        private double componentLevelRarity(int componentId, int levelId, int eventId) {
            double total = componentLevelTotal(componentId, levelId);
            if (total < 10.0) {
                return componentRarity(componentId, eventId);
            }
            return -Math.log(componentLevelProbability(componentId, levelId, eventId));
        }

        private double parameterRarity(int entityId, int eventId, int parameterHash) {
            if (!useParameterFeatures || parameterHash == 0 || parameterVocabularySize <= 1) {
                return 0.0;
            }
            double score = 0.0;
            double templateTotal = globalCount(eventId);
            if (templateTotal >= 20.0) {
                double count = templateParameterCount(entityEventKey(eventId, parameterHash));
                score += min(8.0, -Math.log(smoothedProbability(count, templateTotal,
                        min(parameterVocabularySize, PARAMETER_VOCABULARY_CAP))));
                if (count == 0.0) {
                    score += 2.0;
                }
            }
            double entityTotal = entityTotal(entityId);
            if (entityTotal >= 20.0) {
                double count = entityParameterCount(entityEventKey(entityId, parameterHash));
                score += 0.35 * min(8.0, -Math.log(smoothedProbability(count, entityTotal,
                        min(parameterVocabularySize, PARAMETER_VOCABULARY_CAP))));
            }
            return min(12.0, score);
        }

        private double semanticSignatureRarity(int componentId, int levelId, int phraseHash) {
            if (phraseHash == 0 || semanticSignatureVocabularySize <= 1) {
                return 0.0;
            }
            double total = componentLevelTotal(componentId, levelId);
            if (total < 20.0) {
                return phraseRarity(phraseHash);
            }
            double count = count(semanticSignatureCounts, semanticSignatureKey(componentId, levelId, phraseHash));
            return -Math.log(smoothedProbability(count, total, semanticSignatureVocabularySize));
        }

        private double phraseRarity(int phraseHash) {
            if (phraseHash == 0 || phraseVocabularySize <= 1) {
                return 0.0;
            }
            return -Math.log(smoothedProbability(count(phraseCounts, phraseHash), globalTotal, phraseVocabularySize));
        }

        private double stableSignatureSuppression(int componentId, int levelId, int phraseHash, double minCount,
                double minBuckets, double maxSuppression) {
            if (phraseHash == 0) {
                return 0.0;
            }
            long signatureKey = semanticSignatureKey(componentId, levelId, phraseHash);
            double count = count(semanticSignatureCounts, signatureKey);
            double buckets = count(semanticSignatureBucketCounts, signatureKey);
            if (count < minCount || buckets < minBuckets) {
                return 0.0;
            }
            return min(maxSuppression, Math.log1p(count) + 0.5 * Math.log1p(buckets));
        }

        private double transitionSurprise(int prevEventId, int eventId) {
            double prevTotal = count(prevTotals, prevEventId);
            if (prevTotal < 10.0) {
                return globalRarity(eventId);
            }
            double count = count(transitionCounts, transitionKey(prevEventId, eventId));
            return -Math.log(smoothedProbability(count, prevTotal, vocabularySize));
        }

        private String entrySummary() {
            return String.format(Locale.ROOT,
                    "total=%.1f scale=%.3g features=%s maps=%d dropped=%d sketchParam=%s global=%d entityEvent=%d entityTotals=%d componentEvent=%d componentTotals=%d componentLevelEvent=%d componentLevelTotals=%d templateParam=%d entityParam=%d param=%d signature=%d signatureBuckets=%d phrase=%d transition=%d prevTotals=%d",
                    globalTotal, scale, features, entryCount(), droppedNewKeys, sketchParameterPairs ? "true" : "false",
                    globalCounts.size(), entityEventCounts.size(), entityTotals.size(), componentEventCounts.size(),
                    componentTotals.size(), componentLevelEventCounts.size(), componentLevelTotals.size(),
                    templateParameterCounts.size(), entityParameterCounts.size(), parameterCounts.size(),
                    semanticSignatureCounts.size(), semanticSignatureBucketCounts.size(), phraseCounts.size(),
                    transitionCounts.size(), prevTotals.size());
        }

        private String capSummary() {
            return String.format(Locale.ROOT,
                    "templates_used=%d/%d component_template_pairs_used=%d/%d stable_signatures_used=%d/%d stable_signature_buckets_used=%d/%d dropped_new_keys=%d dropped_new_signature_rate=%.6f",
                    globalCounts.size(), memoryLimits.maxTemplates, componentEventCounts.size(),
                    memoryLimits.maxGroupTemplatePairs, semanticSignatureCounts.size(),
                    memoryLimits.maxStableSignatures, semanticSignatureBucketCounts.size(),
                    memoryLimits.maxStableSignatures, droppedNewKeys,
                    droppedNewSignatures / max(1.0, (double) newSignatureAttempts));
        }

        private int entryCount() {
            return globalCounts.size() + templateBucketCounts.size() + lastTemplateBucket.size()
                    + entityEventCounts.size() + entityTotals.size() + componentEventCounts.size()
                    + componentTotals.size() + componentLevelEventCounts.size() + componentLevelTotals.size()
                    + templateParameterCounts.size() + entityParameterCounts.size() + parameterCounts.size()
                    + semanticSignatureCounts.size() + semanticSignatureBucketCounts.size()
                    + lastSemanticSignatureBucket.size() + phraseCounts.size() + transitionCounts.size()
                    + prevTotals.size();
        }

        private long estimatedModelBytes() {
            long bytes = 192L;
            bytes += intDoubleMapBytes(globalCounts);
            bytes += intDoubleMapBytes(templateBucketCounts);
            bytes += intLongMapBytes(lastTemplateBucket);
            bytes += longDoubleMapBytes(entityEventCounts);
            bytes += intDoubleMapBytes(entityTotals);
            bytes += longDoubleMapBytes(componentEventCounts);
            bytes += intDoubleMapBytes(componentTotals);
            bytes += longDoubleMapBytes(componentLevelEventCounts);
            bytes += longDoubleMapBytes(componentLevelTotals);
            bytes += longDoubleMapBytes(templateParameterCounts);
            bytes += longDoubleMapBytes(entityParameterCounts);
            bytes += intDoubleMapBytes(parameterCounts);
            bytes += longDoubleMapBytes(semanticSignatureCounts);
            bytes += longDoubleMapBytes(semanticSignatureBucketCounts);
            bytes += longLongMapBytes(lastSemanticSignatureBucket);
            bytes += intDoubleMapBytes(phraseCounts);
            bytes += longDoubleMapBytes(transitionCounts);
            bytes += intDoubleMapBytes(prevTotals);
            if (templateParameterSketch != null) {
                bytes += templateParameterSketch.estimatedModelBytes();
            }
            if (entityParameterSketch != null) {
                bytes += entityParameterSketch.estimatedModelBytes();
            }
            return bytes;
        }

        private static double smoothedProbability(double count, double total, int vocabularySize) {
            return (count + 1.0) / (total + max(1, vocabularySize) + 1.0);
        }

        private double count(Map<Integer, Double> map, int key) {
            double value = map.getOrDefault(key, 0.0);
            return lazyDecay ? value * scale : value;
        }

        private double count(Map<Long, Double> map, long key) {
            double value = map.getOrDefault(key, 0.0);
            return lazyDecay ? value * scale : value;
        }

        private double templateParameterCount(long key) {
            if (sketchParameterPairs) {
                return templateParameterSketch.estimate(key) * (lazyDecay ? scale : 1.0);
            }
            return count(templateParameterCounts, key);
        }

        private double entityParameterCount(long key) {
            if (sketchParameterPairs) {
                return entityParameterSketch.estimate(key) * (lazyDecay ? scale : 1.0);
            }
            return count(entityParameterCounts, key);
        }

        private void add(Map<Integer, Double> map, int key, double amount) {
            double storedAmount = lazyDecay ? amount / scale : amount;
            map.put(key, map.getOrDefault(key, 0.0) + storedAmount);
        }

        private void add(Map<Long, Double> map, long key, double amount) {
            double storedAmount = lazyDecay ? amount / scale : amount;
            map.put(key, map.getOrDefault(key, 0.0) + storedAmount);
        }

        private boolean add(Map<Integer, Double> map, int key, double amount, int cap) {
            boolean existing = map.containsKey(key);
            if (!canAdmit(map, existing, cap)) {
                return false;
            }
            add(map, key, amount);
            return true;
        }

        private boolean add(Map<Long, Double> map, long key, double amount, int cap) {
            boolean existing = map.containsKey(key);
            if (!canAdmit(map, existing, cap)) {
                return false;
            }
            add(map, key, amount);
            return true;
        }

        private boolean canAdmit(Map<?, ?> map, boolean existing, int cap) {
            if (!memoryLimits.bounded || existing || map.size() < cap) {
                return true;
            }
            ++droppedNewKeys;
            return false;
        }

        private boolean templateBucketChanged(int eventId, long bucketKey) {
            Long previous = lastTemplateBucket.get(eventId);
            if (!canAdmit(lastTemplateBucket, previous != null, memoryLimits.maxTemplates)) {
                return false;
            }
            lastTemplateBucket.put(eventId, bucketKey);
            return previous == null || previous.longValue() != bucketKey;
        }

        private boolean semanticSignatureBucketChanged(long signatureKey, long bucketKey) {
            Long previous = lastSemanticSignatureBucket.get(signatureKey);
            if (!canAdmit(lastSemanticSignatureBucket, previous != null, memoryLimits.maxStableSignatures)) {
                return false;
            }
            lastSemanticSignatureBucket.put(signatureKey, bucketKey);
            return previous == null || previous.longValue() != bucketKey;
        }

        private void addTemplateParameter(long key, double amount) {
            if (sketchParameterPairs) {
                templateParameterSketch.add(key, lazyDecay ? amount / scale : amount);
            } else {
                add(templateParameterCounts, key, amount, memoryLimits.maxGroupTemplatePairs);
            }
        }

        private void addEntityParameter(long key, double amount) {
            if (sketchParameterPairs) {
                entityParameterSketch.add(key, lazyDecay ? amount / scale : amount);
            } else {
                add(entityParameterCounts, key, amount, memoryLimits.maxGroupTemplatePairs);
            }
        }

        private void rescale() {
            rescaleIntMap(globalCounts);
            rescaleIntMap(templateBucketCounts);
            rescaleLongMap(entityEventCounts);
            rescaleIntMap(entityTotals);
            rescaleLongMap(componentEventCounts);
            rescaleIntMap(componentTotals);
            rescaleLongMap(componentLevelEventCounts);
            rescaleLongMap(componentLevelTotals);
            rescaleLongMap(templateParameterCounts);
            rescaleLongMap(entityParameterCounts);
            if (templateParameterSketch != null) {
                templateParameterSketch.multiply(scale);
            }
            if (entityParameterSketch != null) {
                entityParameterSketch.multiply(scale);
            }
            rescaleIntMap(parameterCounts);
            rescaleLongMap(semanticSignatureCounts);
            rescaleLongMap(semanticSignatureBucketCounts);
            rescaleIntMap(phraseCounts);
            rescaleLongMap(transitionCounts);
            rescaleIntMap(prevTotals);
            pruneBucketMarkers();
            scale = 1.0;
        }

        private void pruneBucketMarkers() {
            lastTemplateBucket.keySet().retainAll(globalCounts.keySet());
            lastSemanticSignatureBucket.keySet().retainAll(semanticSignatureCounts.keySet());
        }

        private void rescaleIntMap(Map<Integer, Double> map) {
            Iterator<Map.Entry<Integer, Double>> iterator = map.entrySet().iterator();
            while (iterator.hasNext()) {
                Map.Entry<Integer, Double> entry = iterator.next();
                double value = entry.getValue() * scale;
                if (value < PRUNE_BELOW) {
                    iterator.remove();
                } else {
                    entry.setValue(value);
                }
            }
        }

        private void rescaleLongMap(Map<Long, Double> map) {
            Iterator<Map.Entry<Long, Double>> iterator = map.entrySet().iterator();
            while (iterator.hasNext()) {
                Map.Entry<Long, Double> entry = iterator.next();
                double value = entry.getValue() * scale;
                if (value < PRUNE_BELOW) {
                    iterator.remove();
                } else {
                    entry.setValue(value);
                }
            }
        }

        private static void decayIntMap(Map<Integer, Double> map, double factor) {
            Iterator<Map.Entry<Integer, Double>> iterator = map.entrySet().iterator();
            while (iterator.hasNext()) {
                Map.Entry<Integer, Double> entry = iterator.next();
                double value = entry.getValue() * factor;
                if (value < PRUNE_BELOW) {
                    iterator.remove();
                } else {
                    entry.setValue(value);
                }
            }
        }

        private static void decayLongMap(Map<Long, Double> map, double factor) {
            Iterator<Map.Entry<Long, Double>> iterator = map.entrySet().iterator();
            while (iterator.hasNext()) {
                Map.Entry<Long, Double> entry = iterator.next();
                double value = entry.getValue() * factor;
                if (value < PRUNE_BELOW) {
                    iterator.remove();
                } else {
                    entry.setValue(value);
                }
            }
        }
    }

    private static final class CountMinSketchLong {
        private static final long[] SEEDS = { 0x9E3779B97F4A7C15L, 0xC2B2AE3D27D4EB4FL, 0x165667B19E3779F9L,
                0x85EBCA77C2B2AE63L, 0x27D4EB2F165667C5L, 0xD6E8FEB86659FD93L, 0xA5A3564E27FDCB2DL,
                0x9FB21C651E98DF25L };

        private final int depth;
        private final int width;
        private final double[][] counts;

        private CountMinSketchLong(int depth, int width) {
            this.depth = max(1, min(depth, SEEDS.length));
            this.width = max(1024, width);
            counts = new double[this.depth][this.width];
        }

        private CountMinSketchLong(CountMinSketchLong other) {
            depth = other.depth;
            width = other.width;
            counts = new double[depth][width];
            for (int i = 0; i < depth; i++) {
                System.arraycopy(other.counts[i], 0, counts[i], 0, width);
            }
        }

        private long estimatedModelBytes() {
            return 48L + 24L + 8L * depth + depth * (24L + 8L * width);
        }

        private void add(long key, double amount) {
            if (amount <= 0.0) {
                return;
            }
            for (int i = 0; i < depth; i++) {
                counts[i][index(key, i)] += amount;
            }
        }

        private double estimate(long key) {
            double result = Double.POSITIVE_INFINITY;
            for (int i = 0; i < depth; i++) {
                result = min(result, counts[i][index(key, i)]);
            }
            return Double.isFinite(result) ? result : 0.0;
        }

        private void multiply(double factor) {
            if (factor == 1.0) {
                return;
            }
            for (int i = 0; i < depth; i++) {
                double[] row = counts[i];
                for (int j = 0; j < row.length; j++) {
                    row[j] *= factor;
                }
            }
        }

        private int index(long key, int row) {
            long mixed = mix64(key ^ SEEDS[row]);
            return (int) ((mixed & 0x7fffffffffffffffL) % width);
        }

        private static long mix64(long value) {
            value ^= value >>> 33;
            value *= 0xff51afd7ed558ccdL;
            value ^= value >>> 33;
            value *= 0xc4ceb9fe1a85ec53L;
            value ^= value >>> 33;
            return value;
        }
    }

    private static final class DecayedScoreHistogram {
        private final double maxScore;
        private final double halfLifeSeconds;
        private final int bucketSeconds;
        private final double[] counts;
        private double total;
        private long lastBucketKey = Long.MIN_VALUE;

        private DecayedScoreHistogram(double maxScore, int bins, double halfLifeSeconds, int bucketSeconds) {
            this.maxScore = max(1.0, maxScore);
            this.halfLifeSeconds = max(1.0, halfLifeSeconds);
            this.bucketSeconds = bucketSeconds;
            counts = new double[max(8, bins)];
        }

        private DecayedScoreHistogram(DecayedScoreHistogram other) {
            maxScore = other.maxScore;
            halfLifeSeconds = other.halfLifeSeconds;
            bucketSeconds = other.bucketSeconds;
            counts = java.util.Arrays.copyOf(other.counts, other.counts.length);
            total = other.total;
            lastBucketKey = other.lastBucketKey;
        }

        private void advanceTo(long bucketKey) {
            if (lastBucketKey == Long.MIN_VALUE) {
                lastBucketKey = bucketKey;
                return;
            }
            long deltaBuckets = bucketKey - lastBucketKey;
            if (deltaBuckets <= 0) {
                return;
            }
            double decay = Math.exp(-(deltaBuckets * (double) bucketSeconds) / halfLifeSeconds);
            for (int i = 0; i < counts.length; i++) {
                counts[i] *= decay;
            }
            total *= decay;
            lastBucketKey = bucketKey;
        }

        private void update(double score, long bucketKey) {
            advanceTo(bucketKey);
            int bin = bin(score);
            counts[bin] += 1.0;
            total += 1.0;
        }

        private double total() {
            return total;
        }

        private double quantile(double quantile) {
            if (total <= 0.0) {
                return Double.POSITIVE_INFINITY;
            }
            double target = max(0.0, min(1.0, quantile)) * total;
            double seen = 0.0;
            for (int i = 0; i < counts.length; i++) {
                seen += counts[i];
                if (seen >= target) {
                    return i * maxScore / (counts.length - 1.0);
                }
            }
            return maxScore;
        }

        private int bin(double score) {
            double clean = Double.isFinite(score) ? max(0.0, min(maxScore, score)) : maxScore;
            int bin = (int) Math.floor(clean / maxScore * (counts.length - 1));
            return min(max(bin, 0), counts.length - 1);
        }
    }

    private static final class OnlineThresholdState {
        private final DecayedScoreHistogram global;
        private final Map<Long, DecayedScoreHistogram> groupHistograms = new HashMap<>();
        private final String groupMode;
        private final int minGroupCount;
        private final double maxScore;
        private final int bins;
        private final double halfLifeSeconds;
        private final int bucketSeconds;

        private OnlineThresholdState(Config config) {
            maxScore = config.rollingScoreMax;
            bins = config.rollingBins;
            halfLifeSeconds = daysToSeconds(config.onlineQuantileHalfLifeDays);
            bucketSeconds = config.bucketSeconds;
            groupMode = "auto".equals(config.onlineThresholdGroup) ? "global" : config.onlineThresholdGroup;
            minGroupCount = max(1, config.onlineThresholdMinCount);
            global = new DecayedScoreHistogram(maxScore, bins, halfLifeSeconds, bucketSeconds);
        }

        private OnlineThresholdState(OnlineThresholdState other) {
            maxScore = other.maxScore;
            bins = other.bins;
            halfLifeSeconds = other.halfLifeSeconds;
            bucketSeconds = other.bucketSeconds;
            groupMode = other.groupMode;
            minGroupCount = other.minGroupCount;
            global = new DecayedScoreHistogram(other.global);
            for (Map.Entry<Long, DecayedScoreHistogram> entry : other.groupHistograms.entrySet()) {
                groupHistograms.put(entry.getKey(), new DecayedScoreHistogram(entry.getValue()));
            }
        }

        private void advanceTo(long bucketKey) {
            global.advanceTo(bucketKey);
        }

        private void update(Dataset dataset, int lineIndex, double score, long bucketKey) {
            global.update(score, bucketKey);
            Long groupKey = groupKey(dataset, lineIndex);
            if (groupKey != null) {
                groupHistogram(groupKey, true).update(score, bucketKey);
            }
        }

        private double quantile(Dataset dataset, int lineIndex, double quantile, long bucketKey) {
            double globalThreshold = global.quantile(quantile);
            Long groupKey = groupKey(dataset, lineIndex);
            if (groupKey != null) {
                DecayedScoreHistogram histogram = groupHistogram(groupKey, false);
                if (histogram != null) {
                    histogram.advanceTo(bucketKey);
                    if (histogram.total() >= minGroupCount) {
                        return max(globalThreshold, histogram.quantile(quantile));
                    }
                }
            }
            return globalThreshold;
        }

        private DecayedScoreHistogram groupHistogram(Long groupKey, boolean create) {
            DecayedScoreHistogram histogram = groupHistograms.get(groupKey);
            if (histogram == null && create) {
                histogram = new DecayedScoreHistogram(maxScore, bins, halfLifeSeconds, bucketSeconds);
                groupHistograms.put(groupKey, histogram);
            }
            return histogram;
        }

        private Long groupKey(Dataset dataset, int lineIndex) {
            if ("global".equals(groupMode)) {
                return null;
            }
            int componentId = dataset.componentIds.values[lineIndex];
            int levelId = dataset.levelIds.values[lineIndex];
            int keyword = keywordClass(dataset.keywordScores.values[lineIndex]);
            if ("component_level".equals(groupMode)) {
                return componentLevelKey(componentId, levelId);
            }
            if ("component_keyword".equals(groupMode)) {
                return entityEventKey(componentId, keyword);
            }
            if ("level_keyword".equals(groupMode)) {
                return entityEventKey(levelId, keyword);
            }
            throw new IllegalArgumentException("unknown online threshold group " + groupMode);
        }
    }

    private static final class AnchoredDynamicThresholdState {
        private final DynamicThresholdStats global;
        private final Map<Integer, DynamicThresholdStats> parentStats = new HashMap<>();
        private final Map<Long, DynamicThresholdStats> leafStats = new HashMap<>();
        private final String groupMode;
        private final int minGroupCount;
        private final double shrinkageK;
        private final double zFactor;
        private final boolean floorParentThreshold;
        private final boolean warmupRelativeDynamic;
        private final double halfLifeSeconds;
        private final int bucketSeconds;
        private boolean lastThresholdAnchorControlled = true;

        private AnchoredDynamicThresholdState(Config config) {
            groupMode = effectiveGroupMode(config);
            minGroupCount = max(1, config.onlineThresholdMinCount);
            shrinkageK = max(1.0, config.anchoredThresholdShrinkageK);
            zFactor = config.zFactor;
            floorParentThreshold = config.anchoredThresholdFloorParent;
            warmupRelativeDynamic = config.useWarmupRelativeDynamicThreshold();
            halfLifeSeconds = daysToSeconds(config.anchoredThresholdHalfLifeDays);
            bucketSeconds = config.bucketSeconds;
            global = new DynamicThresholdStats(halfLifeSeconds, bucketSeconds);
        }

        private static String effectiveGroupMode(Config config) {
            return "auto".equals(config.onlineThresholdGroup) ? "global" : config.onlineThresholdGroup;
        }

        private static AnchoredDynamicThresholdState fromWarmup(Dataset dataset, OnlineBaselineState seed,
                int startBucket, int endBucket, double quantile, Config config) {
            return fromWarmup(dataset, seed, startBucket, endBucket, quantile, config, config.scoreMode, Double.NaN,
                    false);
        }

        private static AnchoredDynamicThresholdState fromWarmup(Dataset dataset, OnlineBaselineState seed,
                int startBucket, int endBucket, double quantile, Config config, String scoreMode,
                double globalAnchorOverride, boolean updateSeed) {
            AnchoredDynamicThresholdState state = new AnchoredDynamicThresholdState(config);
            List<Double> globalScores = new ArrayList<>();
            Map<Integer, List<Double>> parentScores = new HashMap<>();
            Map<Long, List<Double>> leafScores = new HashMap<>();
            OnlineThresholdState histogramAnchor = config.useHistogramAnchorQuantile(scoreMode)
                    ? new OnlineThresholdState(config)
                    : null;
            for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
                long bucketKey = dataset.buckets.get(bucketIndex).bucketKey;
                state.advanceTo(bucketKey);
                if (updateSeed) {
                    seed.advanceTo(bucketKey);
                }
                for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                    double rawScore = onlineDecayedLineScore(dataset, seed, i, config, scoreMode);
                    double score = anchoredThresholdScore(rawScore, config);
                    globalScores.add(score);
                    if (histogramAnchor != null) {
                        histogramAnchor.update(dataset, i, rawScore, bucketKey);
                    }
                    state.global.update(score, bucketKey);
                    Integer parentKey = state.parentKey(dataset, i);
                    if (parentKey != null) {
                        parentScores.computeIfAbsent(parentKey, ignored -> new ArrayList<>()).add(score);
                        state.parent(parentKey, true).update(score, bucketKey);
                    }
                    Long leafKey = state.leafKey(dataset, i);
                    if (leafKey != null) {
                        leafScores.computeIfAbsent(leafKey, ignored -> new ArrayList<>()).add(score);
                        state.leaf(leafKey, true).update(score, bucketKey);
                    }
                    if (updateSeed) {
                        seed.update(dataset, i, 1.0, 1.0, 1.0);
                    }
                }
            }
            double globalAnchor = Double.isFinite(globalAnchorOverride) ? globalAnchorOverride
                    : config.useHistogramAnchorQuantile(scoreMode)
                            ? anchoredThresholdScore(histogramAnchor.global.quantile(quantile), config)
                            : listQuantile(globalScores, quantile);
            state.global.setWarmupAnchor(globalAnchor);
            state.global.captureWarmupDynamic(state.zFactor);
            for (Map.Entry<Integer, DynamicThresholdStats> entry : state.parentStats.entrySet()) {
                List<Double> scores = parentScores.get(entry.getKey());
                double anchor = scores != null && scores.size() >= state.minGroupCount ? listQuantile(scores, quantile)
                        : globalAnchor;
                entry.getValue().setWarmupAnchor(anchor);
                entry.getValue().captureWarmupDynamic(state.zFactor);
            }
            for (Map.Entry<Long, DynamicThresholdStats> entry : state.leafStats.entrySet()) {
                List<Double> scores = leafScores.get(entry.getKey());
                Integer parentKey = state.parentKey(entry.getKey());
                DynamicThresholdStats parent = parentKey == null ? null : state.parent(parentKey, false);
                double parentAnchor = parent == null ? globalAnchor : parent.warmupAnchor();
                double anchor = scores != null && scores.size() >= state.minGroupCount ? listQuantile(scores, quantile)
                        : parentAnchor;
                entry.getValue().setWarmupAnchor(anchor);
                entry.getValue().captureWarmupDynamic(state.zFactor);
            }
            return state;
        }

        private void advanceTo(long bucketKey) {
            global.advanceTo(bucketKey);
        }

        private void update(Dataset dataset, int lineIndex, double score, long bucketKey) {
            global.update(score, bucketKey);
            Integer parentKey = parentKey(dataset, lineIndex);
            if (parentKey != null) {
                parent(parentKey, true).update(score, bucketKey);
            }
            Long leafKey = leafKey(dataset, lineIndex);
            if (leafKey != null) {
                leaf(leafKey, true).update(score, bucketKey);
            }
        }

        private double threshold(Dataset dataset, int lineIndex, long bucketKey) {
            double globalThreshold = global.threshold(zFactor, bucketKey, warmupRelativeDynamic);
            double globalAnchor = global.warmupAnchor();
            Integer parentKey = parentKey(dataset, lineIndex);
            if (parentKey == null) {
                recordThresholdControl(globalThreshold, globalAnchor);
                return globalThreshold;
            }
            DynamicThresholdStats parent = parent(parentKey, false);
            double parentThreshold = globalThreshold;
            double parentAnchorFloor = globalAnchor;
            if (parent != null) {
                double local = parent.threshold(zFactor, bucketKey, warmupRelativeDynamic);
                double blended = blend(local, globalThreshold, parent.count(bucketKey));
                if (floorParentThreshold()) {
                    blended = max(globalThreshold, blended);
                }
                parentThreshold = max(parent.warmupAnchor(), blended);
                parentAnchorFloor = parent.warmupAnchor();
            }
            if ("component".equals(groupMode)) {
                recordThresholdControl(parentThreshold, parentAnchorFloor);
                return parentThreshold;
            }
            Long leafKey = leafKey(dataset, lineIndex);
            if (leafKey == null) {
                recordThresholdControl(parentThreshold, parentAnchorFloor);
                return parentThreshold;
            }
            DynamicThresholdStats leaf = leaf(leafKey, false);
            if (leaf == null) {
                recordThresholdControl(parentThreshold, parentAnchorFloor);
                return parentThreshold;
            }
            double local = leaf.threshold(zFactor, bucketKey, warmupRelativeDynamic);
            double blended = blend(local, parentThreshold, leaf.count(bucketKey));
            if (floorParentThreshold()) {
                blended = max(parentThreshold, blended);
            }
            double threshold = max(leaf.warmupAnchor(), blended);
            recordThresholdControl(threshold, leaf.warmupAnchor());
            return threshold;
        }

        private void recordThresholdControl(double threshold, double anchorFloor) {
            lastThresholdAnchorControlled = threshold <= anchorFloor + 1.0e-12;
        }

        private boolean lastThresholdAnchorControlled() {
            return lastThresholdAnchorControlled;
        }

        private boolean floorParentThreshold() {
            return floorParentThreshold;
        }

        private double blend(double local, double parent, double count) {
            if (!Double.isFinite(local)) {
                return parent;
            }
            if (!Double.isFinite(parent)) {
                return local;
            }
            double weight = count / (count + shrinkageK);
            return weight * local + (1.0 - weight) * parent;
        }

        private DynamicThresholdStats parent(Integer key, boolean create) {
            DynamicThresholdStats stats = parentStats.get(key);
            if (stats == null && create) {
                stats = new DynamicThresholdStats(halfLifeSeconds, bucketSeconds);
                parentStats.put(key, stats);
            }
            return stats;
        }

        private DynamicThresholdStats leaf(Long key, boolean create) {
            DynamicThresholdStats stats = leafStats.get(key);
            if (stats == null && create) {
                stats = new DynamicThresholdStats(halfLifeSeconds, bucketSeconds);
                leafStats.put(key, stats);
            }
            return stats;
        }

        private Integer parentKey(Dataset dataset, int lineIndex) {
            if ("global".equals(groupMode)) {
                return null;
            }
            if ("level_keyword".equals(groupMode)) {
                return dataset.levelIds.values[lineIndex];
            }
            return dataset.componentIds.values[lineIndex];
        }

        private Integer parentKey(long leafKey) {
            if ("component_level".equals(groupMode) || "component_keyword".equals(groupMode)
                    || "level_keyword".equals(groupMode)) {
                return (int) (leafKey >> 32);
            }
            return null;
        }

        private Long leafKey(Dataset dataset, int lineIndex) {
            if ("global".equals(groupMode) || "component".equals(groupMode)) {
                return null;
            }
            int componentId = dataset.componentIds.values[lineIndex];
            int levelId = dataset.levelIds.values[lineIndex];
            int keyword = keywordClass(dataset.keywordScores.values[lineIndex]);
            if ("component_level".equals(groupMode)) {
                return componentLevelKey(componentId, levelId);
            }
            if ("component_keyword".equals(groupMode)) {
                return entityEventKey(componentId, keyword);
            }
            if ("level_keyword".equals(groupMode)) {
                return entityEventKey(levelId, keyword);
            }
            throw new IllegalArgumentException("unknown anchored dynamic threshold group " + groupMode);
        }
    }

    private static final class DynamicThresholdStats {
        private static final int MINIMUM_DEVIATION_SCORES = 10;

        private final double halfLifeSeconds;
        private final int bucketSeconds;
        private double warmupAnchor;
        private double warmupDynamic = Double.NaN;
        private double total;
        private double mean;
        private double m2;
        private double lowerTotal;
        private double lowerMean;
        private double lowerM2;
        private long lastBucketKey = Long.MIN_VALUE;

        private DynamicThresholdStats(double halfLifeSeconds, int bucketSeconds) {
            this.halfLifeSeconds = max(1.0, halfLifeSeconds);
            this.bucketSeconds = bucketSeconds;
        }

        private void setWarmupAnchor(double warmupAnchor) {
            this.warmupAnchor = Double.isFinite(warmupAnchor) ? max(0.0, warmupAnchor) : 0.0;
        }

        private double warmupAnchor() {
            return warmupAnchor;
        }

        private void captureWarmupDynamic(double zFactor) {
            warmupDynamic = rawDynamicThreshold(zFactor);
            if (!Double.isFinite(warmupDynamic)) {
                warmupDynamic = warmupAnchor;
            }
        }

        private void advanceTo(long bucketKey) {
            if (lastBucketKey == Long.MIN_VALUE) {
                lastBucketKey = bucketKey;
                return;
            }
            long deltaBuckets = bucketKey - lastBucketKey;
            if (deltaBuckets <= 0) {
                return;
            }
            double decay = Math.exp(-(deltaBuckets * (double) bucketSeconds) / halfLifeSeconds);
            total *= decay;
            m2 *= decay;
            lowerTotal *= decay;
            lowerM2 *= decay;
            lastBucketKey = bucketKey;
        }

        private void update(double score, long bucketKey) {
            advanceTo(bucketKey);
            if (!Double.isFinite(score)) {
                return;
            }
            double clean = max(0.0, score);
            if (total >= MINIMUM_DEVIATION_SCORES) {
                double gap = mean - clean;
                if (gap > 0.0) {
                    updateLower(gap);
                }
            }
            updatePrimary(clean);
        }

        private void updatePrimary(double value) {
            double previousTotal = total;
            total += 1.0;
            if (previousTotal <= 0.0) {
                mean = value;
                m2 = 0.0;
                return;
            }
            double delta = value - mean;
            mean += delta / total;
            m2 += delta * (value - mean);
        }

        private void updateLower(double value) {
            double previousTotal = lowerTotal;
            lowerTotal += 1.0;
            if (previousTotal <= 0.0) {
                lowerMean = value;
                lowerM2 = 0.0;
                return;
            }
            double delta = value - lowerMean;
            lowerMean += delta / lowerTotal;
            lowerM2 += delta * (value - lowerMean);
        }

        private double threshold(double zFactor, long bucketKey, boolean warmupRelativeDynamic) {
            advanceTo(bucketKey);
            if (total < MINIMUM_DEVIATION_SCORES) {
                return warmupAnchor;
            }
            double dynamic = rawDynamicThreshold(zFactor);
            if (warmupRelativeDynamic) {
                double baseline = max(warmupAnchor, Double.isFinite(warmupDynamic) ? warmupDynamic : dynamic);
                return warmupAnchor + max(0.0, dynamic - baseline);
            }
            return max(warmupAnchor, dynamic);
        }

        private double rawDynamicThreshold(double zFactor) {
            double deviation = primaryDeviation();
            double lower = lowerDeviation();
            if (lowerTotal >= MINIMUM_DEVIATION_SCORES && lower > 0.0) {
                deviation = min(deviation, Math.sqrt(2.0) * lower);
            }
            return mean + max(0.0, zFactor) * deviation;
        }

        private double count(long bucketKey) {
            advanceTo(bucketKey);
            return total;
        }

        private double primaryDeviation() {
            return total <= 1.0 ? 0.0 : Math.sqrt(max(0.0, m2 / total));
        }

        private double lowerDeviation() {
            return lowerTotal <= 1.0 ? 0.0 : Math.sqrt(max(0.0, lowerM2 / lowerTotal));
        }

        private long estimatedModelBytes() {
            return 96L;
        }
    }

    private interface StreamingLineConsumer {
        void accept(StreamLine line) throws IOException;
    }

    private static final class StreamingDatasetInfo {
        private final long sampledLines;
        private final int buckets;
        private final long anomalousLines;

        private StreamingDatasetInfo(long sampledLines, int buckets, long anomalousLines) {
            this.sampledLines = sampledLines;
            this.buckets = buckets;
            this.anomalousLines = anomalousLines;
        }
    }

    private static final class StreamLine {
        private final int bucketIndex;
        private final long bucketKey;
        private final int eventId;
        private final int entityId;
        private final int componentId;
        private final int levelId;
        private final int prevEventId;
        private final int parameterHash;
        private final int phraseHash;
        private final byte keywordScore;
        private final boolean label;

        private StreamLine(int bucketIndex, long bucketKey, int eventId, int entityId, int componentId, int levelId,
                int prevEventId, int parameterHash, int phraseHash, byte keywordScore, boolean label) {
            this.bucketIndex = bucketIndex;
            this.bucketKey = bucketKey;
            this.eventId = eventId;
            this.entityId = entityId;
            this.componentId = componentId;
            this.levelId = levelId;
            this.prevEventId = prevEventId;
            this.parameterHash = parameterHash;
            this.phraseHash = phraseHash;
            this.keywordScore = keywordScore;
            this.label = label;
        }
    }

    private static final class StreamingIds {
        private final MemoryLimits memoryLimits;
        private final Map<String, Integer> entityIds = new HashMap<>();
        private final Map<String, Integer> componentIds = new HashMap<>();
        private final Map<String, Integer> levelIds = new HashMap<>();
        private long droppedEntities;
        private long droppedComponents;
        private long droppedLevels;

        private StreamingIds(MemoryLimits memoryLimits) {
            this.memoryLimits = memoryLimits;
        }

        private int entityId(String value) {
            if (!memoryLimits.entityEnabled()) {
                return 0;
            }
            int id = intern(value, entityIds, memoryLimits.maxEntities);
            if (id == 0) {
                ++droppedEntities;
            }
            return id;
        }

        private int componentId(String value) {
            int id = intern(value, componentIds, memoryLimits.maxGroups);
            if (id == 0) {
                ++droppedComponents;
            }
            return id;
        }

        private int levelId(String value) {
            int id = intern(value, levelIds, memoryLimits.maxGroups);
            if (id == 0) {
                ++droppedLevels;
            }
            return id;
        }

        private static int intern(String value, Map<String, Integer> ids, int cap) {
            Integer existing = ids.get(value);
            if (existing != null) {
                return existing;
            }
            if (ids.size() >= cap) {
                return 0;
            }
            int id = ids.size() + 1;
            ids.put(value, id);
            return id;
        }

        private long estimatedModelBytes() {
            return 48L + stringIntMapBytes(entityIds) + stringIntMapBytes(componentIds) + stringIntMapBytes(levelIds);
        }

        private String capSummary() {
            return String.format(Locale.ROOT,
                    "entities=%d/%d components=%d/%d levels=%d/%d dropped_entities=%d dropped_components=%d dropped_levels=%d entity_mode=%s",
                    entityIds.size(), memoryLimits.maxEntities, componentIds.size(), memoryLimits.maxGroups,
                    levelIds.size(), memoryLimits.maxGroups, droppedEntities, droppedComponents, droppedLevels,
                    memoryLimits.entityMode);
        }
    }

    private static final class StreamingParseState {
        private final TemplateParser parser;
        private final StreamingIds ids;
        private final Map<Integer, Integer> lastEventByEntity = new HashMap<>();
        private final boolean trackEntityContext;

        private StreamingParseState(Config config) {
            ids = new StreamingIds(config.memoryLimits);
            trackEntityContext = config.memoryLimits.entityEnabled();
            parser = newTemplateParser(config);
        }

        private int previousEventId(int entityId) {
            if (!trackEntityContext || entityId == 0) {
                return -1;
            }
            return lastEventByEntity.getOrDefault(entityId, -1);
        }

        private void updateLastEvent(int entityId, int eventId) {
            if (trackEntityContext && entityId != 0) {
                lastEventByEntity.put(entityId, eventId);
            }
        }

        private long estimatedModelBytes() {
            return 64L + parser.estimatedModelBytes() + ids.estimatedModelBytes() + intIntMapBytes(lastEventByEntity);
        }

        private String capSummary() {
            return "parser[" + parser.capSummary() + "];ids[" + ids.capSummary() + "];lastEventByEntity="
                    + lastEventByEntity.size();
        }
    }

    private static final class StreamingScoreSummary {
        private final double maxScore;
        private final long[] bins;
        private long total;

        private StreamingScoreSummary(Config config) {
            maxScore = anchoredThresholdScore(config.rollingScoreMax, config);
            bins = new long[max(128, config.rollingBins)];
        }

        private void add(double score) {
            bins[bin(score)]++;
            ++total;
        }

        private int total() {
            return total > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) total;
        }

        private double quantile(double quantile) {
            if (total <= 0) {
                return 0.0;
            }
            long target = max(1L, (long) Math.ceil(max(0.0, min(1.0, quantile)) * total));
            long seen = 0;
            for (int i = 0; i < bins.length; i++) {
                seen += bins[i];
                if (seen >= target) {
                    return score(i);
                }
            }
            return maxScore;
        }

        private double localKneeStrength(double threshold, Config config) {
            int boundary = bin(threshold);
            int left = previousNonEmpty(boundary - 1);
            int right = nextNonEmpty(boundary + 1);
            if (left < 0 || right < 0) {
                return 0.0;
            }
            double gap = score(right) - score(left);
            return gap / max(1.0e-6, tailScale(config.anchorMinQuantile, config.anchorMaxQuantile));
        }

        private double bestTailKneeStrength(Config config) {
            if (total <= 1) {
                return 0.0;
            }
            int start = bin(quantile(config.anchorMinQuantile));
            int end = bin(quantile(config.anchorMaxQuantile));
            double scale = tailScale(config.anchorMinQuantile, config.anchorMaxQuantile);
            long seen = 0;
            int previous = -1;
            double best = 0.0;
            for (int i = 0; i <= min(end, bins.length - 1); i++) {
                seen += bins[i];
                if (bins[i] == 0) {
                    continue;
                }
                if (i >= start && previous >= 0) {
                    double q = seen / (double) total;
                    double support = max(0.02, (1.0 - q) / max(1.0e-6, 1.0 - config.anchorMinQuantile));
                    best = max(best, (score(i) - score(previous)) * support / max(1.0e-6, scale));
                }
                previous = i;
            }
            return best;
        }

        private double tailScale(double minQuantile, double maxQuantile) {
            if (total <= 1) {
                return 1.0;
            }
            int start = bin(quantile(minQuantile));
            int end = bin(quantile(maxQuantile));
            List<Double> gaps = new ArrayList<>();
            int previous = previousNonEmpty(start);
            for (int i = max(0, start + 1); i <= min(bins.length - 1, end); i++) {
                if (bins[i] == 0) {
                    continue;
                }
                if (previous >= 0) {
                    gaps.add(score(i) - score(previous));
                }
                previous = i;
            }
            if (gaps.isEmpty()) {
                return maxScore / bins.length;
            }
            Collections.sort(gaps);
            return max(1.0e-6, gaps.get(gaps.size() / 2));
        }

        private int previousNonEmpty(int index) {
            for (int i = min(index, bins.length - 1); i >= 0; i--) {
                if (bins[i] > 0) {
                    return i;
                }
            }
            return -1;
        }

        private int nextNonEmpty(int index) {
            for (int i = max(index, 0); i < bins.length; i++) {
                if (bins[i] > 0) {
                    return i;
                }
            }
            return -1;
        }

        private int bin(double score) {
            double clean = Double.isFinite(score) ? max(0.0, min(maxScore, score)) : maxScore;
            return min(max(0, (int) Math.floor(clean / maxScore * (bins.length - 1))), bins.length - 1);
        }

        private double score(int bin) {
            return bin * maxScore / (bins.length - 1.0);
        }
    }

    private static final class StreamingAnchoredSelection {
        private final String scoreMode;
        private final double anchorQuantile;
        private final double anchorThreshold;
        private final double quality;
        private final double kneeStrength;
        private final int warmupLines;

        private StreamingAnchoredSelection(String scoreMode, double anchorQuantile, double anchorThreshold,
                double quality, double kneeStrength, int warmupLines) {
            this.scoreMode = scoreMode;
            this.anchorQuantile = anchorQuantile;
            this.anchorThreshold = anchorThreshold;
            this.quality = quality;
            this.kneeStrength = kneeStrength;
            this.warmupLines = warmupLines;
        }
    }

    private static final class StreamingAnchoredThreshold {
        private final DynamicThresholdStats global;
        private final double zFactor;
        private final boolean warmupRelativeDynamic;
        private boolean lastThresholdAnchorControlled = true;

        private StreamingAnchoredThreshold(Config config) {
            global = new DynamicThresholdStats(daysToSeconds(config.anchoredThresholdHalfLifeDays),
                    config.bucketSeconds);
            zFactor = config.zFactor;
            warmupRelativeDynamic = config.useWarmupRelativeDynamicThreshold();
        }

        private void setWarmupAnchor(double threshold) {
            global.setWarmupAnchor(threshold);
        }

        private void captureWarmupDynamic() {
            global.captureWarmupDynamic(zFactor);
        }

        private void advanceTo(long bucketKey) {
            global.advanceTo(bucketKey);
        }

        private void update(double score, long bucketKey) {
            global.update(score, bucketKey);
        }

        private double threshold(long bucketKey) {
            double threshold = global.threshold(zFactor, bucketKey, warmupRelativeDynamic);
            lastThresholdAnchorControlled = threshold <= global.warmupAnchor() + 1.0e-12;
            return threshold;
        }

        private boolean lastThresholdAnchorControlled() {
            return lastThresholdAnchorControlled;
        }

        private long estimatedModelBytes() {
            return 64L + global.estimatedModelBytes();
        }
    }

    private static final class StreamingProfile {
        private final String datasetName;
        private final int startBucket;
        private final int endBucket;
        private final int intervalLines;
        private final long startNanos = System.nanoTime();
        private long lines;
        private long lastReportLines;
        private long countAdvanceNanos;
        private long thresholdAdvanceNanos;
        private long scoreNanos;
        private long thresholdNanos;
        private long thresholdUpdateNanos;
        private long countUpdateNanos;
        private long predictions;
        private long anchorControlled;
        private long dynamicControlled;
        private double scoreSum;
        private double thresholdSum;
        private final Map<Integer, Long> predictedTemplates = new HashMap<>();

        private StreamingProfile(String datasetName, int startBucket, int endBucket, Config config) {
            this.datasetName = datasetName;
            this.startBucket = startBucket;
            this.endBucket = endBucket;
            intervalLines = max(1, config.progressInterval);
        }

        private void record(StreamLine line, double score, double threshold, boolean prediction,
                boolean anchorControlledDecision) {
            ++lines;
            scoreSum += score;
            thresholdSum += threshold;
            if (anchorControlledDecision) {
                ++anchorControlled;
            } else {
                ++dynamicControlled;
            }
            if (prediction) {
                ++predictions;
                predictedTemplates.put(line.eventId, predictedTemplates.getOrDefault(line.eventId, 0L) + 1L);
            }
        }

        private void maybeReport(int bucketIndex, OnlineBaselineState online) {
            if (lines - lastReportLines >= intervalLines) {
                report("progress", bucketIndex, online, null, null);
                lastReportLines = lines;
            }
        }

        private void report(String phase, int bucketIndex, OnlineBaselineState online,
                StreamingAnchoredThreshold thresholds, StreamingParseState parseState) {
            double elapsed = secondsSince(startNanos);
            long measured = countAdvanceNanos + thresholdAdvanceNanos + scoreNanos + thresholdNanos
                    + thresholdUpdateNanos + countUpdateNanos;
            long modelBytes = online.estimatedModelBytes()
                    + (thresholds == null ? 0L : thresholds.estimatedModelBytes())
                    + (parseState == null ? 0L : parseState.estimatedModelBytes());
            String capState = parseState == null ? online.capSummary()
                    : online.capSummary() + ";" + parseState.capSummary();
            System.out.printf(Locale.ROOT,
                    "streaming_profile_eval dataset=%s phase=%s buckets=%d/%d lines=%d elapsed=%.3f lines_per_sec=%.1f count_advance=%.3f threshold_advance=%.3f score=%.3f threshold=%.3f threshold_update=%.3f count_update=%.3f measured=%.3f prediction_rate=%.6f anchor_controlled=%.6f dynamic_controlled=%.6f avg_score=%.6f avg_threshold=%.6f used_heap_mib=%.2f model_estimate_mib=%.4f top_pred_templates=%s count_state=%s cap_state=%s%n",
                    datasetName, phase, max(0, bucketIndex - startBucket + 1), max(0, endBucket - startBucket), lines,
                    elapsed, lines / max(1.0e-9, elapsed), seconds(countAdvanceNanos), seconds(thresholdAdvanceNanos),
                    seconds(scoreNanos), seconds(thresholdNanos), seconds(thresholdUpdateNanos),
                    seconds(countUpdateNanos), seconds(measured), predictions / max(1.0, (double) lines),
                    anchorControlled / max(1.0, (double) lines), dynamicControlled / max(1.0, (double) lines),
                    scoreSum / max(1.0, (double) lines), thresholdSum / max(1.0, (double) lines), usedHeapMiB(),
                    bytesToMiB(modelBytes), topPredictedTemplates(), online.entrySummary(), capState);
        }

        private String topPredictedTemplates() {
            if (predictedTemplates.isEmpty()) {
                return "-";
            }
            List<Map.Entry<Integer, Long>> entries = new ArrayList<>(predictedTemplates.entrySet());
            entries.sort((a, b) -> Long.compare(b.getValue(), a.getValue()));
            StringBuilder builder = new StringBuilder();
            int limit = min(5, entries.size());
            for (int i = 0; i < limit; i++) {
                if (i > 0) {
                    builder.append(';');
                }
                builder.append(entries.get(i).getKey()).append(':').append(entries.get(i).getValue());
            }
            return builder.toString();
        }
    }

    private static double daysToSeconds(double days) {
        return max(1.0, days * 24.0 * 60.0 * 60.0);
    }

    private static double hoursToSeconds(double hours) {
        return max(1.0, hours * 60.0 * 60.0);
    }

    private static long entityEventKey(int entityId, int eventId) {
        return ((long) entityId << 32) ^ (eventId & 0xffffffffL);
    }

    private static long componentLevelKey(int componentId, int levelId) {
        return ((long) componentId << 32) ^ (levelId & 0xffffffffL);
    }

    private static long componentLevelEventKey(int componentId, int levelId, int eventId) {
        long key = 1469598103934665603L;
        key = (key ^ componentId) * 1099511628211L;
        key = (key ^ levelId) * 1099511628211L;
        key = (key ^ eventId) * 1099511628211L;
        return key;
    }

    private static long semanticSignatureKey(int componentId, int levelId, int phraseHash) {
        long key = 1469598103934665603L;
        key = (key ^ componentId) * 1099511628211L;
        key = (key ^ levelId) * 1099511628211L;
        key = (key ^ phraseHash) * 1099511628211L;
        return key;
    }

    private static long transitionKey(int prevEventId, int eventId) {
        return (((long) prevEventId + 1L) << 32) ^ (eventId & 0xffffffffL);
    }

    private static final class Bucket {
        private final long bucketKey;
        private final Map<Integer, Integer> eventCounts = new HashMap<>();
        private Map<Integer, Integer> entityCounts;
        private Map<Long, Integer> entityEventCounts;
        private final Map<Integer, Byte> maxKeywordByEvent = new HashMap<>();
        private int totalCount;
        private int anomalousCount;

        private Bucket(long bucketKey) {
            this.bucketKey = bucketKey;
        }

        private void add(int eventId, int entityId, boolean anomalous, byte keywordScore,
                boolean trackEntityBucketCounts) {
            eventCounts.put(eventId, eventCounts.getOrDefault(eventId, 0) + 1);
            if (trackEntityBucketCounts) {
                if (entityCounts == null) {
                    entityCounts = new HashMap<>();
                    entityEventCounts = new HashMap<>();
                }
                entityCounts.put(entityId, entityCounts.getOrDefault(entityId, 0) + 1);
                long key = entityEventKey(entityId, eventId);
                entityEventCounts.put(key, entityEventCounts.getOrDefault(key, 0) + 1);
            }
            byte previous = maxKeywordByEvent.getOrDefault(eventId, (byte) 0);
            if (keywordScore > previous) {
                maxKeywordByEvent.put(eventId, keywordScore);
            }
            ++totalCount;
            if (anomalous) {
                ++anomalousCount;
            }
        }
    }

    private static ParsedLine parseBglLikeLine(String line) {
        String[] parts = line.split("\\s+", 10);
        if (parts.length < 9) {
            return null;
        }
        long epochSeconds;
        try {
            epochSeconds = Long.parseLong(parts[1]);
        } catch (NumberFormatException e) {
            return null;
        }
        boolean label = !parts[0].startsWith("-");
        String entity = parts[3];
        String component = normalizeComponent(parts[7]);
        String level = normalizeComponent(parts[8]);
        String message = parts[8] + (parts.length >= 10 ? " " + parts[9] : "");
        return new ParsedLine(epochSeconds, entity, component, level, label, message, keywordScore(message));
    }

    private static final class ParsedLine {
        private final long epochSeconds;
        private final String entity;
        private final String component;
        private final String level;
        private final boolean label;
        private final String message;
        private final byte keywordScore;

        private ParsedLine(long epochSeconds, String entity, String component, String level, boolean label,
                String message, byte keywordScore) {
            this.epochSeconds = epochSeconds;
            this.entity = entity;
            this.component = component;
            this.level = level;
            this.label = label;
            this.message = message;
            this.keywordScore = keywordScore;
        }
    }

    private static byte keywordScore(String message) {
        String lower = message.toLowerCase(Locale.ROOT);
        if (containsAny(lower, "fatal", "panic", "kernel panic", "oops", "segfault", "assertion", "corrupt")) {
            return 8;
        }
        if (containsAny(lower, "critical", "crit", "emerg", "alert", "exception", "abort", "timeout", "timed out",
                "denied", "invalid", "unavailable", "unreachable", " reset ")) {
            return 4;
        }
        if (containsAny(lower, "corrected", "recovered", "recovery")) {
            return 0;
        }
        if (containsAny(lower, "error", "failed", "failure", " fail ", "unable", "no such", "lost", "warning",
                "warn")) {
            return 1;
        }
        return 0;
    }

    private static boolean containsAny(String value, String... needles) {
        for (String needle : needles) {
            if (value.contains(needle)) {
                return true;
            }
        }
        return false;
    }

    private static String normalizeComponent(String component) {
        String lower = component.toLowerCase(Locale.ROOT);
        StringBuilder builder = new StringBuilder(lower.length());
        boolean previousZero = false;
        for (int i = 0; i < lower.length(); i++) {
            char value = lower.charAt(i);
            if (value >= '0' && value <= '9') {
                if (!previousZero) {
                    builder.append('0');
                    previousZero = true;
                }
            } else if (Character.isLetter(value) || value == '/' || value == '.' || value == '_' || value == '-') {
                builder.append(value);
                previousZero = false;
            }
        }
        return builder.length() == 0 ? "unknown" : builder.toString();
    }

    private static int parameterHash(String message) {
        String[] tokens = message.split("\\s+");
        int hash = 0x811c9dc5;
        int kept = 0;
        for (String raw : tokens) {
            String token = cleanParameterToken(raw);
            if (token.isEmpty() || !looksLikeParameterValue(token)) {
                continue;
            }
            for (int i = 0; i < token.length(); i++) {
                hash ^= token.charAt(i);
                hash *= 0x01000193;
            }
            hash ^= 0x9e3779b9 + kept;
            hash *= 0x01000193;
            ++kept;
            if (kept >= 4) {
                break;
            }
        }
        return kept == 0 ? 0 : hash;
    }

    private static int semanticPhraseHash(String message) {
        String[] rawTokens = message.split("\\s+");
        int hash = 0x811c9dc5;
        int kept = 0;
        for (String raw : rawTokens) {
            if (raw.indexOf('/') >= 0 || raw.indexOf('.') >= 0 || containsDigit(raw)) {
                continue;
            }
            String token = semanticWordToken(raw);
            if (token.isEmpty() || isSemanticStopToken(token)) {
                continue;
            }
            for (int i = 0; i < token.length(); i++) {
                hash ^= token.charAt(i);
                hash *= 0x01000193;
            }
            hash ^= 0x9e3779b9 + kept;
            hash *= 0x01000193;
            ++kept;
            if (kept >= 10) {
                break;
            }
        }
        return kept == 0 ? 0 : hash;
    }

    private static String semanticWordToken(String raw) {
        StringBuilder builder = new StringBuilder(raw.length());
        for (int i = 0; i < raw.length(); i++) {
            char value = Character.toLowerCase(raw.charAt(i));
            if (value >= 'a' && value <= 'z') {
                builder.append(value);
            }
        }
        return builder.length() < 2 ? "" : builder.toString();
    }

    private static boolean containsDigit(String value) {
        for (int i = 0; i < value.length(); i++) {
            if (Character.isDigit(value.charAt(i))) {
                return true;
            }
        }
        return false;
    }

    private static boolean isSemanticStopToken(String token) {
        switch (token) {
        case "fatal":
        case "failure":
        case "failed":
        case "error":
        case "warning":
        case "warn":
        case "info":
        case "the":
        case "and":
        case "or":
        case "to":
        case "from":
        case "on":
        case "for":
        case "of":
        case "in":
        case "with":
        case "after":
        case "before":
        case "due":
        case "mode":
        case "node":
        case "card":
        case "link":
            return true;
        default:
            return false;
        }
    }

    private static String cleanParameterToken(String raw) {
        int start = 0;
        int end = raw.length() - 1;
        while (start <= end && !isParameterBoundaryCharacter(raw.charAt(start))) {
            ++start;
        }
        while (end >= start && !isParameterBoundaryCharacter(raw.charAt(end))) {
            --end;
        }
        return start > end ? "" : raw.substring(start, end + 1).toLowerCase(Locale.ROOT);
    }

    private static boolean isParameterBoundaryCharacter(char value) {
        return Character.isLetterOrDigit(value) || value == '.' || value == ':' || value == '/' || value == '-'
                || value == '_' || value == '@';
    }

    private static boolean looksLikeParameterValue(String token) {
        boolean hasDigit = false;
        boolean hasLetter = false;
        boolean hasStructural = false;
        for (int i = 0; i < token.length(); i++) {
            char value = token.charAt(i);
            hasDigit |= Character.isDigit(value);
            hasLetter |= Character.isLetter(value);
            hasStructural |= value == '.' || value == ':' || value == '/' || value == '-' || value == '_'
                    || value == '@';
        }
        if (!hasDigit) {
            return false;
        }
        if (!hasLetter && !hasStructural) {
            return false;
        }
        return token.length() >= 2;
    }

    private static String normalize(String message) {
        String lower = message.toLowerCase(Locale.ROOT);
        StringBuilder builder = new StringBuilder(lower.length());
        boolean previousZero = false;
        for (int i = 0; i < lower.length(); i++) {
            char value = lower.charAt(i);
            if (value >= '0' && value <= '9') {
                if (!previousZero) {
                    builder.append('0');
                    previousZero = true;
                }
            } else {
                builder.append(value);
                previousZero = false;
            }
        }
        return builder.toString();
    }

    private static int positiveHash(String value) {
        int hash = 0x811c9dc5;
        for (int i = 0; i < value.length(); i++) {
            hash ^= value.charAt(i);
            hash *= 0x01000193;
        }
        return hash & 0x7fffffff;
    }

    private interface TemplateParser {
        int parse(String message);

        int templateCount();

        String[] templates();

        long estimatedModelBytes();

        String capSummary();
    }

    private static final class SimpleDrainParser implements TemplateParser {
        private final double similarityThreshold;
        private final int maxTemplates;
        private final long maxParserBytes;
        private final Map<Integer, List<DrainCluster>> clustersByLength = new HashMap<>();
        private int nextId;
        private long totalParses;
        private long droppedNewTemplates;

        private SimpleDrainParser(double similarityThreshold, int maxTemplates, long maxParserBytes) {
            this.similarityThreshold = similarityThreshold;
            this.maxTemplates = max(1, maxTemplates);
            this.maxParserBytes = maxParserBytes;
        }

        @Override
        public int parse(String message) {
            ++totalParses;
            List<String> tokens = tokenize(message);
            List<DrainCluster> candidates = clustersByLength.get(tokens.size());
            DrainCluster best = null;
            double bestSimilarity = -1.0;
            if (candidates != null) {
                for (DrainCluster candidate : candidates) {
                    double similarity = similarity(tokens, candidate.template);
                    if (similarity > bestSimilarity) {
                        bestSimilarity = similarity;
                        best = candidate;
                    }
                }
            }
            if (best != null && bestSimilarity >= similarityThreshold) {
                updateTemplate(best.template, tokens);
                return best.id;
            }
            if (nextId >= maxTemplates || estimatedModelBytes() >= maxParserBytes) {
                ++droppedNewTemplates;
                return best == null ? 0 : best.id;
            }
            DrainCluster cluster = new DrainCluster(nextId++, tokens);
            clustersByLength.computeIfAbsent(tokens.size(), ignored -> new ArrayList<>()).add(cluster);
            return cluster.id;
        }

        @Override
        public int templateCount() {
            return nextId;
        }

        @Override
        public String[] templates() {
            String[] result = new String[nextId];
            for (List<DrainCluster> clusters : clustersByLength.values()) {
                for (DrainCluster cluster : clusters) {
                    result[cluster.id] = String.join(" ", cluster.template);
                }
            }
            for (int i = 0; i < result.length; i++) {
                if (result[i] == null) {
                    result[i] = "<unknown>";
                }
            }
            return result;
        }

        @Override
        public long estimatedModelBytes() {
            long bytes = 96L + hashMapBytes(clustersByLength.size());
            for (List<DrainCluster> clusters : clustersByLength.values()) {
                bytes += 32L + 8L * clusters.size();
                for (DrainCluster cluster : clusters) {
                    bytes += cluster.estimatedModelBytes();
                }
            }
            return bytes;
        }

        @Override
        public String capSummary() {
            return String.format(Locale.ROOT,
                    "templates=%d/%d parser_mib=%.4f/%.4f dropped_new_templates=%d unknown_template_rate=%.6f", nextId,
                    maxTemplates, bytesToMiB(estimatedModelBytes()), bytesToMiB(maxParserBytes), droppedNewTemplates,
                    droppedNewTemplates / max(1.0, (double) totalParses));
        }

        private static List<String> tokenize(String message) {
            if (message == null || message.trim().isEmpty()) {
                return Collections.singletonList("<empty>");
            }
            String[] rawTokens = message.trim().split("\\s+");
            List<String> tokens = new ArrayList<>(rawTokens.length);
            for (String rawToken : rawTokens) {
                String token = normalizeToken(rawToken);
                if (!token.isEmpty()) {
                    tokens.add(token);
                }
            }
            return tokens.isEmpty() ? Collections.singletonList("<empty>") : tokens;
        }

        private static String normalizeToken(String rawToken) {
            for (int i = 0; i < rawToken.length(); i++) {
                if (Character.isDigit(rawToken.charAt(i))) {
                    return "<*>";
                }
            }
            int start = 0;
            int end = rawToken.length() - 1;
            while (start <= end && !isTemplateCharacter(rawToken.charAt(start))) {
                ++start;
            }
            while (end >= start && !isTemplateCharacter(rawToken.charAt(end))) {
                --end;
            }
            return start > end ? "" : rawToken.substring(start, end + 1);
        }

        private static boolean isTemplateCharacter(char value) {
            return value >= 'a' && value <= 'z' || value == '<' || value == '>' || value == '*';
        }

        private static double similarity(List<String> tokens, List<String> template) {
            if (tokens.size() != template.size()) {
                return 0.0;
            }
            int exact = 0;
            int comparable = 0;
            for (int i = 0; i < tokens.size(); i++) {
                String templateToken = template.get(i);
                if (!"<*>".equals(templateToken)) {
                    ++comparable;
                    if (templateToken.equals(tokens.get(i))) {
                        ++exact;
                    }
                }
            }
            return comparable == 0 ? 1.0 : (double) exact / comparable;
        }

        private static void updateTemplate(List<String> template, List<String> tokens) {
            for (int i = 0; i < template.size(); i++) {
                if (!template.get(i).equals(tokens.get(i))) {
                    template.set(i, "<*>");
                }
            }
        }
    }

    private static final class FixedDepthDrainParser implements TemplateParser {
        private final double similarityThreshold;
        private final int maxDepth;
        private final int maxChildren;
        private final int maxTemplates;
        private final int maxNodes;
        private final long maxParserBytes;
        private final DrainNode root = new DrainNode();
        private int nextId;
        private int nodeCount = 1;
        private long totalParses;
        private long droppedNewTemplates;
        private long droppedNewNodes;

        private FixedDepthDrainParser(double similarityThreshold, int maxDepth, int maxChildren, int maxTemplates,
                int maxNodes, long maxParserBytes) {
            this.similarityThreshold = similarityThreshold;
            this.maxDepth = max(3, maxDepth);
            this.maxChildren = max(2, maxChildren);
            this.maxTemplates = max(1, maxTemplates);
            this.maxNodes = max(1, maxNodes);
            this.maxParserBytes = maxParserBytes;
        }

        @Override
        public int parse(String message) {
            ++totalParses;
            List<String> tokens = SimpleDrainParser.tokenize(message);
            DrainNode leaf = findLeaf(tokens, false);
            DrainCluster best = bestCluster(leaf.clusters, tokens);
            if (best != null && SimpleDrainParser.similarity(tokens, best.template) >= similarityThreshold) {
                SimpleDrainParser.updateTemplate(best.template, tokens);
                return best.id;
            }
            if (nextId >= maxTemplates || estimatedModelBytes() >= maxParserBytes) {
                ++droppedNewTemplates;
                return best == null ? 0 : best.id;
            }
            DrainCluster cluster = new DrainCluster(nextId++, tokens);
            findLeaf(tokens, true).clusters.add(cluster);
            return cluster.id;
        }

        @Override
        public int templateCount() {
            return nextId;
        }

        @Override
        public String[] templates() {
            String[] result = new String[nextId];
            collectTemplates(root, result);
            for (int i = 0; i < result.length; i++) {
                if (result[i] == null) {
                    result[i] = "<unknown>";
                }
            }
            return result;
        }

        @Override
        public long estimatedModelBytes() {
            return 96L + root.estimatedModelBytes();
        }

        @Override
        public String capSummary() {
            return String.format(Locale.ROOT,
                    "templates=%d/%d nodes=%d/%d parser_mib=%.4f/%.4f dropped_new_templates=%d dropped_new_nodes=%d unknown_template_rate=%.6f",
                    nextId, maxTemplates, nodeCount, maxNodes, bytesToMiB(estimatedModelBytes()),
                    bytesToMiB(maxParserBytes), droppedNewTemplates, droppedNewNodes,
                    droppedNewTemplates / max(1.0, (double) totalParses));
        }

        private DrainNode findLeaf(List<String> tokens, boolean create) {
            DrainNode node = child(root, Integer.toString(tokens.size()), create);
            if (node == null) {
                return root;
            }
            int tokenDepth = min(tokens.size(), maxDepth - 2);
            for (int i = 0; i < tokenDepth; i++) {
                String token = tokens.get(i);
                DrainNode next = node.children.get(token);
                if (next == null) {
                    next = node.children.get("<*>");
                }
                if (next == null) {
                    if (!create) {
                        return node;
                    }
                    next = child(node, token, true);
                    if (next == null) {
                        return node;
                    }
                }
                node = next;
            }
            return node;
        }

        private DrainNode child(DrainNode node, String token, boolean create) {
            DrainNode existing = node.children.get(token);
            if (existing != null || !create) {
                return existing;
            }
            String key = token;
            if (!"<*>".equals(token) && node.children.size() >= maxChildren) {
                key = "<*>";
                existing = node.children.get(key);
                if (existing != null) {
                    return existing;
                }
            }
            if (nodeCount >= maxNodes || estimatedModelBytes() >= maxParserBytes) {
                ++droppedNewNodes;
                return null;
            }
            DrainNode created = new DrainNode();
            node.children.put(key, created);
            ++nodeCount;
            return created;
        }

        private static DrainCluster bestCluster(List<DrainCluster> candidates, List<String> tokens) {
            DrainCluster best = null;
            double bestSimilarity = -1.0;
            for (DrainCluster candidate : candidates) {
                double similarity = SimpleDrainParser.similarity(tokens, candidate.template);
                if (similarity > bestSimilarity) {
                    bestSimilarity = similarity;
                    best = candidate;
                }
            }
            return best;
        }

        private static void collectTemplates(DrainNode node, String[] result) {
            for (DrainCluster cluster : node.clusters) {
                result[cluster.id] = String.join(" ", cluster.template);
            }
            for (DrainNode child : node.children.values()) {
                collectTemplates(child, result);
            }
        }
    }

    private static final class DrainNode {
        private final Map<String, DrainNode> children = new HashMap<>();
        private final List<DrainCluster> clusters = new ArrayList<>();

        private long estimatedModelBytes() {
            long bytes = 80L + hashMapBytes(children.size()) + 32L + 8L * clusters.size();
            for (String key : children.keySet()) {
                bytes += stringBytes(key);
            }
            for (DrainCluster cluster : clusters) {
                bytes += cluster.estimatedModelBytes();
            }
            for (DrainNode child : children.values()) {
                bytes += child.estimatedModelBytes();
            }
            return bytes;
        }
    }

    private static final class DrainCluster {
        private final int id;
        private final List<String> template;

        private DrainCluster(int id, List<String> template) {
            this.id = id;
            this.template = new ArrayList<>(template);
        }

        private long estimatedModelBytes() {
            long bytes = 56L + 32L + 8L * template.size();
            for (String token : template) {
                bytes += stringBytes(token);
            }
            return bytes;
        }
    }

    private static final class IntList {
        private int[] values = new int[1024];
        private int size;

        private void add(int value) {
            ensure(size + 1);
            values[size++] = value;
        }

        private void ensure(int capacity) {
            if (capacity > values.length) {
                int next = values.length;
                while (next < capacity) {
                    next *= 2;
                }
                values = java.util.Arrays.copyOf(values, next);
            }
        }
    }

    private static final class ByteList {
        private byte[] values = new byte[1024];
        private int size;

        private void add(byte value) {
            ensure(size + 1);
            values[size++] = value;
        }

        private int sum() {
            int sum = 0;
            for (int i = 0; i < size; i++) {
                sum += values[i];
            }
            return sum;
        }

        private void ensure(int capacity) {
            if (capacity > values.length) {
                int next = values.length;
                while (next < capacity) {
                    next *= 2;
                }
                values = java.util.Arrays.copyOf(values, next);
            }
        }
    }

    private static final class TopK {
        private final double[] scores;
        private final boolean[] labels;
        private int size;

        private TopK(int maxK) {
            this.scores = new double[maxK];
            this.labels = new boolean[maxK];
        }

        private void add(double score, boolean label) {
            if (size < scores.length) {
                scores[size] = score;
                labels[size] = label;
                ++size;
                return;
            }
            int worst = 0;
            for (int i = 1; i < size; i++) {
                if (scores[i] < scores[worst]) {
                    worst = i;
                }
            }
            if (score > scores[worst]) {
                scores[worst] = score;
                labels[worst] = label;
            }
        }

        private void sortDescending() {
            for (int i = 0; i < size; i++) {
                for (int j = i + 1; j < size; j++) {
                    if (scores[j] > scores[i]) {
                        double score = scores[i];
                        scores[i] = scores[j];
                        scores[j] = score;
                        boolean label = labels[i];
                        labels[i] = labels[j];
                        labels[j] = label;
                    }
                }
            }
        }
    }

    private static final class ScoreBucket {
        private final int startLine;
        private final double[] scores;
        private final int[] classes;

        private ScoreBucket(int startLine, double[] scores, int[] classes) {
            this.startLine = startLine;
            this.scores = scores;
            this.classes = classes;
        }

        private ScoreBucket truncate(int size) {
            if (size == scores.length) {
                return this;
            }
            return new ScoreBucket(startLine, java.util.Arrays.copyOf(scores, size),
                    java.util.Arrays.copyOf(classes, size));
        }
    }

    private static final class ScoreBlock {
        private final double[] scores;
        private final int[] classes;
        private final int[] lineIndexes;

        private ScoreBlock(double[] scores, int[] classes, int[] lineIndexes) {
            this.scores = scores;
            this.classes = classes;
            this.lineIndexes = lineIndexes;
        }
    }

    private static final class TailCalibration {
        private final double[][] sortedScoresByClass;
        private final int totalCount;

        private TailCalibration(double[][] sortedScoresByClass, int totalCount) {
            this.sortedScoresByClass = sortedScoresByClass;
            this.totalCount = totalCount;
        }

        private double[] scoresForClass(int scoreClass) {
            if (scoreClass < sortedScoresByClass.length && sortedScoresByClass[scoreClass].length > 0) {
                return sortedScoresByClass[scoreClass];
            }
            return sortedScoresByClass[0];
        }
    }

    private static final class RollingLineCalibrator {
        private final java.util.ArrayDeque<ScoreBucket> buckets = new java.util.ArrayDeque<>();
        private final RollingHistogram[] histograms;
        private final boolean conditional;

        private RollingLineCalibrator(double maxScore, int bins, boolean conditional) {
            this.conditional = conditional;
            int count = conditional ? 4 : 1;
            histograms = new RollingHistogram[count];
            for (int i = 0; i < histograms.length; i++) {
                histograms[i] = new RollingHistogram(maxScore, bins);
            }
        }

        private void add(ScoreBucket bucket) {
            buckets.addLast(bucket);
            for (int i = 0; i < bucket.scores.length; i++) {
                histograms[histogramIndex(bucket.classes[i])].add(bucket.scores[i]);
            }
        }

        private void removeOldest() {
            ScoreBucket bucket = buckets.removeFirst();
            for (int i = 0; i < bucket.scores.length; i++) {
                histograms[histogramIndex(bucket.classes[i])].remove(bucket.scores[i]);
            }
        }

        private double pValue(double score, int scoreClass) {
            RollingHistogram histogram = histograms[histogramIndex(scoreClass)];
            if (histogram.total == 0 && conditional) {
                histogram = histograms[0];
            }
            return histogram.upperTailPValue(score);
        }

        private int histogramIndex(int scoreClass) {
            return conditional ? min(max(scoreClass, 0), histograms.length - 1) : 0;
        }

        private int size() {
            return buckets.size();
        }

        private int totalCount() {
            int total = 0;
            for (RollingHistogram histogram : histograms) {
                total += histogram.total;
            }
            return total;
        }
    }

    private static final class RollingHistogram {
        private final double maxScore;
        private final int[] counts;
        private int total;

        private RollingHistogram(double maxScore, int bins) {
            this.maxScore = maxScore;
            this.counts = new int[bins];
        }

        private void add(double score) {
            ++counts[bin(score)];
            ++total;
        }

        private void remove(double score) {
            int bin = bin(score);
            if (counts[bin] > 0) {
                --counts[bin];
                --total;
            }
        }

        private double upperTailPValue(double score) {
            if (total == 0) {
                return 1.0;
            }
            int start = bin(score);
            int tail = 0;
            for (int i = start; i < counts.length; i++) {
                tail += counts[i];
            }
            return (tail + 1.0) / (total + 1.0);
        }

        private int bin(double score) {
            double clean = Double.isFinite(score) ? max(0.0, min(maxScore, score)) : maxScore;
            int bin = (int) Math.floor(clean / maxScore * (counts.length - 1));
            return min(max(bin, 0), counts.length - 1);
        }
    }

    private static final class RollingTemplateBaseline {
        private final java.util.ArrayDeque<Integer> bucketIndexes = new java.util.ArrayDeque<>();
        private final int[] counts;
        private int total;

        private RollingTemplateBaseline(int templateCount) {
            counts = new int[max(1, templateCount)];
        }

        private void add(int bucketIndex, Bucket bucket) {
            bucketIndexes.addLast(bucketIndex);
            total += bucket.totalCount;
            for (Map.Entry<Integer, Integer> entry : bucket.eventCounts.entrySet()) {
                counts[entry.getKey()] += entry.getValue();
            }
        }

        private void removeOldest(Dataset dataset) {
            Bucket bucket = dataset.buckets.get(bucketIndexes.removeFirst());
            total -= bucket.totalCount;
            for (Map.Entry<Integer, Integer> entry : bucket.eventCounts.entrySet()) {
                counts[entry.getKey()] -= entry.getValue();
            }
        }

        private int size() {
            return bucketIndexes.size();
        }

        private double poissonUpperTail(int bucketTotal, int eventId, int observed) {
            if (observed <= 0 || total <= 0) {
                return 1.0;
            }
            double probability = (counts[eventId] + 1.0) / (total + counts.length);
            double lambda = max(1.0e-12, bucketTotal * probability);
            return poissonSurvival(lambda, observed);
        }
    }

    private static double poissonSurvival(double lambda, int observed) {
        if (observed <= 0) {
            return 1.0;
        }
        if (lambda > 100.0) {
            double z = (observed - 0.5 - lambda) / Math.sqrt(lambda);
            return normalUpperTail(z);
        }
        double term = Math.exp(-lambda);
        double cumulative = term;
        for (int k = 1; k < observed; k++) {
            term *= lambda / k;
            cumulative += term;
            if (term < 1.0e-15) {
                break;
            }
        }
        return max(0.0, min(1.0, 1.0 - cumulative));
    }

    private static double normalUpperTail(double z) {
        return 0.5 * erfcApprox(z / Math.sqrt(2.0));
    }

    private static double erfcApprox(double x) {
        double z = Math.abs(x);
        double t = 1.0 / (1.0 + 0.5 * z);
        double ans = t * Math
                .exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t
                        * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? ans : 2.0 - ans;
    }

    private static final class ThresholdChoice {
        private final double threshold;
        private final double validationPrecision;
        private final double validationRecall;
        private final double validationF1;

        private ThresholdChoice(double threshold, double validationPrecision, double validationRecall,
                double validationF1) {
            this.threshold = threshold;
            this.validationPrecision = validationPrecision;
            this.validationRecall = validationRecall;
            this.validationF1 = validationF1;
        }
    }

    private static final class Metrics {
        private long tp;
        private long fp;
        private long fn;
        private long tn;

        private Metrics(long tp, long fp, long fn, long tn) {
            this.tp = tp;
            this.fp = fp;
            this.fn = fn;
            this.tn = tn;
        }

        private Metrics add(boolean label, boolean prediction) {
            if (label && prediction) {
                ++tp;
            } else if (prediction) {
                ++fp;
            } else if (label) {
                ++fn;
            } else {
                ++tn;
            }
            return this;
        }

        private double precision() {
            return tp + fp == 0 ? 0.0 : tp / (double) (tp + fp);
        }

        private double recall() {
            return tp + fn == 0 ? 0.0 : tp / (double) (tp + fn);
        }

        private double f1() {
            double p = precision();
            double r = recall();
            return p + r == 0 ? 0.0 : 2.0 * p * r / (p + r);
        }
    }

    private static final class PredictionRecord {
        private final String datasetName;
        private final double quantile;
        private final double threshold;
        private final int lineIndex;
        private final int bucketIndex;
        private final long bucketKey;
        private final int entityId;
        private final int componentId;
        private final int levelId;
        private final int eventId;
        private final double score;
        private final boolean prediction;
        private final boolean label;
        private final Dataset dataset;

        private PredictionRecord(String datasetName, double quantile, double threshold, int lineIndex, int bucketIndex,
                long bucketKey, int entityId, int componentId, int levelId, int eventId, double score,
                boolean prediction, boolean label, Dataset dataset) {
            this.datasetName = datasetName;
            this.quantile = quantile;
            this.threshold = threshold;
            this.lineIndex = lineIndex;
            this.bucketIndex = bucketIndex;
            this.bucketKey = bucketKey;
            this.entityId = entityId;
            this.componentId = componentId;
            this.levelId = levelId;
            this.eventId = eventId;
            this.score = score;
            this.prediction = prediction;
            this.label = label;
            this.dataset = dataset;
        }

        private static String header() {
            return "dataset,quantile,threshold,line_index,bucket_index,bucket_key,entity,component,level,template_id,score,prediction,label,template,message";
        }

        private String toCsv() {
            String message = dataset.lineMessages.size() == dataset.labels.size ? dataset.lineMessages.get(lineIndex)
                    : "";
            String template = eventId >= 0 && eventId < dataset.templateTexts.length ? dataset.templateTexts[eventId]
                    : "<unknown>";
            return String.format(Locale.ROOT, "%s,%.5f,%.8f,%d,%d,%d,%s,%s,%s,%d,%.8f,%s,%s,%s,%s", csv(datasetName),
                    quantile, threshold, lineIndex, bucketIndex, bucketKey,
                    csv(nameForId(dataset.entityNames, entityId)), csv(nameForId(dataset.componentNames, componentId)),
                    csv(nameForId(dataset.levelNames, levelId)), eventId, score, prediction ? "anomaly" : "not_anomaly",
                    label ? "anomaly" : "not_anomaly", csv(template), csv(message));
        }
    }

    private static final class DiagnosticThreshold {
        private final double threshold;
        private final double precision;
        private final double recall;
        private final double f1;

        private DiagnosticThreshold(double threshold, double precision, double recall, double f1) {
            this.threshold = threshold;
            this.precision = precision;
            this.recall = recall;
            this.f1 = f1;
        }
    }

    private static final class TemplateAggregate {
        private final int eventId;
        private final Map<Integer, Integer> componentCounts = new HashMap<>();
        private final Map<Integer, Integer> levelCounts = new HashMap<>();
        private final List<String> fpSamples = new ArrayList<>();
        private final List<String> fnSamples = new ArrayList<>();
        private final List<String> tpSamples = new ArrayList<>();
        private long tp;
        private long fp;
        private long fn;
        private long tn;
        private double sumScore;
        private double maxScore;

        private TemplateAggregate(int eventId) {
            this.eventId = eventId;
        }

        private void add(Dataset dataset, int lineIndex, double score, boolean label, boolean prediction,
                int sampleLimit) {
            sumScore += score;
            maxScore = max(maxScore, score);
            int componentId = dataset.componentIds.values[lineIndex];
            int levelId = dataset.levelIds.values[lineIndex];
            componentCounts.put(componentId, componentCounts.getOrDefault(componentId, 0) + 1);
            levelCounts.put(levelId, levelCounts.getOrDefault(levelId, 0) + 1);
            if (label && prediction) {
                ++tp;
                addSample(dataset, lineIndex, tpSamples, sampleLimit);
            } else if (prediction) {
                ++fp;
                addSample(dataset, lineIndex, fpSamples, sampleLimit);
            } else if (label) {
                ++fn;
                addSample(dataset, lineIndex, fnSamples, sampleLimit);
            } else {
                ++tn;
            }
        }

        private void addSample(Dataset dataset, int lineIndex, List<String> samples, int sampleLimit) {
            if (samples.size() >= sampleLimit) {
                return;
            }
            if (dataset.lineMessages.size() == dataset.labels.size) {
                samples.add(dataset.lineMessages.get(lineIndex));
            }
        }

        private long total() {
            return tp + fp + fn + tn;
        }

        private long countFor(String group) {
            switch (group) {
            case "fp":
                return fp;
            case "fn":
                return fn;
            case "tp":
                return tp;
            default:
                return 0;
            }
        }

        private TemplateDiagnostic toDiagnostic(String datasetName, Dataset dataset, TrainStats stats, String scoreMode,
                String group, DiagnosticThreshold threshold) {
            String component = nameForId(dataset.componentNames, dominantId(componentCounts));
            String level = nameForId(dataset.levelNames, dominantId(levelCounts));
            List<String> samples = "fp".equals(group) ? fpSamples : "fn".equals(group) ? fnSamples : tpSamples;
            return new TemplateDiagnostic(datasetName, scoreMode, group, threshold, eventId, total(), tp, fp, fn, tn,
                    sumScore / max(1.0, total()), maxScore, stats.globalRarity(eventId),
                    dataset.templateTexts.length > eventId ? dataset.templateTexts[eventId] : "<unknown>", component,
                    level, samples);
        }

        private int dominantId(Map<Integer, Integer> counts) {
            int bestId = -1;
            int bestCount = -1;
            for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
                if (entry.getValue() > bestCount) {
                    bestId = entry.getKey();
                    bestCount = entry.getValue();
                }
            }
            return bestId;
        }
    }

    private static final class TemplateDiagnostic {
        private final String dataset;
        private final String scoreMode;
        private final String group;
        private final DiagnosticThreshold threshold;
        private final int templateId;
        private final long total;
        private final long tp;
        private final long fp;
        private final long fn;
        private final long tn;
        private final double avgScore;
        private final double maxScore;
        private final double globalRarity;
        private final String template;
        private final String dominantComponent;
        private final String dominantLevel;
        private final List<String> samples;

        private TemplateDiagnostic(String dataset, String scoreMode, String group, DiagnosticThreshold threshold,
                int templateId, long total, long tp, long fp, long fn, long tn, double avgScore, double maxScore,
                double globalRarity, String template, String dominantComponent, String dominantLevel,
                List<String> samples) {
            this.dataset = dataset;
            this.scoreMode = scoreMode;
            this.group = group;
            this.threshold = threshold;
            this.templateId = templateId;
            this.total = total;
            this.tp = tp;
            this.fp = fp;
            this.fn = fn;
            this.tn = tn;
            this.avgScore = avgScore;
            this.maxScore = maxScore;
            this.globalRarity = globalRarity;
            this.template = template;
            this.dominantComponent = dominantComponent;
            this.dominantLevel = dominantLevel;
            this.samples = samples;
        }

        private static String header() {
            return "dataset,score_mode,group,threshold,threshold_precision,threshold_recall,threshold_f1,template_id,total,tp,fp,fn,tn,template_precision,template_recall,avg_score,max_score,global_rarity,dominant_component,dominant_level,template,sample_1,sample_2,sample_3";
        }

        private String toCsv() {
            double templatePrecision = tp + fp == 0 ? 0.0 : tp / (double) (tp + fp);
            double templateRecall = tp + fn == 0 ? 0.0 : tp / (double) (tp + fn);
            return String.format(Locale.ROOT,
                    "%s,%s,%s,%.8f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%s,%s,%s,%s,%s",
                    csv(dataset), csv(scoreMode), csv(group), threshold.threshold, threshold.precision,
                    threshold.recall, threshold.f1, templateId, total, tp, fp, fn, tn, templatePrecision,
                    templateRecall, avgScore, maxScore, globalRarity, csv(dominantComponent), csv(dominantLevel),
                    csv(template), csv(sample(0)), csv(sample(1)), csv(sample(2)));
        }

        private String sample(int index) {
            return index < samples.size() ? samples.get(index) : "";
        }
    }

    private static String nameForId(List<String> names, int id) {
        return id >= 0 && id < names.size() ? names.get(id) : "";
    }

    private static String csv(String value) {
        if (value == null) {
            return "";
        }
        boolean quote = value.indexOf(',') >= 0 || value.indexOf('"') >= 0 || value.indexOf('\n') >= 0
                || value.indexOf('\r') >= 0;
        if (!quote) {
            return value;
        }
        return '"' + value.replace("\"", "\"\"") + '"';
    }

    private static final class TopKMetrics {
        private final int[] predicted;
        private final int[] truePositive;
        private final long totalAnomalies;

        private TopKMetrics(int[] predicted, int[] truePositive, long totalAnomalies) {
            this.predicted = predicted;
            this.truePositive = truePositive;
            this.totalAnomalies = totalAnomalies;
        }

        private double precisionAt(int k) {
            return predicted[k - 1] == 0 ? 0.0 : truePositive[k - 1] / (double) predicted[k - 1];
        }

        private double recallAt(int k) {
            return totalAnomalies == 0 ? 0.0 : truePositive[k - 1] / (double) totalAnomalies;
        }
    }

    private static final class Split {
        private final int trainEnd;
        private final int calibrationEnd;

        private Split(int trainEnd, int calibrationEnd) {
            this.trainEnd = trainEnd;
            this.calibrationEnd = calibrationEnd;
        }

        private static Split from(Dataset dataset, Config config) {
            int trainEnd = max(1, (int) Math.floor(config.trainFraction * dataset.buckets.size()));
            int calibrationEnd = min(dataset.buckets.size() - 1, trainEnd
                    + max(1, (int) Math.floor(config.validationFraction * max(1, dataset.buckets.size() - trainEnd))));
            if (calibrationEnd <= trainEnd) {
                calibrationEnd = min(dataset.buckets.size(), trainEnd + 1);
            }
            return new Split(trainEnd, calibrationEnd);
        }
    }

    private static final class FdrResult {
        private final String dataset;
        private final String method;
        private final String context;
        private final double contextAnomalyRate;
        private final int contextShingle;
        private final double q;
        private final int calibrationHypotheses;
        private final double contextLineRecall;
        private final Metrics metrics;

        private FdrResult(String dataset, String method, String context, double contextAnomalyRate, int contextShingle,
                double q, int calibrationHypotheses, double contextLineRecall, Metrics metrics) {
            this.dataset = dataset;
            this.method = method;
            this.context = context;
            this.contextAnomalyRate = contextAnomalyRate;
            this.contextShingle = contextShingle;
            this.q = q;
            this.calibrationHypotheses = calibrationHypotheses;
            this.contextLineRecall = contextLineRecall;
            this.metrics = metrics;
        }

        private static String header() {
            return "dataset,method,context,context_anomaly_rate,context_shingle,q,calibration_hypotheses,context_line_recall,tp,fp,fn,tn,precision,recall,f1";
        }

        private String toCsv() {
            return String.format(Locale.ROOT, "%s,%s,%s,%.6f,%d,%.6f,%d,%.6f,%d,%d,%d,%d,%.6f,%.6f,%.6f", dataset,
                    method, context, contextAnomalyRate, contextShingle, q, calibrationHypotheses, contextLineRecall,
                    metrics.tp, metrics.fp, metrics.fn, metrics.tn, metrics.precision(), metrics.recall(),
                    metrics.f1());
        }
    }

    private static final class OracleResult {
        private final String dataset;
        private final String score;
        private final String context;
        private final double contextAnomalyRate;
        private final int contextShingle;
        private final int evaluatedLines;
        private final long positives;
        private final double bestPrecisionAtRecall50;
        private final double bestRecallAtPrecision50;
        private final double bestF1;
        private final double bestF1Precision;
        private final double bestF1Recall;
        private final double averagePrecision;
        private final boolean targetFeasible;

        private OracleResult(String dataset, String score, String context, double contextAnomalyRate,
                int contextShingle, int evaluatedLines, long positives, double bestPrecisionAtRecall50,
                double bestRecallAtPrecision50, double bestF1, double bestF1Precision, double bestF1Recall,
                double averagePrecision, boolean targetFeasible) {
            this.dataset = dataset;
            this.score = score;
            this.context = context;
            this.contextAnomalyRate = contextAnomalyRate;
            this.contextShingle = contextShingle;
            this.evaluatedLines = evaluatedLines;
            this.positives = positives;
            this.bestPrecisionAtRecall50 = bestPrecisionAtRecall50;
            this.bestRecallAtPrecision50 = bestRecallAtPrecision50;
            this.bestF1 = bestF1;
            this.bestF1Precision = bestF1Precision;
            this.bestF1Recall = bestF1Recall;
            this.averagePrecision = averagePrecision;
            this.targetFeasible = targetFeasible;
        }

        private static String header() {
            return "dataset,score,context,context_anomaly_rate,context_shingle,evaluated_lines,positives,best_precision_at_recall_0_5,best_recall_at_precision_0_5,best_f1,best_f1_precision,best_f1_recall,average_precision,target_feasible";
        }

        private String toCsv() {
            return String.format(Locale.ROOT, "%s,%s,%s,%.6f,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s", dataset, score,
                    context, contextAnomalyRate, contextShingle, evaluatedLines, positives, bestPrecisionAtRecall50,
                    bestRecallAtPrecision50, bestF1, bestF1Precision, bestF1Recall, averagePrecision, targetFeasible);
        }

        private static OracleResult oracleDiagnosticOnlineSemantic(String datasetName, String scoreMode,
                Dataset dataset, TrainStats stats, ContextSet context, int startBucket, int endBucket) {
            int count = dataset.countLines(startBucket, endBucket, context);
            long[] packed = new long[count];
            int index = 0;
            long positives = 0;
            OnlineSemanticStats onlineStats = new OnlineSemanticStats(stats);
            boolean includePhrase = !scoreMode.endsWith("_no_phrase");
            boolean includeStableSuppression = !scoreMode.endsWith("_no_suppress");
            for (int bucketIndex = startBucket; bucketIndex < endBucket; bucketIndex++) {
                if (!context.candidates[bucketIndex]) {
                    for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                        onlineStats.update(dataset, i);
                    }
                    continue;
                }
                for (int i = dataset.lineStarts[bucketIndex]; i < dataset.lineStarts[bucketIndex + 1]; i++) {
                    boolean label = dataset.labels.values[i] != 0;
                    double score = bglSemanticScore(dataset, stats, context, i, includePhrase, includeStableSuppression,
                            true, onlineStats);
                    packed[index++] = pack(score, label);
                    if (label) {
                        ++positives;
                    }
                    onlineStats.update(dataset, i);
                }
            }
            if (index == 0 || positives == 0) {
                return new OracleResult(datasetName, "score_mode_" + scoreMode, context.name, context.anomalyRate,
                        context.shingleSize, index, positives, 0, 0, 0, 0, 0, 0, false);
            }
            java.util.Arrays.sort(packed, 0, index);
            long tp = 0;
            double bestF1 = 0;
            double bestF1Precision = 0;
            double bestF1Recall = 0;
            double bestPrecisionAtRecall50 = 0;
            double bestRecallAtPrecision50 = 0;
            double averagePrecisionNumerator = 0;
            boolean targetFeasible = false;
            for (int i = 0; i < index; i++) {
                if ((packed[i] & 1L) != 0) {
                    ++tp;
                    averagePrecisionNumerator += tp / (double) (i + 1L);
                }
                double precision = tp / (double) (i + 1L);
                double recall = tp / (double) positives;
                double f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
                if (f1 > bestF1) {
                    bestF1 = f1;
                    bestF1Precision = precision;
                    bestF1Recall = recall;
                }
                if (recall >= 0.5) {
                    bestPrecisionAtRecall50 = max(bestPrecisionAtRecall50, precision);
                }
                if (precision >= 0.5) {
                    bestRecallAtPrecision50 = max(bestRecallAtPrecision50, recall);
                }
                if (precision >= 0.5 && recall >= 0.5) {
                    targetFeasible = true;
                }
            }
            double averagePrecision = averagePrecisionNumerator / positives;
            return new OracleResult(datasetName, "score_mode_" + scoreMode, context.name, context.anomalyRate,
                    context.shingleSize, index, positives, bestPrecisionAtRecall50, bestRecallAtPrecision50, bestF1,
                    bestF1Precision, bestF1Recall, averagePrecision, targetFeasible);
        }
    }

    private static final class Result {
        private final String dataset;
        private final String scorer;
        private final String context;
        private final double contextAnomalyRate;
        private final int contextShingle;
        private final double threshold;
        private final double validationPrecision;
        private final double validationRecall;
        private final double validationF1;
        private final double candidateLineRecall;
        private final Metrics metrics;
        private final TopKMetrics topK;

        private Result(String dataset, String scorer, String context, double contextAnomalyRate, int contextShingle,
                double threshold, double validationPrecision, double validationRecall, double validationF1,
                double candidateLineRecall, Metrics metrics, TopKMetrics topK) {
            this.dataset = dataset;
            this.scorer = scorer;
            this.context = context;
            this.contextAnomalyRate = contextAnomalyRate;
            this.contextShingle = contextShingle;
            this.threshold = threshold;
            this.validationPrecision = validationPrecision;
            this.validationRecall = validationRecall;
            this.validationF1 = validationF1;
            this.candidateLineRecall = candidateLineRecall;
            this.metrics = metrics;
            this.topK = topK;
        }

        private static String header() {
            return "dataset,scorer,context,context_anomaly_rate,context_shingle,threshold,validation_precision,validation_recall,validation_f1,candidate_line_recall,tp,fp,fn,tn,precision,recall,f1,precision_at_1,recall_at_1,precision_at_3,recall_at_3,precision_at_5,recall_at_5";
        }

        private String toCsv() {
            return String.format(Locale.ROOT,
                    "%s,%s,%s,%.6f,%d,%.8f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                    dataset, scorer, context, contextAnomalyRate, contextShingle, threshold, validationPrecision,
                    validationRecall, validationF1, candidateLineRecall, metrics.tp, metrics.fp, metrics.fn, metrics.tn,
                    metrics.precision(), metrics.recall(), metrics.f1(), topK.precisionAt(1), topK.recallAt(1),
                    topK.precisionAt(3), topK.recallAt(3), topK.precisionAt(5), topK.recallAt(5));
        }
    }

    private static final class RarityCount {
        private final double rarity;
        private final int count;

        private RarityCount(double rarity, int count) {
            this.rarity = rarity;
            this.count = count;
        }
    }

    private static final class MemoryComponent {
        private final String name;
        private final long bytes;

        private MemoryComponent(String name, long bytes) {
            this.name = name;
            this.bytes = bytes;
        }
    }

    private static final class StaticMemoryEstimate {
        private final String scoreMode;
        private final long bytes;
        private final long onlineCountsBytes;
        private final long thresholdBytes;
        private final long parserBytes;
        private final long idStateBytes;
        private final long streamContextBytes;

        private StaticMemoryEstimate(String scoreMode, long bytes, long onlineCountsBytes, long thresholdBytes,
                long parserBytes, long idStateBytes, long streamContextBytes) {
            this.scoreMode = scoreMode;
            this.bytes = bytes;
            this.onlineCountsBytes = onlineCountsBytes;
            this.thresholdBytes = thresholdBytes;
            this.parserBytes = parserBytes;
            this.idStateBytes = idStateBytes;
            this.streamContextBytes = streamContextBytes;
        }
    }

    private static final class MemoryLimits {
        private final boolean bounded;
        private final double budgetMiB;
        private final int maxTemplates;
        private final int maxGroups;
        private final int maxGroupTemplatePairs;
        private final int maxStableSignatures;
        private final int maxEntities;
        private final long parserCapBytes;
        private final int maxParserNodes;
        private final int averageEntityChars;
        private final int averageGroupChars;
        private final String entityMode;

        private MemoryLimits(boolean bounded, double budgetMiB, int maxTemplates, int maxGroups,
                int maxGroupTemplatePairs, int maxStableSignatures, int maxEntities, long parserCapBytes,
                int maxParserNodes, int averageEntityChars, int averageGroupChars, String entityMode) {
            this.bounded = bounded;
            this.budgetMiB = budgetMiB;
            this.maxTemplates = max(1, maxTemplates);
            this.maxGroups = max(1, maxGroups);
            this.maxGroupTemplatePairs = max(1, maxGroupTemplatePairs);
            this.maxStableSignatures = max(1, maxStableSignatures);
            this.maxEntities = max(0, maxEntities);
            this.parserCapBytes = max(0L, parserCapBytes);
            this.maxParserNodes = max(1, maxParserNodes);
            this.averageEntityChars = max(0, averageEntityChars);
            this.averageGroupChars = max(0, averageGroupChars);
            this.entityMode = entityMode;
        }

        private static MemoryLimits from(Map<String, String> values, String memoryProfile) {
            String profile = memoryProfile.toLowerCase(Locale.ROOT);
            boolean bounded = !"rich".equals(profile);
            double budget = profileBudgetMiB(profile);
            int maxTemplates = 100000;
            int maxGroups = 1024;
            int maxGroupTemplatePairs = 1000000;
            int maxStableSignatures = 1000000;
            int maxEntities = 0;
            long parserCapBytes = miBToBytes(128.0);
            int maxParserNodes = 200000;
            String defaultEntityMode = bounded ? "none" : "raw";
            if (!bounded) {
                maxTemplates = Integer.MAX_VALUE / 4;
                maxGroups = Integer.MAX_VALUE / 4;
                maxGroupTemplatePairs = Integer.MAX_VALUE / 4;
                maxStableSignatures = Integer.MAX_VALUE / 4;
                maxEntities = Integer.MAX_VALUE / 4;
                parserCapBytes = Long.MAX_VALUE;
                maxParserNodes = Integer.MAX_VALUE / 4;
            } else if ("tiny".equals(profile) || "tiny-10mb".equals(profile) || "tiny_10mb".equals(profile)) {
                maxTemplates = 2000;
                maxGroups = 32;
                maxGroupTemplatePairs = 10000;
                maxStableSignatures = 5000;
                parserCapBytes = miBToBytes(4.0);
                maxParserNodes = 20000;
            } else if ("small-25mb".equals(profile) || "small_25mb".equals(profile) || "compact-25mb".equals(profile)
                    || "compact_25mb".equals(profile)) {
                maxTemplates = 5000;
                maxGroups = 64;
                maxGroupTemplatePairs = 25000;
                maxStableSignatures = 20000;
                parserCapBytes = miBToBytes(8.0);
                maxParserNodes = 50000;
            } else if ("compact".equals(profile) || "compact-50mb".equals(profile) || "compact_50mb".equals(profile)) {
                maxTemplates = 10000;
                maxGroups = 128;
                maxGroupTemplatePairs = 75000;
                maxStableSignatures = 50000;
                parserCapBytes = miBToBytes(16.0);
                maxParserNodes = 100000;
            } else if ("default-100mb".equals(profile) || "default_100mb".equals(profile)) {
                maxTemplates = 20000;
                maxGroups = 128;
                maxGroupTemplatePairs = 150000;
                maxStableSignatures = 100000;
                parserCapBytes = miBToBytes(24.0);
                maxParserNodes = 150000;
            } else if ("custom".equals(profile)) {
                budget = 100.0;
                maxTemplates = 20000;
                maxGroups = 128;
                maxGroupTemplatePairs = 150000;
                maxStableSignatures = 100000;
                parserCapBytes = miBToBytes(24.0);
                maxParserNodes = 150000;
            }
            budget = parseDouble(values, "model-memory-budget-mib", budget);
            maxTemplates = parseInt(values, "max-templates", maxTemplates);
            maxGroups = parseInt(values, "max-groups", maxGroups);
            maxGroupTemplatePairs = parseInt(values, "max-group-template-pairs", maxGroupTemplatePairs);
            maxStableSignatures = parseInt(values, "max-stable-signatures", maxStableSignatures);
            maxEntities = parseInt(values, "max-entities", maxEntities);
            parserCapBytes = miBToBytes(
                    parseDouble(values, "parser-memory-cap-mib", parserCapBytes / (1024.0 * 1024.0)));
            maxParserNodes = parseInt(values, "max-parser-nodes", maxParserNodes);
            int averageEntityChars = parseInt(values, "average-entity-chars", 32);
            int averageGroupChars = parseInt(values, "average-group-chars", 32);
            String entityMode = values.getOrDefault("entity-mode", defaultEntityMode).toLowerCase(Locale.ROOT);
            return new MemoryLimits(bounded, budget, maxTemplates, maxGroups, maxGroupTemplatePairs,
                    maxStableSignatures, maxEntities, parserCapBytes, maxParserNodes, averageEntityChars,
                    averageGroupChars, entityMode);
        }

        private static double profileBudgetMiB(String profile) {
            if ("tiny".equals(profile) || "tiny-10mb".equals(profile) || "tiny_10mb".equals(profile)) {
                return 10.0;
            }
            if ("small-25mb".equals(profile) || "small_25mb".equals(profile) || "compact-25mb".equals(profile)
                    || "compact_25mb".equals(profile)) {
                return 25.0;
            }
            if ("compact".equals(profile) || "compact-50mb".equals(profile) || "compact_50mb".equals(profile)) {
                return 50.0;
            }
            if ("default-100mb".equals(profile) || "default_100mb".equals(profile)) {
                return 100.0;
            }
            return Double.POSITIVE_INFINITY;
        }

        private static int parseInt(Map<String, String> values, String key, int defaultValue) {
            return Integer.parseInt(values.getOrDefault(key, Integer.toString(defaultValue)));
        }

        private static double parseDouble(Map<String, String> values, String key, double defaultValue) {
            return Double.parseDouble(values.getOrDefault(key, Double.toString(defaultValue)));
        }

        private static long miBToBytes(double mib) {
            if (Double.isInfinite(mib)) {
                return Long.MAX_VALUE;
            }
            return (long) Math.ceil(mib * 1024.0 * 1024.0);
        }

        private boolean entityEnabled() {
            return !"none".equals(entityMode) && maxEntities > 0;
        }

        private long budgetBytes() {
            return miBToBytes(budgetMiB);
        }

        private String summary() {
            return String.format(Locale.ROOT,
                    "templates=%d groups=%d group_template_pairs=%d stable_signatures=%d entities=%d parser_cap_mib=%.2f entity_mode=%s",
                    maxTemplates, maxGroups, maxGroupTemplatePairs, maxStableSignatures, maxEntities,
                    bytesToMiB(parserCapBytes), entityMode);
        }
    }

    private static final class Config {
        private final List<String> datasets;
        private final String bglPath;
        private final String thunderbirdPath;
        private final String output;
        private final int bucketSeconds;
        private final int bglSampleModulo;
        private final int thunderbirdSampleModulo;
        private final double trainFraction;
        private final double validationFraction;
        private final int topK;
        private final int rareEventMaxTrainCount;
        private final String parserMode;
        private final double drainSimilarity;
        private final int drainDepth;
        private final int drainMaxChildren;
        private final int numberOfTrees;
        private final int sampleSize;
        private final int outputAfter;
        private final double zFactor;
        private final long seed;
        private final int progressInterval;
        private final int top;
        private final boolean includeUngated;
        private final List<Double> contextAnomalyRates;
        private final String thresholdMode;
        private final List<Double> fdrQValues;
        private final String statsPeriod;
        private final String scoreMode;
        private final String memoryProfile;
        private final MemoryLimits memoryLimits;
        private final int rollingHorizonBuckets;
        private final int rollingBins;
        private final double rollingScoreMax;
        private final boolean excludeAlertUpdates;
        private final int expandBuckets;
        private final int tailBlockBuckets;
        private final int tailMinSelected;
        private final double tailSignificance;
        private final boolean parameterFeatures;
        private final boolean entityBucketCounts;
        private final double diagnosticRecallTarget;
        private final int diagnosticTopTemplates;
        private final int diagnosticSamples;
        private final List<QuantilePair> seedExpandPairs;
        private final double diagnosticQuantile;
        private final double predictionFraction;
        private final double onlineTemplateHalfLifeDays;
        private final double onlineStableHalfLifeDays;
        private final double onlineFastHalfLifeHours;
        private final double onlineQuantileHalfLifeDays;
        private final double onlineAlertUpdateWeight;
        private final boolean onlineWinsorizeThresholdUpdates;
        private final double onlineWinsorizeScoreMargin;
        private final int onlineDecayIntervalBuckets;
        private final double onlineKeywordGateQuantile;
        private final boolean onlineStrictThreshold;
        private final String onlineThresholdGroup;
        private final int onlineThresholdMinCount;
        private final boolean onlineUpdateCounts;
        private final boolean onlineUpdateThresholds;
        private final double onlineUpdateGuardQuantile;
        private final double onlineTiebreakWeight;
        private final boolean onlineLazyDecay;
        private final boolean onlineCountSketchParameters;
        private final int onlineCountSketchDepth;
        private final int onlineCountSketchWidth;
        private final double anchoredThresholdHalfLifeDays;
        private final double anchoredThresholdShrinkageK;
        private final boolean anchoredLogScores;
        private final boolean anchoredThresholdFloorParent;
        private final String anchoredThresholdDynamicMode;
        private final String anchorSelect;
        private final double anchorMinQuantile;
        private final double anchorMaxQuantile;
        private final int anchorSignatureCap;
        private final double anchorCalibrationFraction;
        private final boolean profileOnline;
        private final int profileMaxEvalBuckets;

        private Config(List<String> datasets, String bglPath, String thunderbirdPath, String output, int bucketSeconds,
                int bglSampleModulo, int thunderbirdSampleModulo, double trainFraction, double validationFraction,
                int topK, int rareEventMaxTrainCount, String parserMode, double drainSimilarity, int drainDepth,
                int drainMaxChildren, int numberOfTrees, int sampleSize, int outputAfter, double zFactor, long seed,
                int progressInterval, int top, boolean includeUngated, List<Double> contextAnomalyRates,
                String thresholdMode, List<Double> fdrQValues, String statsPeriod, String scoreMode,
                String memoryProfile, MemoryLimits memoryLimits, int rollingHorizonBuckets, int rollingBins,
                double rollingScoreMax, boolean excludeAlertUpdates, int expandBuckets, int tailBlockBuckets,
                int tailMinSelected, double tailSignificance, boolean parameterFeatures, boolean entityBucketCounts,
                double diagnosticRecallTarget, int diagnosticTopTemplates, int diagnosticSamples,
                List<QuantilePair> seedExpandPairs, double diagnosticQuantile, double predictionFraction,
                double onlineTemplateHalfLifeDays, double onlineStableHalfLifeDays, double onlineFastHalfLifeHours,
                double onlineQuantileHalfLifeDays, double onlineAlertUpdateWeight,
                boolean onlineWinsorizeThresholdUpdates, double onlineWinsorizeScoreMargin,
                int onlineDecayIntervalBuckets, double onlineKeywordGateQuantile, boolean onlineStrictThreshold,
                String onlineThresholdGroup, int onlineThresholdMinCount, boolean onlineUpdateCounts,
                boolean onlineUpdateThresholds, double onlineUpdateGuardQuantile, double onlineTiebreakWeight,
                boolean onlineLazyDecay, boolean onlineCountSketchParameters, int onlineCountSketchDepth,
                int onlineCountSketchWidth, double anchoredThresholdHalfLifeDays, double anchoredThresholdShrinkageK,
                boolean anchoredLogScores, boolean anchoredThresholdFloorParent, String anchoredThresholdDynamicMode,
                String anchorSelect, double anchorMinQuantile, double anchorMaxQuantile, int anchorSignatureCap,
                double anchorCalibrationFraction, boolean profileOnline, int profileMaxEvalBuckets) {
            this.datasets = datasets;
            this.bglPath = bglPath;
            this.thunderbirdPath = thunderbirdPath;
            this.output = output;
            this.bucketSeconds = bucketSeconds;
            this.bglSampleModulo = bglSampleModulo;
            this.thunderbirdSampleModulo = thunderbirdSampleModulo;
            this.trainFraction = trainFraction;
            this.validationFraction = validationFraction;
            this.topK = topK;
            this.rareEventMaxTrainCount = rareEventMaxTrainCount;
            this.parserMode = parserMode;
            this.drainSimilarity = drainSimilarity;
            this.drainDepth = drainDepth;
            this.drainMaxChildren = drainMaxChildren;
            this.numberOfTrees = numberOfTrees;
            this.sampleSize = sampleSize;
            this.outputAfter = outputAfter;
            this.zFactor = zFactor;
            this.seed = seed;
            this.progressInterval = progressInterval;
            this.top = top;
            this.includeUngated = includeUngated;
            this.contextAnomalyRates = contextAnomalyRates;
            this.thresholdMode = thresholdMode;
            this.fdrQValues = fdrQValues;
            this.statsPeriod = statsPeriod;
            this.scoreMode = scoreMode;
            this.memoryProfile = memoryProfile;
            this.memoryLimits = memoryLimits;
            this.rollingHorizonBuckets = rollingHorizonBuckets;
            this.rollingBins = rollingBins;
            this.rollingScoreMax = rollingScoreMax;
            this.excludeAlertUpdates = excludeAlertUpdates;
            this.expandBuckets = expandBuckets;
            this.tailBlockBuckets = tailBlockBuckets;
            this.tailMinSelected = tailMinSelected;
            this.tailSignificance = tailSignificance;
            this.parameterFeatures = parameterFeatures;
            this.entityBucketCounts = entityBucketCounts;
            this.diagnosticRecallTarget = diagnosticRecallTarget;
            this.diagnosticTopTemplates = diagnosticTopTemplates;
            this.diagnosticSamples = diagnosticSamples;
            this.seedExpandPairs = seedExpandPairs;
            this.diagnosticQuantile = diagnosticQuantile;
            this.predictionFraction = predictionFraction;
            this.onlineTemplateHalfLifeDays = onlineTemplateHalfLifeDays;
            this.onlineStableHalfLifeDays = onlineStableHalfLifeDays;
            this.onlineFastHalfLifeHours = onlineFastHalfLifeHours;
            this.onlineQuantileHalfLifeDays = onlineQuantileHalfLifeDays;
            this.onlineAlertUpdateWeight = onlineAlertUpdateWeight;
            this.onlineWinsorizeThresholdUpdates = onlineWinsorizeThresholdUpdates;
            this.onlineWinsorizeScoreMargin = onlineWinsorizeScoreMargin;
            this.onlineDecayIntervalBuckets = onlineDecayIntervalBuckets;
            this.onlineKeywordGateQuantile = onlineKeywordGateQuantile;
            this.onlineStrictThreshold = onlineStrictThreshold;
            this.onlineThresholdGroup = onlineThresholdGroup;
            this.onlineThresholdMinCount = onlineThresholdMinCount;
            this.onlineUpdateCounts = onlineUpdateCounts;
            this.onlineUpdateThresholds = onlineUpdateThresholds;
            this.onlineUpdateGuardQuantile = onlineUpdateGuardQuantile;
            this.onlineTiebreakWeight = onlineTiebreakWeight;
            this.onlineLazyDecay = onlineLazyDecay;
            this.onlineCountSketchParameters = onlineCountSketchParameters;
            this.onlineCountSketchDepth = onlineCountSketchDepth;
            this.onlineCountSketchWidth = onlineCountSketchWidth;
            this.anchoredThresholdHalfLifeDays = anchoredThresholdHalfLifeDays;
            this.anchoredThresholdShrinkageK = anchoredThresholdShrinkageK;
            this.anchoredLogScores = anchoredLogScores;
            this.anchoredThresholdFloorParent = anchoredThresholdFloorParent;
            this.anchoredThresholdDynamicMode = anchoredThresholdDynamicMode;
            this.anchorSelect = anchorSelect;
            this.anchorMinQuantile = anchorMinQuantile;
            this.anchorMaxQuantile = anchorMaxQuantile;
            this.anchorSignatureCap = anchorSignatureCap;
            this.anchorCalibrationFraction = anchorCalibrationFraction;
            this.profileOnline = profileOnline;
            this.profileMaxEvalBuckets = profileMaxEvalBuckets;
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
            String root = values.getOrDefault("data-root", "/tmp/loghub_data");
            String trainFractionValue = values.getOrDefault("warmup-fraction",
                    values.getOrDefault("train-fraction", "0.10"));
            String predictionFractionValue = values.getOrDefault("score-fraction",
                    values.getOrDefault("prediction-fraction", "0.01"));
            String thresholdModeValue = values.getOrDefault("threshold-mode", "supervised").toLowerCase(Locale.ROOT);
            String scoreModeDefault = "estimate-memory-static".equals(thresholdModeValue)
                    || "estimate_memory_static".equals(thresholdModeValue) ? "auto" : "combined";
            String scoreModeValue = values.getOrDefault("score-mode", scoreModeDefault).toLowerCase(Locale.ROOT);
            String memoryProfileValue = values
                    .getOrDefault("memory-profile", defaultMemoryProfile(values, thresholdModeValue))
                    .toLowerCase(Locale.ROOT);
            MemoryLimits memoryLimits = MemoryLimits.from(values, memoryProfileValue);
            String fdrDefault = "online".equals(thresholdModeValue) || "anchored-dynamic".equals(thresholdModeValue)
                    || "anchored_dynamic".equals(thresholdModeValue) || "anchored-online".equals(thresholdModeValue)
                    || "anchored_online".equals(thresholdModeValue)
                    || "streaming-anchored-online".equals(thresholdModeValue)
                    || "streaming_anchored_online".equals(thresholdModeValue)
                    || "anchored-online-streaming".equals(thresholdModeValue)
                    || "anchored_online_streaming".equals(thresholdModeValue) ? "0.995,0.997,0.999"
                            : "0.20,0.30,0.40,0.50,0.60";
            return new Config(split(values.getOrDefault("datasets", "bgl,thunderbird")),
                    values.getOrDefault("bgl", root + "/BGL/BGL.log"),
                    values.getOrDefault("thunderbird", root + "/Thunderbird/Thunderbird.log"),
                    values.getOrDefault("output", "benchmark-results/log-ad/line_level_log_anomaly_results.csv"),
                    Integer.parseInt(values.getOrDefault("bucket-seconds", "600")),
                    Integer.parseInt(values.getOrDefault("bgl-sample-modulo", "1")),
                    Integer.parseInt(values.getOrDefault("thunderbird-sample-modulo", "10")),
                    Double.parseDouble(trainFractionValue),
                    Double.parseDouble(values.getOrDefault("validation-fraction", "0.10")),
                    Integer.parseInt(values.getOrDefault("top-k", "25")),
                    Integer.parseInt(values.getOrDefault("rare-max-train-count", "5")),
                    values.getOrDefault("parser", "simple").toLowerCase(Locale.ROOT),
                    Double.parseDouble(values.getOrDefault("drain-similarity", "0.5")),
                    Integer.parseInt(values.getOrDefault("drain-depth", "4")),
                    Integer.parseInt(values.getOrDefault("drain-max-children", "100")),
                    Integer.parseInt(values.getOrDefault("trees", "100")),
                    Integer.parseInt(values.getOrDefault("sample-size", "256")),
                    Integer.parseInt(values.getOrDefault("output-after", "32")),
                    Double.parseDouble(values.getOrDefault("z-factor", "3.0")),
                    Long.parseLong(values.getOrDefault("seed", "42")),
                    Integer.parseInt(values.getOrDefault("progress-interval", "5000000")),
                    Integer.parseInt(values.getOrDefault("top", "20")),
                    Boolean.parseBoolean(values.getOrDefault("include-ungated", "true")),
                    splitDoubles(values.getOrDefault("context-anomaly-rates", "0.02,0.05,0.10,0.20")),
                    thresholdModeValue, splitDoubles(values.getOrDefault("fdr-q", fdrDefault)),
                    values.getOrDefault("stats-period", "train").toLowerCase(Locale.ROOT), scoreModeValue,
                    memoryProfileValue, memoryLimits,
                    Integer.parseInt(values.getOrDefault("rolling-horizon-buckets", "144")),
                    Integer.parseInt(values.getOrDefault("rolling-bins", "4096")),
                    Double.parseDouble(values.getOrDefault("rolling-score-max", "80.0")),
                    Boolean.parseBoolean(values.getOrDefault("exclude-alert-updates", "true")),
                    Integer.parseInt(values.getOrDefault("expand-buckets", "2")),
                    Integer.parseInt(values.getOrDefault("tail-block-buckets", "144")),
                    Integer.parseInt(values.getOrDefault("tail-min-selected", "5")),
                    Double.parseDouble(values.getOrDefault("tail-significance", "0.05")),
                    Boolean.parseBoolean(values.getOrDefault("parameter-features", "false")),
                    Boolean.parseBoolean(values.getOrDefault("entity-bucket-counts", "false")),
                    Double.parseDouble(values.getOrDefault("diagnostic-recall-target", "0.50")),
                    Integer.parseInt(values.getOrDefault("diagnostic-top-templates", values.getOrDefault("top", "20"))),
                    Integer.parseInt(values.getOrDefault("diagnostic-samples", "3")),
                    splitQuantilePairs(values.getOrDefault("seed-expand-pairs", "0.999:0.99,0.999:0.995,0.9995:0.995")),
                    Double.parseDouble(values.getOrDefault("diagnostic-quantile", "NaN")),
                    Double.parseDouble(predictionFractionValue),
                    Double.parseDouble(values.getOrDefault("online-template-half-life-days", "7.0")),
                    Double.parseDouble(values.getOrDefault("online-stable-half-life-days", "30.0")),
                    Double.parseDouble(values.getOrDefault("online-fast-half-life-hours", "6.0")),
                    Double.parseDouble(values.getOrDefault("online-quantile-half-life-days", "1.0")),
                    Double.parseDouble(values.getOrDefault("online-alert-update-weight", "0.05")),
                    Boolean.parseBoolean(values.getOrDefault("online-winsorize-threshold-updates", "true")),
                    Double.parseDouble(values.getOrDefault("online-winsorize-score-margin", "5.0")),
                    Integer.parseInt(values.getOrDefault("online-decay-interval-buckets", "12")),
                    Double.parseDouble(values.getOrDefault("online-keyword-gate-quantile", "0.90")),
                    Boolean.parseBoolean(values.getOrDefault("online-strict-threshold", "false")),
                    values.getOrDefault("online-threshold-group", "global").toLowerCase(Locale.ROOT),
                    Integer.parseInt(values.getOrDefault("online-threshold-min-count", "200")),
                    Boolean.parseBoolean(values.getOrDefault("online-update-counts", "true")),
                    Boolean.parseBoolean(values.getOrDefault("online-update-thresholds", "true")),
                    Double.parseDouble(values.getOrDefault("online-update-guard-q", "NaN")),
                    Double.parseDouble(values.getOrDefault("online-tiebreak-weight", "0.001")),
                    Boolean.parseBoolean(values.getOrDefault("online-lazy-decay", "true")),
                    Boolean.parseBoolean(values.getOrDefault("online-count-sketch-parameters", "true")),
                    Integer.parseInt(values.getOrDefault("online-count-sketch-depth", "4")),
                    Integer.parseInt(values.getOrDefault("online-count-sketch-width", "524288")),
                    Double.parseDouble(values.getOrDefault("anchored-threshold-half-life-days", "7.0")),
                    Double.parseDouble(values.getOrDefault("anchored-threshold-shrinkage-k", "1000")),
                    Boolean.parseBoolean(values.getOrDefault("anchored-log-scores", "true")),
                    Boolean.parseBoolean(values.getOrDefault("anchored-threshold-floor-parent", "true")),
                    values.getOrDefault("anchored-threshold-dynamic",
                            values.getOrDefault("anchored-dynamic-mode", "relative")).toLowerCase(Locale.ROOT),
                    values.getOrDefault("anchor-select", "auto").toLowerCase(Locale.ROOT),
                    Double.parseDouble(values.getOrDefault("anchor-min-quantile", "0.980")),
                    Double.parseDouble(values.getOrDefault("anchor-max-quantile", "0.9995")),
                    Integer.parseInt(values.getOrDefault("anchor-signature-cap", "200")),
                    Double.parseDouble(values.getOrDefault("anchor-calibration-fraction", "1.0")),
                    Boolean.parseBoolean(values.getOrDefault("profile-online", "false")),
                    Integer.parseInt(values.getOrDefault("profile-max-eval-buckets", "0")));
        }

        private static String defaultMemoryProfile(Map<String, String> values, String thresholdMode) {
            if (!values.containsKey("model-memory-budget-mib")) {
                return "estimate-memory-static".equals(thresholdMode) || "estimate_memory_static".equals(thresholdMode)
                        ? "default-100mb"
                        : "rich";
            }
            double budgetMiB = Double.parseDouble(values.get("model-memory-budget-mib"));
            if (budgetMiB <= 10.0) {
                return "tiny-10mb";
            }
            if (budgetMiB <= 25.0) {
                return "compact-25mb";
            }
            if (budgetMiB <= 50.0) {
                return "compact-50mb";
            }
            if (budgetMiB < 80.0) {
                return "compact-50mb";
            }
            if (budgetMiB <= 100.0) {
                return "default-100mb";
            }
            return "custom";
        }

        private boolean useParameterFeatures() {
            if (useCompactOnlineState()) {
                return parameterFeatures || scoreMode.contains("param");
            }
            return parameterFeatures || "auto".equals(scoreMode) || scoreMode.contains("param")
                    || scoreMode.contains("tiebreak") || scoreMode.startsWith("bgl_") || useSemanticFeatures();
        }

        private boolean useCompactOnlineState() {
            return "compact".equals(memoryProfile) || "default-100mb".equals(memoryProfile)
                    || "default_100mb".equals(memoryProfile) || "tiny".equals(memoryProfile)
                    || "tiny-10mb".equals(memoryProfile) || "tiny_10mb".equals(memoryProfile)
                    || "small-25mb".equals(memoryProfile) || "small_25mb".equals(memoryProfile)
                    || "compact-25mb".equals(memoryProfile) || "compact_25mb".equals(memoryProfile)
                    || "compact-50mb".equals(memoryProfile) || "compact_50mb".equals(memoryProfile)
                    || "custom".equals(memoryProfile);
        }

        private boolean useFixedAnchorQuantile() {
            return "fixed".equals(anchorSelect) || "quantile".equals(anchorSelect);
        }

        private boolean useHistogramAnchorQuantile() {
            return "histogram".equals(anchorSelect) || "histogram-quantile".equals(anchorSelect)
                    || "histogram_quantile".equals(anchorSelect);
        }

        private boolean useHistogramAnchorQuantile(String scoreMode) {
            return useHistogramAnchorQuantile() || ("auto".equals(anchorSelect) && scoreMode.startsWith("global_"));
        }

        private double fixedAnchorQuantile() {
            double quantile = fdrQValues.isEmpty() ? 0.995 : fdrQValues.get(0);
            return max(anchorMinQuantile, min(anchorMaxQuantile, quantile));
        }

        private boolean useWarmupRelativeDynamicThreshold() {
            return "relative".equals(anchoredThresholdDynamicMode)
                    || "warmup-relative".equals(anchoredThresholdDynamicMode)
                    || "warmup_relative".equals(anchoredThresholdDynamicMode)
                    || "shift".equals(anchoredThresholdDynamicMode);
        }

        private boolean useEntityBucketCounts() {
            if (useCompactOnlineState()) {
                return entityBucketCounts || scoreMode.contains("spike") || scoreMode.startsWith("bgl_")
                        || useSemanticFeatures();
            }
            return entityBucketCounts || scoreMode.contains("spike") || scoreMode.contains("tiebreak")
                    || scoreMode.startsWith("bgl_") || scoreMode.equals("combined") || useSemanticFeatures();
        }

        private boolean useSemanticFeatures() {
            return scoreMode.contains("semantic");
        }

        private boolean useFixedDepthDrainParser() {
            return "drain".equals(parserMode) || "fixed-drain".equals(parserMode) || "tree-drain".equals(parserMode);
        }

        private boolean storeLineMessages() {
            return "diagnostic".equals(thresholdMode) || "predict".equals(thresholdMode);
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

        private static List<Double> splitDoubles(String csv) {
            List<Double> result = new ArrayList<>();
            for (String item : csv.split(",")) {
                String trimmed = item.trim();
                if (!trimmed.isEmpty()) {
                    result.add(Double.parseDouble(trimmed));
                }
            }
            return result;
        }

        private static List<QuantilePair> splitQuantilePairs(String csv) {
            List<QuantilePair> result = new ArrayList<>();
            for (String item : csv.split(",")) {
                String trimmed = item.trim();
                if (trimmed.isEmpty()) {
                    continue;
                }
                String[] parts = trimmed.split(":");
                if (parts.length != 2) {
                    throw new IllegalArgumentException("expected seed:expand pair, got " + trimmed);
                }
                result.add(new QuantilePair(Double.parseDouble(parts[0]), Double.parseDouble(parts[1])));
            }
            return result;
        }
    }

    private static final class QuantilePair {
        private final double seedQuantile;
        private final double expandQuantile;

        private QuantilePair(double seedQuantile, double expandQuantile) {
            this.seedQuantile = seedQuantile;
            this.expandQuantile = expandQuantile;
        }
    }
}
