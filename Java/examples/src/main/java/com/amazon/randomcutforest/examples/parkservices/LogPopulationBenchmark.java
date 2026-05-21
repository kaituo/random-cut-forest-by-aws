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
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.BiFunction;

import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.GlobalLocalAnomalyDetector;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.config.ScoringStrategy;
import com.amazon.randomcutforest.parkservices.returntypes.GenericAnomalyDescriptor;

/**
 * Log population-analysis benchmark using the same GlobalLocalAnomalyDetector
 * style as {@link StringGLADexample}. HDFS is evaluated at block/session level.
 * BGL is evaluated on population members defined as (10-minute bucket, node),
 * with an additional bucket-level rollup for comparison with time-bucket log AD.
 */
public final class LogPopulationBenchmark {

    private LogPopulationBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        Map<String, Metrics> allMetrics = new LinkedHashMap<>();

        if (config.hdfsOccurrence != null) {
            Metrics metrics = runHdfs(config);
            allMetrics.put(metrics.name, metrics);
            if (config.includeHdfsRcf) {
                Metrics rcfMetrics = runHdfsRcf(config);
                allMetrics.put(rcfMetrics.name, rcfMetrics);
            }
        }
        if (config.bglLog != null) {
            BglPopulationResult result = runBgl(config);
            allMetrics.put(result.entityMetrics.name, result.entityMetrics);
            allMetrics.put(result.bucketMetrics.name, result.bucketMetrics);
        }

        System.out.println(Metrics.header());
        for (Metrics metrics : allMetrics.values()) {
            System.out.println(metrics.toCsv());
        }

        if (config.output != null) {
            write(config.output, allMetrics);
        }
    }

    private static Metrics runHdfs(Config config) throws IOException {
        GlobalLocalAnomalyDetector<SparseVector> glad = detector(config);
        Metrics metrics = new Metrics("HDFS_session_GLAD", config.hdfsWarmup);
        long row = 0;
        long started = System.nanoTime();
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(config.hdfsOccurrence), StandardCharsets.UTF_8)) {
            String header = reader.readLine();
            if (header == null) {
                throw new IllegalArgumentException("empty HDFS occurrence matrix");
            }
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",", -1);
                if (parts.length < 4) {
                    continue;
                }
                boolean label = "Fail".equals(parts[1]);
                SparseVector vector = SparseVector.fromDense(parts, 3, "hdfs:" + parts[0]);
                GenericAnomalyDescriptor<SparseVector> result = glad.process(vector, 1.0f, null, true);
                boolean prediction = result.getAnomalyGrade() > 0;
                metrics.record(row, label, prediction, result.getScore(), result.getThreshold());
                ++row;
            }
        }
        metrics.elapsedSeconds = (System.nanoTime() - started) / 1_000_000_000.0;
        metrics.state = String.format(Locale.ROOT, "capacity=%d,z=%.2f,clusters=%d", config.capacity, config.zFactor,
                glad.getClusters() == null ? 0 : glad.getClusters().size());
        return metrics;
    }

    private static Metrics runHdfsRcf(Config config) throws IOException {
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().dimensions(29).shingleSize(1)
                .sampleSize(256).numberOfTrees(50).randomSeed(config.seed).outputAfter(Math.max(32, config.hdfsWarmup / 2))
                .transformMethod(TransformMethod.NORMALIZE).scoringStrategy(ScoringStrategy.MULTI_MODE_RECALL)
                .anomalyRate(0.02).autoAdjust(true).build();
        Metrics metrics = new Metrics("HDFS_session_RCF", config.hdfsWarmup);
        long row = 0;
        long started = System.nanoTime();
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(config.hdfsOccurrence), StandardCharsets.UTF_8)) {
            String header = reader.readLine();
            if (header == null) {
                throw new IllegalArgumentException("empty HDFS occurrence matrix");
            }
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",", -1);
                if (parts.length < 32) {
                    continue;
                }
                boolean label = "Fail".equals(parts[1]);
                double[] vector = new double[29];
                for (int i = 0; i < vector.length; i++) {
                    vector[i] = Double.parseDouble(parts[i + 3]);
                }
                AnomalyDescriptor result = forest.process(vector, row);
                boolean prediction = result.getAnomalyGrade() > 0;
                metrics.record(row, label, prediction, result.getRCFScore(), result.getThreshold());
                ++row;
            }
        }
        metrics.elapsedSeconds = (System.nanoTime() - started) / 1_000_000_000.0;
        metrics.state = "rcf=50x256,shingle=1";
        return metrics;
    }

    private static BglPopulationResult runBgl(Config config) throws IOException {
        GlobalLocalAnomalyDetector<SparseVector> glad = detector(config);
        SimpleDrainParser drain = new SimpleDrainParser(config.drainSimilarity);
        Metrics entityMetrics = new Metrics("BGL_10m_node_GLAD_entity", 0);
        BucketRollup rollup = new BucketRollup(config.bglWarmupBuckets);

        long started = System.nanoTime();
        BglBucket current = null;
        long bucketOrdinal = 0;
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(config.bglLog), StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                ParsedBgl parsed = ParsedBgl.parse(line);
                if (parsed == null) {
                    continue;
                }
                long bucketKey = Math.floorDiv(parsed.epochSeconds, config.bucketSeconds);
                if (current == null) {
                    current = new BglBucket(bucketKey, bucketOrdinal++);
                } else if (bucketKey != current.bucketKey) {
                    processBglBucket(current, drain, glad, entityMetrics, rollup);
                    current = new BglBucket(bucketKey, bucketOrdinal++);
                }
                current.add(parsed);
            }
        }
        if (current != null) {
            processBglBucket(current, drain, glad, entityMetrics, rollup);
        }
        entityMetrics.elapsedSeconds = (System.nanoTime() - started) / 1_000_000_000.0;
        entityMetrics.state = String.format(Locale.ROOT, "capacity=%d,z=%.2f,templates=%d,clusters=%d", config.capacity,
                config.zFactor, drain.templateCount(), glad.getClusters() == null ? 0 : glad.getClusters().size());

        Metrics bucketMetrics = rollup.toMetrics("BGL_10m_node_GLAD_bucket");
        bucketMetrics.elapsedSeconds = entityMetrics.elapsedSeconds;
        bucketMetrics.state = entityMetrics.state;
        return new BglPopulationResult(entityMetrics, bucketMetrics);
    }

    private static GlobalLocalAnomalyDetector<SparseVector> detector(Config config) {
        GlobalLocalAnomalyDetector<SparseVector> glad = GlobalLocalAnomalyDetector.builder().randomSeed(config.seed)
                .capacity(config.capacity).timeDecay(1.0 / config.capacity)
                .doNotReclusterWithin(config.doNotReclusterWithin).maxAllowed(config.maxAllowed)
                .numberOfRepresentatives(config.representatives).build();
        glad.setGlobalDistance(SparseVector::l2LogDistance);
        glad.setZfactor(config.zFactor);
        glad.setLowerThreshold(config.lowerThreshold);
        return glad;
    }

    private static void processBglBucket(BglBucket bucket, SimpleDrainParser drain,
            GlobalLocalAnomalyDetector<SparseVector> glad, Metrics entityMetrics, BucketRollup rollup) {
        boolean bucketPrediction = false;
        boolean bucketLabel = false;
        for (NodeAggregate aggregate : bucket.nodes.values()) {
            SparseVector vector = aggregate.toVector(drain, bucket.bucketKey);
            GenericAnomalyDescriptor<SparseVector> result = glad.process(vector, 1.0f, null, true);
            boolean prediction = result.getAnomalyGrade() > 0;
            bucketPrediction = bucketPrediction || prediction;
            bucketLabel = bucketLabel || aggregate.label;
            entityMetrics.record(bucket.ordinal, aggregate.label, prediction, result.getScore(), result.getThreshold());
        }
        rollup.record(bucket.ordinal, bucketLabel, bucketPrediction);
    }

    private static void write(String output, Map<String, Metrics> metrics) throws IOException {
        Path path = Paths.get(output);
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write(Metrics.header());
            writer.newLine();
            for (Metrics item : metrics.values()) {
                writer.write(item.toCsv());
                writer.newLine();
            }
        }
    }

    private static final class BglPopulationResult {
        private final Metrics entityMetrics;
        private final Metrics bucketMetrics;

        private BglPopulationResult(Metrics entityMetrics, Metrics bucketMetrics) {
            this.entityMetrics = entityMetrics;
            this.bucketMetrics = bucketMetrics;
        }
    }

    private static final class BglBucket {
        private final long bucketKey;
        private final long ordinal;
        private final Map<String, NodeAggregate> nodes = new LinkedHashMap<>();

        private BglBucket(long bucketKey, long ordinal) {
            this.bucketKey = bucketKey;
            this.ordinal = ordinal;
        }

        private void add(ParsedBgl parsed) {
            nodes.computeIfAbsent(parsed.node, NodeAggregate::new).add(parsed);
        }
    }

    private static final class NodeAggregate {
        private static final int LOG_COUNT = 0;
        private static final int INFO_COUNT = 1;
        private static final int WARNING_COUNT = 2;
        private static final int ERROR_COUNT = 3;
        private static final int FATAL_COUNT = 4;
        private static final int FAILURE_COUNT = 5;
        private static final int TEMPLATE_OFFSET = 6;

        private final String node;
        private final Map<String, Integer> messageCounts = new HashMap<>();
        private final Map<String, Integer> levelCounts = new HashMap<>();
        private boolean label;

        private NodeAggregate(String node) {
            this.node = node;
        }

        private void add(ParsedBgl parsed) {
            messageCounts.merge(parsed.message, 1, Integer::sum);
            levelCounts.merge(parsed.level, 1, Integer::sum);
            label = label || parsed.label;
        }

        private SparseVector toVector(SimpleDrainParser drain, long bucketKey) {
            Map<Integer, Double> values = new TreeMap<>();
            int logCount = 0;
            for (Map.Entry<String, Integer> entry : messageCounts.entrySet()) {
                int template = drain.parse(entry.getKey());
                int count = entry.getValue();
                logCount += count;
                values.merge(TEMPLATE_OFFSET + template, (double) count, Double::sum);
            }
            values.put(LOG_COUNT, (double) logCount);
            values.put(INFO_COUNT, (double) levelCounts.getOrDefault("INFO", 0));
            values.put(WARNING_COUNT, (double) levelCounts.getOrDefault("WARNING", 0));
            values.put(ERROR_COUNT, (double) levelCounts.getOrDefault("ERROR", 0));
            values.put(FATAL_COUNT, (double) levelCounts.getOrDefault("FATAL", 0));
            values.put(FAILURE_COUNT, (double) levelCounts.getOrDefault("FAILURE", 0));
            return SparseVector.fromMap(values, "bgl:" + bucketKey + ":" + node);
        }
    }

    private static final class ParsedBgl {
        private final long epochSeconds;
        private final String node;
        private final String level;
        private final String message;
        private final boolean label;

        private ParsedBgl(long epochSeconds, String node, String level, String message, boolean label) {
            this.epochSeconds = epochSeconds;
            this.node = node;
            this.level = level;
            this.message = message;
            this.label = label;
        }

        private static ParsedBgl parse(String line) {
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
            String content = parts.length >= 10 ? parts[9] : "";
            String level = parts[8];
            String message = level + " " + parts[7] + " " + content;
            return new ParsedBgl(epochSeconds, parts[3], level, message, !"-".equals(parts[0]));
        }
    }

    private static final class SparseVector {
        private final int[] indices;
        private final double[] values;
        private final String key;

        private SparseVector(int[] indices, double[] values, String key) {
            this.indices = indices;
            this.values = values;
            this.key = key;
        }

        private static SparseVector fromDense(String[] parts, int offset, String key) {
            List<Integer> indexList = new ArrayList<>();
            List<Double> valueList = new ArrayList<>();
            for (int i = offset; i < parts.length; i++) {
                double value = Double.parseDouble(parts[i]);
                if (value != 0.0) {
                    indexList.add(i - offset);
                    valueList.add(value);
                }
            }
            return fromLists(indexList, valueList, key);
        }

        private static SparseVector fromMap(Map<Integer, Double> values, String key) {
            List<Integer> indexList = new ArrayList<>();
            List<Double> valueList = new ArrayList<>();
            for (Map.Entry<Integer, Double> entry : values.entrySet()) {
                if (entry.getValue() != 0.0) {
                    indexList.add(entry.getKey());
                    valueList.add(entry.getValue());
                }
            }
            return fromLists(indexList, valueList, key);
        }

        private static SparseVector fromLists(List<Integer> indexList, List<Double> valueList, String key) {
            int[] indices = new int[indexList.size()];
            double[] values = new double[indexList.size()];
            for (int i = 0; i < indices.length; i++) {
                indices[i] = indexList.get(i);
                values[i] = valueList.get(i);
            }
            return new SparseVector(indices, values, key);
        }

        private static double l2LogDistance(SparseVector left, SparseVector right) {
            int i = 0;
            int j = 0;
            double sum = 0.0;
            while (i < left.indices.length || j < right.indices.length) {
                if (j >= right.indices.length || i < left.indices.length && left.indices[i] < right.indices[j]) {
                    double value = Math.log1p(left.values[i]);
                    sum += value * value;
                    ++i;
                } else if (i >= left.indices.length || right.indices[j] < left.indices[i]) {
                    double value = Math.log1p(right.values[j]);
                    sum += value * value;
                    ++j;
                } else {
                    double delta = Math.log1p(left.values[i]) - Math.log1p(right.values[j]);
                    sum += delta * delta;
                    ++i;
                    ++j;
                }
            }
            return Math.sqrt(sum);
        }

        @Override
        public String toString() {
            return key;
        }
    }

    private static final class SimpleDrainParser {
        private final double similarityThreshold;
        private final Map<Integer, List<List<String>>> clustersByLength = new HashMap<>();
        private final Map<Integer, List<Integer>> idsByLength = new HashMap<>();
        private int nextId;

        private SimpleDrainParser(double similarityThreshold) {
            this.similarityThreshold = similarityThreshold;
        }

        private int parse(String message) {
            List<String> tokens = tokenize(message);
            List<List<String>> clusters = clustersByLength.computeIfAbsent(tokens.size(), ignored -> new ArrayList<>());
            List<Integer> ids = idsByLength.computeIfAbsent(tokens.size(), ignored -> new ArrayList<>());
            int best = -1;
            double bestSimilarity = -1;
            for (int i = 0; i < clusters.size(); i++) {
                double similarity = similarity(tokens, clusters.get(i));
                if (similarity > bestSimilarity) {
                    bestSimilarity = similarity;
                    best = i;
                }
            }
            if (best >= 0 && bestSimilarity >= similarityThreshold) {
                update(clusters.get(best), tokens);
                return ids.get(best);
            }
            int id = nextId++;
            clusters.add(new ArrayList<>(tokens));
            ids.add(id);
            return id;
        }

        private int templateCount() {
            return nextId;
        }

        private static List<String> tokenize(String message) {
            if (message == null || message.trim().isEmpty()) {
                return Collections.singletonList("<empty>");
            }
            String[] raw = message.toLowerCase(Locale.ROOT).split("\\s+");
            List<String> tokens = new ArrayList<>(raw.length);
            for (String token : raw) {
                tokens.add(hasDigit(token) ? "<*>" : trim(token));
            }
            tokens.removeIf(String::isEmpty);
            return tokens.isEmpty() ? Collections.singletonList("<empty>") : tokens;
        }

        private static boolean hasDigit(String token) {
            for (int i = 0; i < token.length(); i++) {
                if (Character.isDigit(token.charAt(i))) {
                    return true;
                }
            }
            return false;
        }

        private static String trim(String token) {
            int start = 0;
            int end = token.length() - 1;
            while (start <= end && !isTokenChar(token.charAt(start))) {
                ++start;
            }
            while (end >= start && !isTokenChar(token.charAt(end))) {
                --end;
            }
            return start > end ? "" : token.substring(start, end + 1);
        }

        private static boolean isTokenChar(char value) {
            return value >= 'a' && value <= 'z' || value == '<' || value == '>' || value == '*';
        }

        private static double similarity(List<String> tokens, List<String> template) {
            int comparable = 0;
            int exact = 0;
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

        private static void update(List<String> template, List<String> tokens) {
            for (int i = 0; i < template.size(); i++) {
                if (!template.get(i).equals(tokens.get(i))) {
                    template.set(i, "<*>");
                }
            }
        }
    }

    private static final class BucketRollup {
        private final long warmupBuckets;
        private long evaluated;
        private long positives;
        private long predicted;
        private long tp;
        private long fp;
        private long fn;
        private long tn;

        private BucketRollup(long warmupBuckets) {
            this.warmupBuckets = warmupBuckets;
        }

        private void record(long bucketOrdinal, boolean label, boolean prediction) {
            if (bucketOrdinal < warmupBuckets) {
                return;
            }
            ++evaluated;
            if (label) {
                ++positives;
            }
            if (prediction) {
                ++predicted;
            }
            if (label && prediction) {
                ++tp;
            } else if (!label && prediction) {
                ++fp;
            } else if (label) {
                ++fn;
            } else {
                ++tn;
            }
        }

        private Metrics toMetrics(String name) {
            Metrics metrics = new Metrics(name, 0);
            metrics.evaluated = evaluated;
            metrics.positives = positives;
            metrics.predicted = predicted;
            metrics.tp = tp;
            metrics.fp = fp;
            metrics.fn = fn;
            metrics.tn = tn;
            return metrics;
        }
    }

    private static final class Metrics {
        private final String name;
        private final long warmup;
        private long evaluated;
        private long positives;
        private long predicted;
        private long tp;
        private long fp;
        private long fn;
        private long tn;
        private double maxScore;
        private double maxThreshold;
        private double elapsedSeconds;
        private String state = "";

        private Metrics(String name, long warmup) {
            this.name = name;
            this.warmup = warmup;
        }

        private void record(long ordinal, boolean label, boolean prediction, double score, double threshold) {
            maxScore = Math.max(maxScore, score);
            maxThreshold = Math.max(maxThreshold, threshold);
            if (ordinal < warmup) {
                return;
            }
            ++evaluated;
            if (label) {
                ++positives;
            }
            if (prediction) {
                ++predicted;
            }
            if (label && prediction) {
                ++tp;
            } else if (!label && prediction) {
                ++fp;
            } else if (label) {
                ++fn;
            } else {
                ++tn;
            }
        }

        private double precision() {
            return tp + fp == 0 ? 0 : (double) tp / (tp + fp);
        }

        private double recall() {
            return tp + fn == 0 ? 0 : (double) tp / (tp + fn);
        }

        private double f1() {
            double precision = precision();
            double recall = recall();
            return precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
        }

        private static String header() {
            return "name,evaluated,positive,predicted,tp,fp,fn,tn,precision,recall,f1,max_score,max_threshold,elapsed_s,state";
        }

        private String toCsv() {
            return String.format(Locale.ROOT,
                    "%s,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%s", name, evaluated,
                    positives, predicted, tp, fp, fn, tn, precision(), recall(), f1(), maxScore, maxThreshold,
                    elapsedSeconds, state);
        }
    }

    private static final class Config {
        private String hdfsOccurrence;
        private String bglLog;
        private String output;
        private long bucketSeconds = 600;
        private int hdfsWarmup = 2000;
        private int bglWarmupBuckets = 80;
        private int capacity = 2000;
        private int doNotReclusterWithin = 1000;
        private int representatives = 3;
        private int maxAllowed = 10;
        private double zFactor = 3.0;
        private double lowerThreshold = 1.2;
        private double drainSimilarity = 0.5;
        private long seed = 42L;
        private boolean includeHdfsRcf;

        private static Config parse(String[] args) {
            Config config = new Config();
            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                if ("--hdfs-occurrence".equals(arg)) {
                    config.hdfsOccurrence = next(args, ++i, arg);
                } else if ("--bgl-log".equals(arg)) {
                    config.bglLog = next(args, ++i, arg);
                } else if ("--output".equals(arg)) {
                    config.output = next(args, ++i, arg);
                } else if ("--bucket-seconds".equals(arg)) {
                    config.bucketSeconds = Long.parseLong(next(args, ++i, arg));
                } else if ("--hdfs-warmup".equals(arg)) {
                    config.hdfsWarmup = Integer.parseInt(next(args, ++i, arg));
                } else if ("--bgl-warmup-buckets".equals(arg)) {
                    config.bglWarmupBuckets = Integer.parseInt(next(args, ++i, arg));
                } else if ("--capacity".equals(arg)) {
                    config.capacity = Integer.parseInt(next(args, ++i, arg));
                } else if ("--do-not-recluster-within".equals(arg)) {
                    config.doNotReclusterWithin = Integer.parseInt(next(args, ++i, arg));
                } else if ("--representatives".equals(arg)) {
                    config.representatives = Integer.parseInt(next(args, ++i, arg));
                } else if ("--max-allowed".equals(arg)) {
                    config.maxAllowed = Integer.parseInt(next(args, ++i, arg));
                } else if ("--z-factor".equals(arg)) {
                    config.zFactor = Double.parseDouble(next(args, ++i, arg));
                } else if ("--lower-threshold".equals(arg)) {
                    config.lowerThreshold = Double.parseDouble(next(args, ++i, arg));
                } else if ("--drain-similarity".equals(arg)) {
                    config.drainSimilarity = Double.parseDouble(next(args, ++i, arg));
                } else if ("--seed".equals(arg)) {
                    config.seed = Long.parseLong(next(args, ++i, arg));
                } else if ("--include-hdfs-rcf".equals(arg)) {
                    config.includeHdfsRcf = true;
                } else {
                    throw new IllegalArgumentException("unknown argument " + arg);
                }
            }
            return config;
        }

        private static String next(String[] args, int index, String flag) {
            if (index >= args.length) {
                throw new IllegalArgumentException(flag + " requires a value");
            }
            return args[index];
        }
    }
}
