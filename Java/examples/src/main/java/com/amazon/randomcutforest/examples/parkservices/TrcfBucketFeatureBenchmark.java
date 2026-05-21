/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

package com.amazon.randomcutforest.examples.parkservices;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
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
 * Bucket-level TRCF benchmark over Drain parsed-event count vectors.
 */
public final class TrcfBucketFeatureBenchmark {

    private static final Charset RAW_LOG_CHARSET = StandardCharsets.ISO_8859_1;

    private TrcfBucketFeatureBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        List<Result> results = new ArrayList<>();
        for (String dataset : config.datasets) {
            int sampleModulo = "thunderbird".equals(dataset) ? config.thunderbirdSampleModulo : config.bglSampleModulo;
            String input = "thunderbird".equals(dataset) ? config.thunderbirdPath : config.bglPath;
            List<Bucket> buckets = loadBuckets(dataset, input, sampleModulo, config);
            int trainCount = Math.max(1, (int) Math.floor(config.trainFraction * buckets.size()));
            TrainStats stats = TrainStats.from(buckets, trainCount, config);
            for (Representation representation : Representation.values()) {
                FeatureData data = buildFeatureData(buckets, trainCount, stats, representation, config);
                writeFeatures(config.outputPrefix + "_" + dataset + "_" + representation.name().toLowerCase(Locale.ROOT)
                        + ".csv", data);
                results.add(runTrcf(dataset, representation, data, trainCount, config));
            }
        }
        writeResults(config.output, results);
        printResults(results);
    }

    private static List<Bucket> loadBuckets(String dataset, String input, int sampleModulo, Config config)
            throws IOException {
        System.out.printf(Locale.ROOT, "loading dataset=%s input=%s bucket_seconds=%d sample_modulo=%d%n", dataset,
                input, config.bucketSeconds, sampleModulo);
        SimpleDrainParser parser = new SimpleDrainParser(config.drainSimilarity);
        List<Bucket> buckets = new ArrayList<>();
        Bucket current = null;
        long originalLine = 0;
        long sampled = 0;
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
                    buckets.add(current);
                }
                int eventId = parser.parse(normalize(parsed.message));
                current.add(eventId, parsed.label);
                ++sampled;
                if (sampled % 5_000_000L == 0) {
                    System.out.printf(Locale.ROOT, "  %s sampled=%d original=%d buckets=%d templates=%d%n", dataset,
                            sampled, originalLine, buckets.size(), parser.templateCount());
                }
            }
        }
        long positives = buckets.stream().filter(bucket -> bucket.label).count();
        System.out.printf(Locale.ROOT, "loaded dataset=%s sampled_logs=%d buckets=%d positive_buckets=%d templates=%d%n",
                dataset, sampled, buckets.size(), positives, parser.templateCount());
        return buckets;
    }

    private static FeatureData buildFeatureData(List<Bucket> buckets, int trainCount, TrainStats stats,
            Representation representation, Config config) {
        int dimensions = representation.dimensions(config);
        double[][] values = new double[buckets.size()][dimensions];
        boolean[] labels = new boolean[buckets.size()];
        for (int i = 0; i < buckets.size(); i++) {
            labels[i] = buckets.get(i).label;
            values[i] = representation.vectorize(buckets.get(i), stats, config);
        }
        return new FeatureData(representation.name(), values, labels, trainCount);
    }

    private static Result runTrcf(String dataset, Representation representation, FeatureData data, int trainCount,
            Config config) {
        int outputAfter = Math.max(1, Math.min(config.outputAfter, Math.max(1, trainCount / 2)));
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder()
                .dimensions(data.values[0].length * config.shingleSize)
                .shingleSize(config.shingleSize).sampleSize(config.sampleSize).numberOfTrees(config.numberOfTrees)
                .randomSeed(config.seed).outputAfter(outputAfter).transformMethod(TransformMethod.NORMALIZE)
                .scoringStrategy(ScoringStrategy.MULTI_MODE_RECALL).anomalyRate(config.anomalyRate).autoAdjust(true)
                .build();
        Metrics metrics = new Metrics();
        for (int i = 0; i < data.values.length; i++) {
            AnomalyDescriptor result = forest.process(data.values[i], i);
            if (i >= trainCount) {
                boolean prediction = result.getAnomalyGrade() > 0.0;
                metrics.add(data.labels[i], prediction, result.getRCFScore());
            }
        }
        return metrics.result(dataset, "TRCF + " + representation.displayName, data.values.length, trainCount);
    }

    private static void writeFeatures(String output, FeatureData data) throws IOException {
        Path path = Paths.get(output);
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write("index,train,label");
            for (int i = 0; i < data.values[0].length; i++) {
                writer.write(",f" + i);
            }
            writer.newLine();
            for (int row = 0; row < data.values.length; row++) {
                writer.write(Integer.toString(row));
                writer.write(row < data.trainCount ? ",1," : ",0,");
                writer.write(data.labels[row] ? "1" : "0");
                for (double value : data.values[row]) {
                    writer.write(',');
                    writer.write(String.format(Locale.ROOT, "%.8f", value));
                }
                writer.newLine();
            }
        }
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

    private static void printResults(List<Result> results) {
        System.out.println(Result.header());
        for (Result result : results) {
            System.out.println(result.toCsv());
        }
    }

    private enum Representation {
        A("Representation A rarity-only") {
            @Override
            int dimensions(Config config) {
                return 9;
            }

            @Override
            double[] vectorize(Bucket bucket, TrainStats stats, Config config) {
                BasicStats basic = BasicStats.of(bucket, stats);
                return new double[] { Math.log1p(basic.totalCount), basic.uniqueEventCount, basic.rareEventCount,
                        basic.oovCount, basic.maxRarity, basic.sumRarity, basic.avgRarity, basic.entropy,
                        basic.dominantEventRatio };
            }
        },
        B("Representation B top-50 counts + rarity") {
            @Override
            int dimensions(Config config) {
                return config.topK + 5;
            }

            @Override
            double[] vectorize(Bucket bucket, TrainStats stats, Config config) {
                BasicStats basic = BasicStats.of(bucket, stats);
                double[] vector = new double[config.topK + 5];
                int other = 0;
                for (Map.Entry<Integer, Integer> entry : bucket.eventCounts.entrySet()) {
                    Integer index = stats.topEventIndex.get(entry.getKey());
                    if (index != null && index < config.topK) {
                        vector[index] = Math.log1p(entry.getValue());
                    } else {
                        other += entry.getValue();
                    }
                }
                int offset = config.topK;
                vector[offset] = Math.log1p(other);
                vector[offset + 1] = Math.log1p(basic.rareEventCount);
                vector[offset + 2] = Math.log1p(basic.oovCount);
                vector[offset + 3] = basic.entropy;
                vector[offset + 4] = Math.log1p(basic.totalCount);
                return vector;
            }
        },
        C("Representation C hash buckets") {
            @Override
            int dimensions(Config config) {
                return config.hashBuckets + 4;
            }

            @Override
            double[] vectorize(Bucket bucket, TrainStats stats, Config config) {
                BasicStats basic = BasicStats.of(bucket, stats);
                double[] vector = new double[config.hashBuckets + 4];
                for (Map.Entry<Integer, Integer> entry : bucket.eventCounts.entrySet()) {
                    int index = positiveHash("E" + entry.getKey()) % config.hashBuckets;
                    vector[index] += entry.getValue();
                }
                for (int i = 0; i < config.hashBuckets; i++) {
                    vector[i] = Math.log1p(vector[i]);
                }
                int offset = config.hashBuckets;
                vector[offset] = Math.log1p(basic.rareEventCount);
                vector[offset + 1] = Math.log1p(basic.oovCount);
                vector[offset + 2] = basic.entropy;
                vector[offset + 3] = Math.log1p(basic.totalCount);
                return vector;
            }
        };

        private final String displayName;

        Representation(String displayName) {
            this.displayName = displayName;
        }

        abstract int dimensions(Config config);

        abstract double[] vectorize(Bucket bucket, TrainStats stats, Config config);
    }

    private static final class BasicStats {
        private final int totalCount;
        private final int uniqueEventCount;
        private final int rareEventCount;
        private final int oovCount;
        private final double maxRarity;
        private final double sumRarity;
        private final double avgRarity;
        private final double entropy;
        private final double dominantEventRatio;

        private BasicStats(int totalCount, int uniqueEventCount, int rareEventCount, int oovCount, double maxRarity,
                double sumRarity, double avgRarity, double entropy, double dominantEventRatio) {
            this.totalCount = totalCount;
            this.uniqueEventCount = uniqueEventCount;
            this.rareEventCount = rareEventCount;
            this.oovCount = oovCount;
            this.maxRarity = maxRarity;
            this.sumRarity = sumRarity;
            this.avgRarity = avgRarity;
            this.entropy = entropy;
            this.dominantEventRatio = dominantEventRatio;
        }

        private static BasicStats of(Bucket bucket, TrainStats stats) {
            int total = bucket.totalCount;
            int rare = 0;
            int oov = 0;
            int dominant = 0;
            double maxRarity = 0.0;
            double sumRarity = 0.0;
            double entropy = 0.0;
            for (Map.Entry<Integer, Integer> entry : bucket.eventCounts.entrySet()) {
                int eventId = entry.getKey();
                int count = entry.getValue();
                dominant = Math.max(dominant, count);
                if (!stats.trainCounts.containsKey(eventId)) {
                    oov += count;
                }
                if (stats.rareEvents.contains(eventId)) {
                    rare += count;
                }
                double rarity = stats.rarity(eventId);
                maxRarity = Math.max(maxRarity, rarity);
                sumRarity += rarity * count;
                double p = total == 0 ? 0.0 : count / (double) total;
                if (p > 0.0) {
                    entropy -= p * Math.log(p);
                }
            }
            double avgRarity = total == 0 ? 0.0 : sumRarity / total;
            double dominantRatio = total == 0 ? 0.0 : dominant / (double) total;
            return new BasicStats(total, bucket.eventCounts.size(), rare, oov, maxRarity, sumRarity, avgRarity, entropy,
                    dominantRatio);
        }
    }

    private static final class TrainStats {
        private final Map<Integer, Integer> trainCounts;
        private final Set<Integer> rareEvents;
        private final Map<Integer, Integer> topEventIndex;
        private final int trainTotal;
        private final int vocabularySize;

        private TrainStats(Map<Integer, Integer> trainCounts, Set<Integer> rareEvents, Map<Integer, Integer> topEventIndex,
                int trainTotal, int vocabularySize) {
            this.trainCounts = trainCounts;
            this.rareEvents = rareEvents;
            this.topEventIndex = topEventIndex;
            this.trainTotal = trainTotal;
            this.vocabularySize = vocabularySize;
        }

        private static TrainStats from(List<Bucket> buckets, int trainCount, Config config) {
            Map<Integer, Integer> counts = new HashMap<>();
            int total = 0;
            for (int i = 0; i < trainCount; i++) {
                for (Map.Entry<Integer, Integer> entry : buckets.get(i).eventCounts.entrySet()) {
                    counts.put(entry.getKey(), counts.getOrDefault(entry.getKey(), 0) + entry.getValue());
                    total += entry.getValue();
                }
            }
            Set<Integer> rare = new HashSet<>();
            for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
                if (entry.getValue() <= config.rareEventMaxTrainCount) {
                    rare.add(entry.getKey());
                }
            }
            List<Map.Entry<Integer, Integer>> sorted = new ArrayList<>(counts.entrySet());
            sorted.sort(Collections.reverseOrder(Comparator.comparingInt(Map.Entry::getValue)));
            Map<Integer, Integer> topIndex = new HashMap<>();
            for (int i = 0; i < Math.min(config.topK, sorted.size()); i++) {
                topIndex.put(sorted.get(i).getKey(), i);
            }
            return new TrainStats(counts, rare, topIndex, total, Math.max(1, counts.size()));
        }

        private double rarity(int eventId) {
            int count = trainCounts.getOrDefault(eventId, 0);
            double probability = (count + 1.0) / (trainTotal + vocabularySize + 1.0);
            return -Math.log(probability);
        }
    }

    private static final class Metrics {
        private final List<ScoredLabel> rows = new ArrayList<>();
        private long tp;
        private long fp;
        private long fn;
        private long tn;
        private long positives;
        private long predicted;

        private void add(boolean label, boolean prediction, double score) {
            rows.add(new ScoredLabel(score, label));
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

        private Result result(String dataset, String model, long records, long trainRecords) {
            int k = (int) positives;
            long hits = topKHits(k);
            double precisionAtK = k == 0 ? 0.0 : hits / (double) k;
            double recallAtK = positives == 0 ? 0.0 : hits / (double) positives;
            return new Result(dataset, model, records, trainRecords, rows.size(), positives, predicted, tp, fp, fn, tn,
                    precision(), recall(), f1(), k, precisionAtK, recallAtK);
        }

        private long topKHits(int k) {
            if (k <= 0) {
                return 0;
            }
            List<ScoredLabel> sorted = new ArrayList<>(rows);
            sorted.sort(Collections.reverseOrder(Comparator.comparingDouble(row -> row.score)));
            long hits = 0;
            for (int i = 0; i < Math.min(k, sorted.size()); i++) {
                if (sorted.get(i).label) {
                    ++hits;
                }
            }
            return hits;
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
            return p + r == 0 ? 0.0 : 2 * p * r / (p + r);
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

    private static final class Bucket {
        private final long bucketKey;
        private final Map<Integer, Integer> eventCounts = new HashMap<>();
        private int totalCount;
        private boolean label;

        private Bucket(long bucketKey) {
            this.bucketKey = bucketKey;
        }

        private void add(int eventId, boolean anomalous) {
            eventCounts.put(eventId, eventCounts.getOrDefault(eventId, 0) + 1);
            ++totalCount;
            label |= anomalous;
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
        String message = parts[8] + (parts.length >= 10 ? " " + parts[9] : "");
        return new ParsedLine(epochSeconds, label, message);
    }

    private static final class ParsedLine {
        private final long epochSeconds;
        private final boolean label;
        private final String message;

        private ParsedLine(long epochSeconds, boolean label, String message) {
            this.epochSeconds = epochSeconds;
            this.label = label;
            this.message = message;
        }
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

    private static long countLines(Path path) throws IOException {
        long lines = 0;
        try (BufferedReader reader = Files.newBufferedReader(path, RAW_LOG_CHARSET)) {
            while (reader.readLine() != null) {
                ++lines;
            }
        }
        return lines;
    }

    private static final class SimpleDrainParser {
        private final double similarityThreshold;
        private final Map<Integer, List<DrainCluster>> clustersByLength = new HashMap<>();
        private int nextId;

        private SimpleDrainParser(double similarityThreshold) {
            this.similarityThreshold = similarityThreshold;
        }

        private int parse(String message) {
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
            DrainCluster cluster = new DrainCluster(nextId++, tokens);
            clustersByLength.computeIfAbsent(tokens.size(), ignored -> new ArrayList<>()).add(cluster);
            return cluster.id;
        }

        private int templateCount() {
            return nextId;
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

    private static final class DrainCluster {
        private final int id;
        private final List<String> template;

        private DrainCluster(int id, List<String> template) {
            this.id = id;
            this.template = new ArrayList<>(template);
        }
    }

    private static final class Result {
        private final String dataset;
        private final String model;
        private final long records;
        private final long trainRecords;
        private final long testRecords;
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

        private Result(String dataset, String model, long records, long trainRecords, long testRecords, long positives,
                long predicted, long tp, long fp, long fn, long tn, double precision, double recall, double f1, int k,
                double precisionAtK, double recallAtK) {
            this.dataset = dataset;
            this.model = model;
            this.records = records;
            this.trainRecords = trainRecords;
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
        }

        private static String header() {
            return "dataset,model,records,train_records,test_records,positives,predicted,tp,fp,fn,tn,precision,recall,f1,k,precision_at_k,recall_at_k";
        }

        private String toCsv() {
            return String.format(Locale.ROOT,
                    "%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d,%.6f,%.6f", dataset, model, records,
                    trainRecords, testRecords, positives, predicted, tp, fp, fn, tn, precision, recall, f1, k,
                    precisionAtK, recallAtK);
        }
    }

    private static final class Config {
        private final List<String> datasets;
        private final String bglPath;
        private final String thunderbirdPath;
        private final String output;
        private final String outputPrefix;
        private final int bucketSeconds;
        private final int bglSampleModulo;
        private final int thunderbirdSampleModulo;
        private final double trainFraction;
        private final int topK;
        private final int hashBuckets;
        private final int rareEventMaxTrainCount;
        private final int numberOfTrees;
        private final int sampleSize;
        private final int shingleSize;
        private final int outputAfter;
        private final double anomalyRate;
        private final double drainSimilarity;
        private final long seed;

        private Config(List<String> datasets, String bglPath, String thunderbirdPath, String output, String outputPrefix,
                int bucketSeconds, int bglSampleModulo, int thunderbirdSampleModulo, double trainFraction, int topK,
                int hashBuckets, int rareEventMaxTrainCount, int numberOfTrees, int sampleSize, int shingleSize,
                int outputAfter, double anomalyRate, double drainSimilarity, long seed) {
            this.datasets = datasets;
            this.bglPath = bglPath;
            this.thunderbirdPath = thunderbirdPath;
            this.output = output;
            this.outputPrefix = outputPrefix;
            this.bucketSeconds = bucketSeconds;
            this.bglSampleModulo = bglSampleModulo;
            this.thunderbirdSampleModulo = thunderbirdSampleModulo;
            this.trainFraction = trainFraction;
            this.topK = topK;
            this.hashBuckets = hashBuckets;
            this.rareEventMaxTrainCount = rareEventMaxTrainCount;
            this.numberOfTrees = numberOfTrees;
            this.sampleSize = sampleSize;
            this.shingleSize = shingleSize;
            this.outputAfter = outputAfter;
            this.anomalyRate = anomalyRate;
            this.drainSimilarity = drainSimilarity;
            this.seed = seed;
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
            return new Config(split(values.getOrDefault("datasets", "bgl,thunderbird")),
                    values.getOrDefault("bgl", root + "/BGL/BGL.log"),
                    values.getOrDefault("thunderbird", root + "/Thunderbird/Thunderbird.log"),
                    values.getOrDefault("output", "benchmark-results/log-ad/trcf_bucket_metrics.csv"),
                    values.getOrDefault("output-prefix", "benchmark-results/log-ad/bucket_features"),
                    Integer.parseInt(values.getOrDefault("bucket-seconds", "600")),
                    Integer.parseInt(values.getOrDefault("bgl-sample-modulo", "1")),
                    Integer.parseInt(values.getOrDefault("thunderbird-sample-modulo", "10")),
                    Double.parseDouble(values.getOrDefault("train-fraction", "0.10")),
                    Integer.parseInt(values.getOrDefault("top-k", "50")),
                    Integer.parseInt(values.getOrDefault("hash-buckets", "64")),
                    Integer.parseInt(values.getOrDefault("rare-max-train-count", "5")),
                    Integer.parseInt(values.getOrDefault("trees", "50")),
                    Integer.parseInt(values.getOrDefault("sample-size", "256")),
                    Integer.parseInt(values.getOrDefault("shingle-size", "4")),
                    Integer.parseInt(values.getOrDefault("output-after", "32")),
                    Double.parseDouble(values.getOrDefault("anomaly-rate", "0.02")),
                    Double.parseDouble(values.getOrDefault("drain-similarity", "0.5")),
                    Long.parseLong(values.getOrDefault("seed", "42")));
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
