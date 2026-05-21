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
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Builds production-style bucket metadata and rarity-evidence features.
 */
public final class ProductionBucketFeatureBuilder {

    private static final Charset RAW_LOG_CHARSET = StandardCharsets.ISO_8859_1;

    private ProductionBucketFeatureBuilder() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        for (String dataset : config.datasets) {
            int sampleModulo = "thunderbird".equals(dataset) ? config.thunderbirdSampleModulo : config.bglSampleModulo;
            String input = "thunderbird".equals(dataset) ? config.thunderbirdPath : config.bglPath;
            List<Bucket> buckets = loadBuckets(dataset, input, sampleModulo, config);
            int trainCount = Math.max(1, (int) Math.floor(config.trainFraction * buckets.size()));
            TrainStats stats = TrainStats.from(buckets, trainCount, config);
            writeMetadata(config.metadataPrefix + "_" + dataset + ".csv", buckets, trainCount);
            writeFeatures(config.featurePrefix + "_" + dataset + "_evidence.csv", buckets, trainCount, stats, "evidence",
                    0);
            writeFeatures(config.featurePrefix + "_" + dataset + "_b25e.csv", buckets, trainCount, stats, "b25e", 25);
            writeFeatures(config.featurePrefix + "_" + dataset + "_b50e.csv", buckets, trainCount, stats, "b50e", 50);
            printDistribution(dataset, buckets, trainCount);
        }
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
        long positives = buckets.stream().filter(bucket -> bucket.anomalousCount > 0).count();
        System.out.printf(Locale.ROOT,
                "loaded dataset=%s sampled_logs=%d buckets=%d positive_any_buckets=%d templates=%d%n", dataset, sampled,
                buckets.size(), positives, parser.templateCount());
        return buckets;
    }

    private static void writeMetadata(String output, List<Bucket> buckets, int trainCount) throws IOException {
        Path path = Paths.get(output);
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write("index,train,bucket_key,total_count,anomalous_count,anomalous_fraction,label_any");
            writer.newLine();
            for (int i = 0; i < buckets.size(); i++) {
                Bucket bucket = buckets.get(i);
                writer.write(String.format(Locale.ROOT, "%d,%d,%d,%d,%d,%.8f,%d", i, i < trainCount ? 1 : 0,
                        bucket.bucketKey, bucket.totalCount, bucket.anomalousCount,
                        bucket.totalCount == 0 ? 0.0 : bucket.anomalousCount / (double) bucket.totalCount,
                        bucket.anomalousCount > 0 ? 1 : 0));
                writer.newLine();
            }
        }
    }

    private static void writeFeatures(String output, List<Bucket> buckets, int trainCount, TrainStats stats, String name,
            int topK) throws IOException {
        Path path = Paths.get(output);
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            int dimensions = topK == 0 ? EvidenceStats.DIMENSIONS : topK + 1 + EvidenceStats.DIMENSIONS;
            writer.write("index,train,label");
            for (int i = 0; i < dimensions; i++) {
                writer.write(",f" + i);
            }
            writer.newLine();
            for (int row = 0; row < buckets.size(); row++) {
                double[] vector = vectorize(buckets.get(row), stats, topK);
                writer.write(Integer.toString(row));
                writer.write(row < trainCount ? ",1," : ",0,");
                writer.write(buckets.get(row).anomalousCount > 0 ? "1" : "0");
                for (double value : vector) {
                    writer.write(',');
                    writer.write(String.format(Locale.ROOT, "%.8f", value));
                }
                writer.newLine();
            }
        }
        System.out.printf(Locale.ROOT, "wrote feature=%s output=%s%n", name, output);
    }

    private static double[] vectorize(Bucket bucket, TrainStats stats, int topK) {
        EvidenceStats evidence = EvidenceStats.of(bucket, stats);
        if (topK == 0) {
            return evidence.toVector();
        }
        double[] vector = new double[topK + 1 + EvidenceStats.DIMENSIONS];
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
        System.arraycopy(evidence.toVector(), 0, vector, topK + 1, EvidenceStats.DIMENSIONS);
        return vector;
    }

    private static void printDistribution(String dataset, List<Bucket> buckets, int trainCount) {
        System.out.printf(Locale.ROOT, "distribution dataset=%s split=test buckets=%d%n", dataset,
                buckets.size() - trainCount);
        for (int k : new int[] { 1, 2, 3, 5, 10 }) {
            long hits = 0;
            for (int i = trainCount; i < buckets.size(); i++) {
                if (buckets.get(i).anomalousCount >= k) {
                    ++hits;
                }
            }
            System.out.printf(Locale.ROOT, "  count>=%d rate=%.6f buckets=%d%n", k,
                    hits / (double) Math.max(1, buckets.size() - trainCount), hits);
        }
        for (double threshold : new double[] { 0.001, 0.005, 0.01 }) {
            long hits = 0;
            for (int i = trainCount; i < buckets.size(); i++) {
                Bucket bucket = buckets.get(i);
                double fraction = bucket.totalCount == 0 ? 0.0 : bucket.anomalousCount / (double) bucket.totalCount;
                if (fraction >= threshold) {
                    ++hits;
                }
            }
            System.out.printf(Locale.ROOT, "  fraction>=%.4f rate=%.6f buckets=%d%n", threshold,
                    hits / (double) Math.max(1, buckets.size() - trainCount), hits);
        }
    }

    private static final class EvidenceStats {
        private static final int DIMENSIONS = 14;

        private final double[] values;

        private EvidenceStats(double[] values) {
            this.values = values;
        }

        private static EvidenceStats of(Bucket bucket, TrainStats stats) {
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
                dominant = Math.max(dominant, count);
                if (!stats.trainCounts.containsKey(eventId)) {
                    oov += count;
                }
                if (stats.rareEvents.contains(eventId)) {
                    rare += count;
                }
                double rarity = stats.rarity(eventId);
                rarityCounts.add(new RarityCount(rarity, count));
                maxRarity = Math.max(maxRarity, rarity);
                sumRarity += rarity * count;
                clippedRarity += Math.min(20.0, rarity) * count;
                double p = total == 0 ? 0.0 : count / (double) total;
                if (p > 0.0) {
                    entropy -= p * Math.log(p);
                }
            }
            for (Map.Entry<Integer, Integer> entry : stats.topEventIndex.entrySet()) {
                if (entry.getValue() < 50) {
                    int eventId = entry.getKey();
                    double bucketRate = total == 0 ? 0.0 : bucket.eventCounts.getOrDefault(eventId, 0) / (double) total;
                    topShift += Math.abs(bucketRate - stats.trainRate(eventId));
                }
            }
            double[] vector = new double[] { Math.log1p(total), Math.log1p(bucket.eventCounts.size()),
                    Math.log1p(rare), Math.log1p(oov), total == 0 ? 0.0 : rare / (double) total,
                    total == 0 ? 0.0 : oov / (double) total, maxRarity, total == 0 ? 0.0 : sumRarity / total,
                    weightedQuantile(rarityCounts, total, 0.95), weightedQuantile(rarityCounts, total, 0.99),
                    Math.log1p(clippedRarity), entropy, total == 0 ? 0.0 : dominant / (double) total, topShift };
            return new EvidenceStats(vector);
        }

        private static double weightedQuantile(List<RarityCount> values, int total, double quantile) {
            if (total <= 0 || values.isEmpty()) {
                return 0.0;
            }
            values.sort(Comparator.comparingDouble(value -> value.rarity));
            int target = Math.max(1, (int) Math.ceil(total * quantile));
            int seen = 0;
            for (RarityCount value : values) {
                seen += value.count;
                if (seen >= target) {
                    return value.rarity;
                }
            }
            return values.get(values.size() - 1).rarity;
        }

        private double[] toVector() {
            return values.clone();
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
            sorted.sort((a, b) -> Integer.compare(b.getValue(), a.getValue()));
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

        private double trainRate(int eventId) {
            return trainTotal == 0 ? 0.0 : trainCounts.getOrDefault(eventId, 0) / (double) trainTotal;
        }
    }

    private static final class Bucket {
        private final long bucketKey;
        private final Map<Integer, Integer> eventCounts = new HashMap<>();
        private int totalCount;
        private int anomalousCount;

        private Bucket(long bucketKey) {
            this.bucketKey = bucketKey;
        }

        private void add(int eventId, boolean anomalous) {
            eventCounts.put(eventId, eventCounts.getOrDefault(eventId, 0) + 1);
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

    private static final class Config {
        private final List<String> datasets;
        private final String bglPath;
        private final String thunderbirdPath;
        private final String featurePrefix;
        private final String metadataPrefix;
        private final int bucketSeconds;
        private final int bglSampleModulo;
        private final int thunderbirdSampleModulo;
        private final double trainFraction;
        private final int topK;
        private final int rareEventMaxTrainCount;
        private final double drainSimilarity;

        private Config(List<String> datasets, String bglPath, String thunderbirdPath, String featurePrefix,
                String metadataPrefix, int bucketSeconds, int bglSampleModulo, int thunderbirdSampleModulo,
                double trainFraction, int topK, int rareEventMaxTrainCount, double drainSimilarity) {
            this.datasets = datasets;
            this.bglPath = bglPath;
            this.thunderbirdPath = thunderbirdPath;
            this.featurePrefix = featurePrefix;
            this.metadataPrefix = metadataPrefix;
            this.bucketSeconds = bucketSeconds;
            this.bglSampleModulo = bglSampleModulo;
            this.thunderbirdSampleModulo = thunderbirdSampleModulo;
            this.trainFraction = trainFraction;
            this.topK = topK;
            this.rareEventMaxTrainCount = rareEventMaxTrainCount;
            this.drainSimilarity = drainSimilarity;
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
                    values.getOrDefault("feature-prefix", "benchmark-results/log-ad/production_bucket_features"),
                    values.getOrDefault("metadata-prefix", "benchmark-results/log-ad/bucket_metadata"),
                    Integer.parseInt(values.getOrDefault("bucket-seconds", "600")),
                    Integer.parseInt(values.getOrDefault("bgl-sample-modulo", "1")),
                    Integer.parseInt(values.getOrDefault("thunderbird-sample-modulo", "10")),
                    Double.parseDouble(values.getOrDefault("train-fraction", "0.10")),
                    Integer.parseInt(values.getOrDefault("top-k", "50")),
                    Integer.parseInt(values.getOrDefault("rare-max-train-count", "5")),
                    Double.parseDouble(values.getOrDefault("drain-similarity", "0.5")));
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
