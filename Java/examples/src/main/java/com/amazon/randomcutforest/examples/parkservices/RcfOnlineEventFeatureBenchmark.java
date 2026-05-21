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
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import com.amazon.randomcutforest.RandomCutForest;

/**
 * Online RCF benchmark for line-level parsed-event features. This is intended
 * for BGL/Thunderbird where the paper evaluates event-level labels. Each test
 * record is scored first and then used to update the forest.
 */
public final class RcfOnlineEventFeatureBenchmark {

    private static final Charset RAW_LOG_CHARSET = StandardCharsets.ISO_8859_1;
    private static final long PAPER_THUNDERBIRD_LINES = 211_212_192L;

    private RcfOnlineEventFeatureBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        List<Result> all = new ArrayList<>();
        for (String dataset : config.datasets) {
            long started = System.nanoTime();
            if ("bgl".equals(dataset)) {
                all.addAll(runLineDataset("BGL", config.bglPath, countLines(Paths.get(config.bglPath)), config));
            } else if ("thunderbird".equals(dataset)) {
                all.addAll(runLineDataset("Thunderbird", config.thunderbirdPath, PAPER_THUNDERBIRD_LINES, config));
            } else {
                throw new IllegalArgumentException("unsupported line-level dataset: " + dataset);
            }
            System.out.printf(Locale.ROOT, "finished dataset=%s elapsed_s=%.3f%n", dataset,
                    (System.nanoTime() - started) / 1_000_000_000.0);
        }
        writeResults(config.output, all);
        printResults(all);
    }

    private static List<Result> runLineDataset(String dataset, String input, long totalLines, Config config)
            throws IOException {
        System.out.printf(Locale.ROOT, "running dataset=%s input=%s sample_modulo=%d trees=%d sample_size=%d%n",
                dataset, input, config.sampleModulo, config.numberOfTrees, config.sampleSize);
        long trainCutoff = (long) Math.floor(totalLines * config.trainFraction);
        SimpleDrainParser parser = new SimpleDrainParser(config.drainSimilarity);
        List<LineModel> models = createModels(dataset, config);
        long sampled = 0;
        long originalLine = 0;
        long trainLines = 0;
        long testLines = 0;
        long positives = 0;

        try (BufferedReader reader = Files.newBufferedReader(Paths.get(input), RAW_LOG_CHARSET)) {
            String line;
            while ((line = reader.readLine()) != null) {
                ++originalLine;
                if (config.sampleModulo > 1 && positiveHash(Long.toString(originalLine)) % config.sampleModulo != 0) {
                    continue;
                }
                ParsedLine parsed = parseBglLikeLine(line);
                if (parsed == null || parsed.message.isEmpty()) {
                    continue;
                }
                boolean train = originalLine <= trainCutoff;
                int eventId = parser.parse(normalize(parsed.message));
                for (LineModel model : models) {
                    model.process(eventId, parsed.label, train);
                }
                ++sampled;
                if (train) {
                    ++trainLines;
                } else {
                    ++testLines;
                    if (parsed.label) {
                        ++positives;
                    }
                }
                if (sampled % 5_000_000L == 0) {
                    System.out.printf(Locale.ROOT, "  %s sampled=%d original=%d templates=%d%n", dataset, sampled,
                            originalLine, parser.templateCount());
                }
            }
        }
        for (LineModel model : models) {
            model.finish();
        }
        List<Result> results = new ArrayList<>(models.size());
        for (LineModel model : models) {
            results.add(model.result(sampled, trainLines, testLines, positives, testLines - positives));
        }
        return results;
    }

    private static List<LineModel> createModels(String dataset, Config config) {
        List<LineModel> models = new ArrayList<>();
        if (config.models.contains("onehot")) {
            models.add(new VectorLineModel(dataset, "RCF online parsed-event onehot", config.numberOfTrees,
                    config.sampleSize, config.seed, new OneHotEncoder(config.hashDimensions)));
        }
        if (config.models.contains("rarity")) {
            models.add(new VectorLineModel(dataset, "RCF online event rarity features", config.numberOfTrees,
                    config.sampleSize, config.seed + 1, new RarityEncoder()));
        }
        if (config.models.contains("embedding")) {
            models.add(new VectorLineModel(dataset, "RCF online event embedding", config.numberOfTrees,
                    config.sampleSize, config.seed + 2, new EmbeddingEncoder(config.embeddingDimensions)));
        }
        if (config.models.contains("embedding_rarity")) {
            models.add(new VectorLineModel(dataset, "RCF online event embedding+rarity", config.numberOfTrees,
                    config.sampleSize, config.seed + 3, new EmbeddingRarityEncoder(config.embeddingDimensions)));
        }
        if (config.models.contains("window")) {
            models.add(new WindowCountModel(dataset, "RCF online template-count window", config.numberOfTrees,
                    config.sampleSize, config.seed + 4, config.hashDimensions, config.windowSize));
        }
        return models;
    }

    private interface LineModel {
        void process(int eventId, boolean label, boolean train);

        void finish();

        Result result(long sampledLines, long trainLines, long testLines, long positives, long negatives);
    }

    private interface Encoder {
        int dimensions();

        void encode(int eventId, float[] vector);

        void update(int eventId);
    }

    private static final class VectorLineModel implements LineModel {
        private final String dataset;
        private final String name;
        private final Encoder encoder;
        private final RandomCutForest forest;
        private final ScoreBuffer scores = new ScoreBuffer();
        private final float[] vector;

        private VectorLineModel(String dataset, String name, int trees, int sampleSize, long seed, Encoder encoder) {
            this.dataset = dataset;
            this.name = name;
            this.encoder = encoder;
            this.vector = new float[encoder.dimensions()];
            this.forest = RandomCutForest.builder().compact(true).dimensions(encoder.dimensions()).numberOfTrees(trees)
                    .sampleSize(sampleSize).randomSeed(seed).parallelExecutionEnabled(false).build();
        }

        @Override
        public void process(int eventId, boolean label, boolean train) {
            Arrays.fill(vector, 0.0f);
            encoder.encode(eventId, vector);
            if (!train) {
                scores.add((float) forest.getAnomalyScore(vector), label);
            }
            forest.update(vector);
            encoder.update(eventId);
        }

        @Override
        public void finish() {
            // no pending state
        }

        @Override
        public Result result(long sampledLines, long trainLines, long testLines, long positives, long negatives) {
            return new Result(dataset, name, sampledLines, trainLines, scores.size, scores.positives, scores.negatives,
                    scores.auc());
        }
    }

    private static final class WindowCountModel implements LineModel {
        private final String dataset;
        private final String name;
        private final int dimensions;
        private final int windowSize;
        private final RandomCutForest forest;
        private final ScoreBuffer scores = new ScoreBuffer();
        private final float[] vector;
        private boolean currentLabel;
        private boolean currentTrain = true;
        private int count;
        private long trainWindows;

        private WindowCountModel(String dataset, String name, int trees, int sampleSize, long seed, int dimensions,
                int windowSize) {
            this.dataset = dataset;
            this.name = name + " n=" + windowSize;
            this.dimensions = dimensions;
            this.windowSize = windowSize;
            this.vector = new float[dimensions];
            this.forest = RandomCutForest.builder().compact(true).dimensions(dimensions).numberOfTrees(trees)
                    .sampleSize(sampleSize).randomSeed(seed).parallelExecutionEnabled(false).build();
        }

        @Override
        public void process(int eventId, boolean label, boolean train) {
            if (count > 0 && train != currentTrain) {
                flush();
            }
            currentTrain = train;
            vector[positiveHash("E" + eventId) % dimensions] += 1.0f;
            currentLabel |= label;
            ++count;
            if (count >= windowSize) {
                flush();
            }
        }

        @Override
        public void finish() {
            if (count > 0) {
                flush();
            }
        }

        private void flush() {
            normalize(vector);
            if (!currentTrain) {
                scores.add((float) forest.getAnomalyScore(vector), currentLabel);
            } else {
                ++trainWindows;
            }
            forest.update(vector);
            Arrays.fill(vector, 0.0f);
            currentLabel = false;
            count = 0;
        }

        @Override
        public Result result(long sampledLines, long trainLines, long testLines, long positives, long negatives) {
            return new Result(dataset, name, trainWindows + scores.size, trainWindows, scores.size, scores.positives,
                    scores.negatives, scores.auc());
        }
    }

    private static final class OneHotEncoder implements Encoder {
        private final int dimensions;

        private OneHotEncoder(int dimensions) {
            this.dimensions = dimensions;
        }

        @Override
        public int dimensions() {
            return dimensions;
        }

        @Override
        public void encode(int eventId, float[] vector) {
            vector[positiveHash("E" + eventId) % dimensions] = 1.0f;
        }

        @Override
        public void update(int eventId) {
            // stateless
        }
    }

    private static final class RarityEncoder implements Encoder {
        private final Map<Integer, Long> counts = new HashMap<>();
        private long seen;

        @Override
        public int dimensions() {
            return 8;
        }

        @Override
        public void encode(int eventId, float[] vector) {
            long count = counts.getOrDefault(eventId, 0L);
            double denominator = seen + Math.max(1, counts.size()) + 1.0;
            double probability = (count + 1.0) / denominator;
            vector[0] = (float) Math.log1p(seen);
            vector[1] = (float) Math.log1p(count);
            vector[2] = (float) -Math.log(probability);
            vector[3] = count == 0 ? 1.0f : 0.0f;
            vector[4] = (positiveHash("h0:" + eventId) % 10_000) / 10_000.0f;
            vector[5] = (positiveHash("h1:" + eventId) % 10_000) / 10_000.0f;
            vector[6] = counts.isEmpty() ? 0.0f : (float) Math.log1p(counts.size());
            vector[7] = seen == 0 ? 0.0f : (float) (count / (double) seen);
        }

        @Override
        public void update(int eventId) {
            counts.put(eventId, counts.getOrDefault(eventId, 0L) + 1);
            ++seen;
        }
    }

    private static final class EmbeddingEncoder implements Encoder {
        private final int dimensions;

        private EmbeddingEncoder(int dimensions) {
            this.dimensions = dimensions;
        }

        @Override
        public int dimensions() {
            return dimensions;
        }

        @Override
        public void encode(int eventId, float[] vector) {
            float scale = (float) (1.0 / Math.sqrt(dimensions));
            for (int i = 0; i < dimensions; i++) {
                int hash = positiveHash("emb:" + eventId + ":" + i);
                vector[i] = (hash & 1) == 0 ? scale : -scale;
            }
        }

        @Override
        public void update(int eventId) {
            // stateless
        }
    }

    private static final class EmbeddingRarityEncoder implements Encoder {
        private final EmbeddingEncoder embedding;
        private final RarityEncoder rarity = new RarityEncoder();

        private EmbeddingRarityEncoder(int embeddingDimensions) {
            this.embedding = new EmbeddingEncoder(embeddingDimensions);
        }

        @Override
        public int dimensions() {
            return embedding.dimensions() + rarity.dimensions();
        }

        @Override
        public void encode(int eventId, float[] vector) {
            float[] rarityVector = new float[rarity.dimensions()];
            embedding.encode(eventId, vector);
            rarity.encode(eventId, rarityVector);
            System.arraycopy(rarityVector, 0, vector, embedding.dimensions(), rarityVector.length);
        }

        @Override
        public void update(int eventId) {
            rarity.update(eventId);
        }
    }

    private static ParsedLine parseBglLikeLine(String line) {
        String[] parts = line.split("\\s+", 10);
        if (parts.length < 9) {
            return null;
        }
        boolean label = !parts[0].startsWith("-");
        String message = parts[8] + (parts.length >= 10 ? " " + parts[9] : "");
        return new ParsedLine(label, message);
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

    private static void normalize(float[] values) {
        double sumSquares = 0.0;
        for (float value : values) {
            sumSquares += value * value;
        }
        if (sumSquares == 0.0) {
            return;
        }
        float scale = (float) (1.0 / Math.sqrt(sumSquares));
        for (int i = 0; i < values.length; i++) {
            values[i] *= scale;
        }
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

    private static int positiveHash(String value) {
        int hash = 0x811c9dc5;
        for (int i = 0; i < value.length(); i++) {
            hash ^= value.charAt(i);
            hash *= 0x01000193;
        }
        return hash & 0x7fffffff;
    }

    private static void writeResults(String output, List<Result> results) throws IOException {
        Path path = Paths.get(output);
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write("dataset,method,records,train_records,test_records,positives,negatives,auc");
            writer.newLine();
            for (Result result : results) {
                writer.write(result.toCsv());
                writer.newLine();
            }
        }
    }

    private static void printResults(List<Result> results) {
        System.out.println("dataset,method,auc,test_records,positives,negatives");
        for (Result result : results) {
            System.out.printf(Locale.ROOT, "%s,%s,%.6f,%d,%d,%d%n", result.dataset, result.method, result.auc,
                    result.testRecords, result.positives, result.negatives);
        }
    }

    private static final class ScoreBuffer {
        private float[] scores = new float[1 << 20];
        private byte[] labels = new byte[1 << 20];
        private int size;
        private long positives;
        private long negatives;

        private void add(float score, boolean label) {
            if (size == scores.length) {
                scores = Arrays.copyOf(scores, scores.length * 2);
                labels = Arrays.copyOf(labels, labels.length * 2);
            }
            scores[size] = Float.isFinite(score) ? score : 0.0f;
            labels[size] = (byte) (label ? 1 : 0);
            if (label) {
                ++positives;
            } else {
                ++negatives;
            }
            ++size;
        }

        private double auc() {
            if (positives == 0 || negatives == 0) {
                return Double.NaN;
            }
            long[] packed = new long[size];
            for (int i = 0; i < size; i++) {
                int bits = Float.floatToIntBits(scores[i]);
                packed[i] = ((long) bits << 32) | (labels[i] & 1L);
            }
            Arrays.sort(packed);
            double rankSumPositive = 0.0;
            long rank = 1;
            int index = 0;
            while (index < packed.length) {
                int bits = (int) (packed[index] >>> 32);
                int end = index;
                long positiveInTie = 0;
                while (end < packed.length && (int) (packed[end] >>> 32) == bits) {
                    positiveInTie += packed[end] & 1L;
                    ++end;
                }
                long groupSize = end - index;
                double averageRank = (rank + rank + groupSize - 1) / 2.0;
                rankSumPositive += positiveInTie * averageRank;
                rank += groupSize;
                index = end;
            }
            return (rankSumPositive - positives * (positives + 1) / 2.0) / (positives * (double) negatives);
        }
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

    private static final class ParsedLine {
        private final boolean label;
        private final String message;

        private ParsedLine(boolean label, String message) {
            this.label = label;
            this.message = message;
        }
    }

    private static final class Result {
        private final String dataset;
        private final String method;
        private final long records;
        private final long trainRecords;
        private final long testRecords;
        private final long positives;
        private final long negatives;
        private final double auc;

        private Result(String dataset, String method, long records, long trainRecords, long testRecords, long positives,
                long negatives, double auc) {
            this.dataset = dataset;
            this.method = method;
            this.records = records;
            this.trainRecords = trainRecords;
            this.testRecords = testRecords;
            this.positives = positives;
            this.negatives = negatives;
            this.auc = auc;
        }

        private String toCsv() {
            return String.format(Locale.ROOT, "%s,%s,%d,%d,%d,%d,%d,%.6f", dataset, method, records, trainRecords,
                    testRecords, positives, negatives, auc);
        }
    }

    private static final class Config {
        private final List<String> datasets;
        private final String bglPath;
        private final String thunderbirdPath;
        private final String output;
        private final int numberOfTrees;
        private final int sampleSize;
        private final int hashDimensions;
        private final int embeddingDimensions;
        private final int windowSize;
        private final int sampleModulo;
        private final long seed;
        private final double trainFraction;
        private final double drainSimilarity;
        private final List<String> models;

        private Config(List<String> datasets, String bglPath, String thunderbirdPath, String output, int numberOfTrees,
                int sampleSize, int hashDimensions, int embeddingDimensions, int windowSize, int sampleModulo, long seed,
                double trainFraction, double drainSimilarity, List<String> models) {
            this.datasets = datasets;
            this.bglPath = bglPath;
            this.thunderbirdPath = thunderbirdPath;
            this.output = output;
            this.numberOfTrees = numberOfTrees;
            this.sampleSize = sampleSize;
            this.hashDimensions = hashDimensions;
            this.embeddingDimensions = embeddingDimensions;
            this.windowSize = windowSize;
            this.sampleModulo = sampleModulo;
            this.seed = seed;
            this.trainFraction = trainFraction;
            this.drainSimilarity = drainSimilarity;
            this.models = models;
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
                    values.getOrDefault("output", "benchmark-results/log-ad/rcf_online_event_features.csv"),
                    Integer.parseInt(values.getOrDefault("trees", "20")),
                    Integer.parseInt(values.getOrDefault("sample-size", "256")),
                    Integer.parseInt(values.getOrDefault("hash-dimensions", "512")),
                    Integer.parseInt(values.getOrDefault("embedding-dimensions", "16")),
                    Integer.parseInt(values.getOrDefault("window-size", "1000")),
                    Integer.parseInt(values.getOrDefault("sample-modulo", "10")),
                    Long.parseLong(values.getOrDefault("seed", "42")),
                    Double.parseDouble(values.getOrDefault("train-fraction", "0.05")),
                    Double.parseDouble(values.getOrDefault("drain-similarity", "0.5")),
                    split(values.getOrDefault("models", "onehot,rarity,embedding,embedding_rarity,window")));
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
