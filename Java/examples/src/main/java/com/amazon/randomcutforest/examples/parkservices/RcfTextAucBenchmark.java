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

import static java.lang.Math.floor;

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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.amazon.randomcutforest.RandomCutForest;

/**
 * Paper-style AUC benchmark for RCF over three log representations:
 * Drain-style parsed events, normalized words, and normalized character
 * trigrams. The benchmark follows the dataset granularity in arXiv:2312.01934:
 * BGL and Thunderbird are line-level, Hadoop and HDFS are sequence-level. RCF is
 * trained on the initial split and then scored/updated online over the test
 * split.
 */
public final class RcfTextAucBenchmark {

    private static final Pattern BLOCK_ID = Pattern.compile("\\bblk_-?\\d+\\b");
    private static final long PAPER_THUNDERBIRD_LINES = 211_212_192L;
    private static final Charset RAW_LOG_CHARSET = StandardCharsets.ISO_8859_1;

    private RcfTextAucBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        List<Result> results = new ArrayList<>();

        for (String dataset : config.datasets) {
            long started = System.nanoTime();
            if ("bgl".equals(dataset)) {
                results.addAll(runLineDataset("BGL", config.bglPath, config.bglTrainFraction, 1, 0,
                        countLines(Paths.get(config.bglPath)), config));
            } else if ("thunderbird".equals(dataset)) {
                long trainCutoff = (long) floor(PAPER_THUNDERBIRD_LINES * config.thunderbirdTrainFraction);
                results.addAll(runLineDataset("Thunderbird", config.thunderbirdPath, config.thunderbirdTrainFraction,
                        config.thunderbirdSampleModulo, trainCutoff, PAPER_THUNDERBIRD_LINES, config));
            } else if ("hadoop".equals(dataset)) {
                results.addAll(runHadoop(config));
            } else if ("hdfs".equals(dataset)) {
                results.addAll(runHdfs(config));
            } else {
                throw new IllegalArgumentException("unsupported dataset: " + dataset);
            }
            double elapsedSeconds = (System.nanoTime() - started) / 1_000_000_000.0;
            System.out.printf(Locale.ROOT, "finished dataset=%s elapsed_s=%.3f%n", dataset, elapsedSeconds);
        }

        writeResults(results, config.output);
        printResults(results);
    }

    private static List<Result> runLineDataset(String dataset, String input, double trainFraction, int sampleModulo,
            long originalTrainCutoff, long totalLines, Config config) throws IOException {
        System.out.printf(Locale.ROOT, "running dataset=%s input=%s sample_modulo=%d%n", dataset, input, sampleModulo);
        ModelSet models = new ModelSet(config, dataset, (long) floor(totalLines * trainFraction / sampleModulo));
        long sampled = 0;
        long originalLineNumber = 0;
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(input), RAW_LOG_CHARSET)) {
            String line;
            while ((line = reader.readLine()) != null) {
                ++originalLineNumber;
                if (sampleModulo > 1 && positiveHash(Long.toString(originalLineNumber)) % sampleModulo != 0) {
                    continue;
                }
                ParsedLine parsed = parseBglLikeLine(line);
                if (parsed == null) {
                    continue;
                }
                ++sampled;
                boolean train;
                if (sampleModulo > 1 && originalTrainCutoff > 0) {
                    train = originalLineNumber <= originalTrainCutoff;
                } else {
                    train = sampled <= models.trainRecords;
                }
                models.process(parsed.label, parsed.message, train);
                if (sampled % 5_000_000L == 0) {
                    System.out.printf(Locale.ROOT, "  %s sampled=%d original=%d%n", dataset, sampled,
                            originalLineNumber);
                }
            }
        }
        return models.results(sampled);
    }

    private static List<Result> runHadoop(Config config) throws IOException {
        System.out.printf(Locale.ROOT, "running dataset=Hadoop input=%s%n", config.hadoopPath);
        Map<String, Boolean> labels = loadHadoopLabels(Paths.get(config.hadoopPath, "abnormal_label.txt"));
        List<Path> applications = new ArrayList<>();
        try (var stream = Files.list(Paths.get(config.hadoopPath))) {
            stream.filter(Files::isDirectory).filter(path -> path.getFileName().toString().startsWith("application_"))
                    .sorted().forEach(applications::add);
        }
        ModelSet models = new ModelSet(config, "Hadoop", (long) floor(applications.size() * config.hadoopTrainFraction));
        long index = 0;
        for (Path application : applications) {
            String appId = application.getFileName().toString();
            boolean label = labels.getOrDefault(appId, true);
            boolean train = index < models.trainRecords;
            SequenceVectorizer vectorizer = new SequenceVectorizer(models);
            List<Path> files = new ArrayList<>();
            try (var stream = Files.walk(application)) {
                stream.filter(Files::isRegularFile).filter(path -> path.getFileName().toString().endsWith(".log"))
                        .sorted().forEach(files::add);
            }
            for (Path file : files) {
                try (BufferedReader reader = Files.newBufferedReader(file, RAW_LOG_CHARSET)) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String message = parseHadoopMessage(line);
                        if (message != null) {
                            vectorizer.add(message);
                        }
                    }
                }
            }
            models.process(label, vectorizer, train);
            ++index;
        }
        return models.results(applications.size());
    }

    private static List<Result> runHdfs(Config config) throws IOException {
        System.out.printf(Locale.ROOT, "running dataset=HDFS matrix=%s raw=%s%n", config.hdfsMatrixPath,
                config.hdfsRawPath);
        HdfsMatrix matrix = loadHdfsMatrix(Paths.get(config.hdfsMatrixPath), config.dimensions);
        int records = matrix.blockIds.size();
        float[] wordFeatures = new float[records * config.dimensions];
        float[] trigramFeatures = new float[records * config.dimensions];
        Map<String, Integer> blockIndex = new HashMap<>(records * 2);
        for (int i = 0; i < records; i++) {
            blockIndex.put(matrix.blockIds.get(i), i);
        }

        try (BufferedReader reader = Files.newBufferedReader(Paths.get(config.hdfsRawPath), RAW_LOG_CHARSET)) {
            String line;
            long rawLines = 0;
            while ((line = reader.readLine()) != null) {
                ++rawLines;
                String message = parseHdfsMessage(line);
                if (message == null) {
                    continue;
                }
                List<Integer> indexes = hdfsBlockIndexes(line, blockIndex);
                if (indexes.isEmpty()) {
                    continue;
                }
                String normalized = normalize(message);
                for (int index : indexes) {
                    addWords(normalized, wordFeatures, index * config.dimensions, config.dimensions);
                    addTrigrams(normalized, trigramFeatures, index * config.dimensions, config.dimensions);
                }
                if (rawLines % 2_000_000L == 0) {
                    System.out.printf(Locale.ROOT, "  HDFS raw_lines=%d%n", rawLines);
                }
            }
        }
        normalizeRows(matrix.eventFeatures, records, config.dimensions);
        normalizeRows(wordFeatures, records, config.dimensions);
        normalizeRows(trigramFeatures, records, config.dimensions);

        ModelSet models = new ModelSet(config, "HDFS", (long) floor(records * config.hdfsTrainFraction));
        float[] scratch = new float[config.dimensions];
        for (int i = 0; i < records; i++) {
            boolean train = i < models.trainRecords;
            models.process(matrix.labels[i], matrix.eventFeatures, wordFeatures, trigramFeatures, i, scratch, train);
        }
        return models.results(records);
    }

    private static HdfsMatrix loadHdfsMatrix(Path matrixPath, int dimensions) throws IOException {
        List<String> blockIds = new ArrayList<>(575_061);
        List<Boolean> labels = new ArrayList<>(575_061);
        float[] features = new float[575_061 * dimensions];
        int row = 0;
        try (BufferedReader reader = Files.newBufferedReader(matrixPath, StandardCharsets.UTF_8)) {
            String header = reader.readLine();
            if (header == null) {
                throw new IllegalArgumentException("empty HDFS matrix: " + matrixPath);
            }
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",", -1);
                if (parts.length < 4) {
                    continue;
                }
                if (row * dimensions >= features.length) {
                    features = Arrays.copyOf(features, features.length + 100_000 * dimensions);
                }
                blockIds.add(parts[0]);
                labels.add("Fail".equalsIgnoreCase(parts[1]));
                int base = row * dimensions;
                for (int i = 3; i < parts.length; i++) {
                    int count = Integer.parseInt(parts[i]);
                    if (count != 0) {
                        int bucket = positiveHash("E" + (i - 2)) % dimensions;
                        features[base + bucket] += count;
                    }
                }
                ++row;
            }
        }
        boolean[] labelArray = new boolean[labels.size()];
        for (int i = 0; i < labels.size(); i++) {
            labelArray[i] = labels.get(i);
        }
        return new HdfsMatrix(blockIds, labelArray, Arrays.copyOf(features, labels.size() * dimensions));
    }

    private static Map<String, Boolean> loadHadoopLabels(Path labelsPath) throws IOException {
        Map<String, Boolean> labels = new LinkedHashMap<>();
        boolean abnormalSection = false;
        try (BufferedReader reader = Files.newBufferedReader(labelsPath, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                String trimmed = line.trim();
                if ("Normal:".equals(trimmed)) {
                    abnormalSection = false;
                } else if (trimmed.endsWith(":")) {
                    abnormalSection = true;
                } else if (trimmed.startsWith("+")) {
                    labels.put(trimmed.substring(1).trim(), abnormalSection);
                }
            }
        }
        return labels;
    }

    private static ParsedLine parseBglLikeLine(String line) {
        String[] parts = line.split("\\s+", 10);
        if (parts.length < 9) {
            return null;
        }
        boolean label = !"-".equals(parts[0]);
        String message = parts[8];
        if (parts.length >= 10) {
            message += " " + parts[9];
        }
        return new ParsedLine(label, message);
    }

    private static String parseHadoopMessage(String line) {
        String[] parts = line.split("\\s+", 4);
        if (parts.length < 4) {
            return null;
        }
        return parts[2] + " " + parts[3];
    }

    private static String parseHdfsMessage(String line) {
        String[] parts = line.split("\\s+", 5);
        if (parts.length < 5) {
            return null;
        }
        return parts[3] + " " + parts[4];
    }

    private static List<Integer> hdfsBlockIndexes(String line, Map<String, Integer> blockIndex) {
        Matcher matcher = BLOCK_ID.matcher(line);
        List<Integer> indexes = Collections.emptyList();
        while (matcher.find()) {
            Integer index = blockIndex.get(matcher.group());
            if (index != null) {
                if (indexes.isEmpty()) {
                    indexes = new ArrayList<>(2);
                }
                indexes.add(index);
            }
        }
        return indexes;
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

    private static void writeResults(List<Result> results, String output) throws IOException {
        Path path = Paths.get(output);
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write("dataset,method,train_setup,records,train_records,test_records,positives,negatives,auc");
            writer.newLine();
            for (Result result : results) {
                writer.write(result.toCsv());
                writer.newLine();
            }
            writer.newLine();
            writer.write("method,average_auc");
            writer.newLine();
            for (Map.Entry<String, Double> entry : averageAuc(results).entrySet()) {
                writer.write(entry.getKey() + "," + String.format(Locale.ROOT, "%.6f", entry.getValue()));
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
        System.out.println();
        System.out.println("method,average_auc");
        for (Map.Entry<String, Double> entry : averageAuc(results).entrySet()) {
            System.out.printf(Locale.ROOT, "%s,%.6f%n", entry.getKey(), entry.getValue());
        }
    }

    private static Map<String, Double> averageAuc(List<Result> results) {
        Map<String, double[]> sums = new LinkedHashMap<>();
        for (Result result : results) {
            double[] value = sums.computeIfAbsent(result.method, ignored -> new double[2]);
            value[0] += result.auc;
            value[1] += 1.0;
        }
        Map<String, Double> averages = new LinkedHashMap<>();
        for (Map.Entry<String, double[]> entry : sums.entrySet()) {
            averages.put(entry.getKey(), entry.getValue()[0] / entry.getValue()[1]);
        }
        return averages;
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

    private static void addWords(String normalized, float[] vector, int offset, int dimensions) {
        int start = -1;
        for (int i = 0; i <= normalized.length(); i++) {
            boolean boundary = i == normalized.length() || Character.isWhitespace(normalized.charAt(i));
            if (boundary) {
                if (start >= 0 && i > start) {
                    int bucket = positiveHash(normalized, start, i) % dimensions;
                    vector[offset + bucket] += 1.0f;
                }
                start = -1;
            } else if (start < 0) {
                start = i;
            }
        }
    }

    private static void addTrigrams(String normalized, float[] vector, int offset, int dimensions) {
        if (normalized.length() < 3) {
            if (!normalized.isEmpty()) {
                vector[offset + positiveHash(normalized) % dimensions] += 1.0f;
            }
            return;
        }
        for (int i = 0; i <= normalized.length() - 3; i++) {
            int bucket = positiveHash(normalized, i, i + 3) % dimensions;
            vector[offset + bucket] += 1.0f;
        }
    }

    private static void normalizeRows(float[] values, int rows, int dimensions) {
        for (int row = 0; row < rows; row++) {
            normalizeVector(values, row * dimensions, dimensions);
        }
    }

    private static void normalizeVector(float[] values, int offset, int dimensions) {
        double sumSquares = 0.0;
        for (int i = 0; i < dimensions; i++) {
            float value = values[offset + i];
            sumSquares += value * value;
        }
        if (sumSquares == 0.0) {
            return;
        }
        float scale = (float) (1.0 / Math.sqrt(sumSquares));
        for (int i = 0; i < dimensions; i++) {
            values[offset + i] *= scale;
        }
    }

    private static int positiveHash(String value) {
        return positiveHash(value, 0, value.length());
    }

    private static int positiveHash(String value, int start, int end) {
        int hash = 0x811c9dc5;
        for (int i = start; i < end; i++) {
            hash ^= value.charAt(i);
            hash *= 0x01000193;
        }
        return hash & 0x7fffffff;
    }

    private enum Representation {
        EVENTS("RCF + parsed events"),
        TRIGRAMS("RCF + char trigrams"),
        WORDS("RCF + words");

        private final String method;

        Representation(String method) {
            this.method = method;
        }
    }

    private static final class ModelSet {
        private final long trainRecords;
        private long observedTrainRecords;
        private final Variant events;
        private final Variant trigrams;
        private final Variant words;

        private ModelSet(Config config, String dataset, long trainRecords) {
            this.trainRecords = trainRecords;
            this.events = new Variant(dataset, Representation.EVENTS, config);
            this.trigrams = new Variant(dataset, Representation.TRIGRAMS, config);
            this.words = new Variant(dataset, Representation.WORDS, config);
        }

        private void process(boolean label, String message, boolean train) {
            if (train) {
                ++observedTrainRecords;
            }
            String normalized = normalize(message);
            events.process(label, normalized, train);
            trigrams.process(label, normalized, train);
            words.process(label, normalized, train);
        }

        private void process(boolean label, SequenceVectorizer vectorizer, boolean train) {
            if (train) {
                ++observedTrainRecords;
            }
            events.processVector(label, vectorizer.events, train);
            trigrams.processVector(label, vectorizer.trigrams, train);
            words.processVector(label, vectorizer.words, train);
        }

        private void process(boolean label, float[] eventFeatures, float[] wordFeatures, float[] trigramFeatures,
                int row, float[] scratch, boolean train) {
            if (train) {
                ++observedTrainRecords;
            }
            events.processVector(label, eventFeatures, row, scratch, train);
            trigrams.processVector(label, trigramFeatures, row, scratch, train);
            words.processVector(label, wordFeatures, row, scratch, train);
        }

        private List<Result> results(long records) {
            List<Result> results = new ArrayList<>(3);
            results.add(events.result(records, observedTrainRecords));
            results.add(trigrams.result(records, observedTrainRecords));
            results.add(words.result(records, observedTrainRecords));
            return results;
        }
    }

    private static final class Variant {
        private final String dataset;
        private final Representation representation;
        private final int dimensions;
        private final SimpleDrainParser parser = new SimpleDrainParser(0.5);
        private final RandomCutForest forest;
        private final ScoreBuffer scores = new ScoreBuffer();
        private final float[] vector;
        private final int[] touched;
        private int touchedCount;

        private Variant(String dataset, Representation representation, Config config) {
            this.dataset = dataset;
            this.representation = representation;
            this.dimensions = config.dimensions;
            this.vector = new float[dimensions];
            this.touched = new int[dimensions];
            this.forest = RandomCutForest.builder().compact(true).dimensions(dimensions)
                    .numberOfTrees(config.numberOfTrees).sampleSize(config.sampleSize).randomSeed(config.seed)
                    .parallelExecutionEnabled(false).build();
        }

        private void process(boolean label, String normalized, boolean train) {
            clear();
            if (representation == Representation.EVENTS) {
                int eventId = parser.parse(normalized);
                add(positiveHash("E" + eventId) % dimensions, 1.0f);
            } else if (representation == Representation.TRIGRAMS) {
                addTrigrams(normalized);
            } else {
                addWords(normalized);
            }
            normalize();
            scoreOrTrain(label, vector, train);
        }

        private void processVector(boolean label, float[] source, boolean train) {
            scoreOrTrain(label, source, train);
        }

        private void processVector(boolean label, float[] source, int row, float[] scratch, boolean train) {
            System.arraycopy(source, row * dimensions, scratch, 0, dimensions);
            scoreOrTrain(label, scratch, train);
        }

        private void scoreOrTrain(boolean label, float[] point, boolean train) {
            if (!train) {
                double score = forest.getAnomalyScore(point);
                scores.add((float) score, label);
            }
            forest.update(point);
        }

        private Result result(long records, long trainRecords) {
            return new Result(dataset, representation.method, "unfiltered-online", records, trainRecords, scores.size,
                    scores.positives, scores.negatives, scores.auc());
        }

        private void clear() {
            for (int i = 0; i < touchedCount; i++) {
                vector[touched[i]] = 0.0f;
            }
            touchedCount = 0;
        }

        private void addWords(String normalized) {
            int start = -1;
            for (int i = 0; i <= normalized.length(); i++) {
                boolean boundary = i == normalized.length() || Character.isWhitespace(normalized.charAt(i));
                if (boundary) {
                    if (start >= 0 && i > start) {
                        add(positiveHash(normalized, start, i) % dimensions, 1.0f);
                    }
                    start = -1;
                } else if (start < 0) {
                    start = i;
                }
            }
        }

        private void addTrigrams(String normalized) {
            if (normalized.length() < 3) {
                if (!normalized.isEmpty()) {
                    add(positiveHash(normalized) % dimensions, 1.0f);
                }
                return;
            }
            for (int i = 0; i <= normalized.length() - 3; i++) {
                add(positiveHash(normalized, i, i + 3) % dimensions, 1.0f);
            }
        }

        private void add(int index, float increment) {
            if (vector[index] == 0.0f) {
                touched[touchedCount++] = index;
            }
            vector[index] += increment;
        }

        private void normalize() {
            double sumSquares = 0.0;
            for (int i = 0; i < touchedCount; i++) {
                float value = vector[touched[i]];
                sumSquares += value * value;
            }
            if (sumSquares == 0.0) {
                return;
            }
            float scale = (float) (1.0 / Math.sqrt(sumSquares));
            for (int i = 0; i < touchedCount; i++) {
                vector[touched[i]] *= scale;
            }
        }
    }

    private static final class SequenceVectorizer {
        private final ModelSet models;
        private final float[] events;
        private final float[] trigrams;
        private final float[] words;

        private SequenceVectorizer(ModelSet models) {
            this.models = models;
            this.events = new float[models.events.dimensions];
            this.trigrams = new float[models.events.dimensions];
            this.words = new float[models.events.dimensions];
        }

        private void add(String message) {
            String normalized = normalize(message);
            int eventId = models.events.parser.parse(normalized);
            events[positiveHash("E" + eventId) % events.length] += 1.0f;
            RcfTextAucBenchmark.addTrigrams(normalized, trigrams, 0, trigrams.length);
            RcfTextAucBenchmark.addWords(normalized, words, 0, words.length);
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
            boolean hasDigit = false;
            for (int i = 0; i < rawToken.length(); i++) {
                if (Character.isDigit(rawToken.charAt(i))) {
                    hasDigit = true;
                    break;
                }
            }
            if (hasDigit) {
                return "<*>";
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

    private static final class HdfsMatrix {
        private final List<String> blockIds;
        private final boolean[] labels;
        private final float[] eventFeatures;

        private HdfsMatrix(List<String> blockIds, boolean[] labels, float[] eventFeatures) {
            this.blockIds = blockIds;
            this.labels = labels;
            this.eventFeatures = eventFeatures;
        }
    }

    private static final class Result {
        private final String dataset;
        private final String method;
        private final String trainSetup;
        private final long records;
        private final long trainRecords;
        private final long testRecords;
        private final long positives;
        private final long negatives;
        private final double auc;

        private Result(String dataset, String method, String trainSetup, long records, long trainRecords,
                long testRecords, long positives, long negatives, double auc) {
            this.dataset = dataset;
            this.method = method;
            this.trainSetup = trainSetup;
            this.records = records;
            this.trainRecords = trainRecords;
            this.testRecords = testRecords;
            this.positives = positives;
            this.negatives = negatives;
            this.auc = auc;
        }

        private String toCsv() {
            return String.format(Locale.ROOT, "%s,%s,%s,%d,%d,%d,%d,%d,%.6f", dataset, method, trainSetup, records,
                    trainRecords, testRecords, positives, negatives, auc);
        }
    }

    private static final class Config {
        private final List<String> datasets;
        private final String bglPath;
        private final String thunderbirdPath;
        private final String hadoopPath;
        private final String hdfsRawPath;
        private final String hdfsMatrixPath;
        private final String output;
        private final int dimensions;
        private final int numberOfTrees;
        private final int sampleSize;
        private final long seed;
        private final double bglTrainFraction;
        private final double thunderbirdTrainFraction;
        private final double hadoopTrainFraction;
        private final double hdfsTrainFraction;
        private final int thunderbirdSampleModulo;

        private Config(List<String> datasets, String bglPath, String thunderbirdPath, String hadoopPath,
                String hdfsRawPath, String hdfsMatrixPath, String output, int dimensions, int numberOfTrees,
                int sampleSize, long seed, double bglTrainFraction, double thunderbirdTrainFraction,
                double hadoopTrainFraction, double hdfsTrainFraction, int thunderbirdSampleModulo) {
            this.datasets = datasets;
            this.bglPath = bglPath;
            this.thunderbirdPath = thunderbirdPath;
            this.hadoopPath = hadoopPath;
            this.hdfsRawPath = hdfsRawPath;
            this.hdfsMatrixPath = hdfsMatrixPath;
            this.output = output;
            this.dimensions = dimensions;
            this.numberOfTrees = numberOfTrees;
            this.sampleSize = sampleSize;
            this.seed = seed;
            this.bglTrainFraction = bglTrainFraction;
            this.thunderbirdTrainFraction = thunderbirdTrainFraction;
            this.hadoopTrainFraction = hadoopTrainFraction;
            this.hdfsTrainFraction = hdfsTrainFraction;
            this.thunderbirdSampleModulo = thunderbirdSampleModulo;
        }

        private static Config parse(String[] args) {
            Map<String, String> values = new HashMap<>();
            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                if (!arg.startsWith("--")) {
                    throw new IllegalArgumentException("unexpected argument: " + arg);
                }
                if (i + 1 >= args.length) {
                    throw new IllegalArgumentException("missing value for " + arg);
                }
                values.put(arg.substring(2), args[++i]);
            }
            String root = values.getOrDefault("data-root", "/tmp/loghub_data");
            return new Config(split(values.getOrDefault("datasets", "bgl,hadoop,hdfs,thunderbird")),
                    values.getOrDefault("bgl", root + "/BGL/BGL.log"),
                    values.getOrDefault("thunderbird", root + "/Thunderbird/Thunderbird.log"),
                    values.getOrDefault("hadoop", root + "/Hadoop"),
                    values.getOrDefault("hdfs-raw", root + "/HDFS/HDFS.log"),
                    values.getOrDefault("hdfs-matrix", root + "/HDFS/preprocessed/Event_occurrence_matrix.csv"),
                    values.getOrDefault("output", "benchmark-results/log-ad/rcf_text_auc_results.csv"),
                    Integer.parseInt(values.getOrDefault("dimensions", "256")),
                    Integer.parseInt(values.getOrDefault("trees", "50")),
                    Integer.parseInt(values.getOrDefault("sample-size", "256")),
                    Long.parseLong(values.getOrDefault("seed", "42")),
                    Double.parseDouble(values.getOrDefault("bgl-train-fraction", "0.05")),
                    Double.parseDouble(values.getOrDefault("thunderbird-train-fraction", "0.05")),
                    Double.parseDouble(values.getOrDefault("hadoop-train-fraction", "0.50")),
                    Double.parseDouble(values.getOrDefault("hdfs-train-fraction", "0.05")),
                    Integer.parseInt(values.getOrDefault("thunderbird-sample-modulo", "10")));
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
