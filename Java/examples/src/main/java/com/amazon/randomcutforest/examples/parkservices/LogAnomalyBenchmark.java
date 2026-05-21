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
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.config.ScoringStrategy;
import com.amazon.randomcutforest.parkservices.log.LogAnomalyDetector;
import com.amazon.randomcutforest.parkservices.log.LogAnomalyResult;
import com.amazon.randomcutforest.parkservices.log.LogBucketStats;
import com.amazon.randomcutforest.parkservices.log.LogBucketVectorizer;
import com.amazon.randomcutforest.parkservices.log.LogVectorSchema;

/**
 * Streams raw LogHub log files through bucketed online detectors. This runner is
 * intentionally small and dependency-free so the same code can be used against
 * full raw datasets without materializing a structured CSV first.
 */
public final class LogAnomalyBenchmark {

    private static final Pattern BLOCK_ID = Pattern.compile("\\bblk_-?\\d+\\b");

    private LogAnomalyBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        List<OnlineDetector> detectors = createDetectors(config);

        Map<String, Metrics> metrics = new LinkedHashMap<>();
        for (OnlineDetector detector : detectors) {
            metrics.put(detector.name(), new Metrics(detector.name(), config.warmupBuckets));
        }

        DatasetReader reader = createReader(config);
        long started = System.nanoTime();
        DatasetSummary summary = reader.stream(config.bucketSeconds, config.includeEmptyBuckets, bucket -> {
            for (OnlineDetector detector : detectors) {
                long detectorStarted = System.nanoTime();
                Detection detection = detector.process(bucket, bucket.ordinal);
                long elapsed = System.nanoTime() - detectorStarted;
                metrics.get(detector.name()).record(bucket, detection.anomaly, elapsed, detector.stateSize());
            }
        });
        long elapsed = System.nanoTime() - started;

        System.out.printf(Locale.ROOT,
                "dataset=%s input=%s bucket_seconds=%d buckets=%d positive_buckets=%d logs=%d anomalous_logs=%d elapsed_s=%.3f%n",
                config.dataset, config.input, config.bucketSeconds, summary.bucketCount, summary.positiveBucketCount,
                summary.logCount, summary.anomalousLogCount, elapsed / 1_000_000_000.0);
        System.out.println();
        System.out.println(Metrics.header());
        for (Metrics value : metrics.values()) {
            System.out.println(value.toCsv());
        }

        if (config.output != null) {
            writeMetrics(config, summary, metrics);
        }
    }

    private static DatasetReader createReader(Config config) throws IOException {
        if ("bgl".equals(config.dataset)) {
            return new BglReader(config.input);
        }
        if ("hdfs".equals(config.dataset)) {
            if (config.labels == null) {
                throw new IllegalArgumentException("--labels is required for HDFS");
            }
            return new HdfsReader(config.input, config.labels);
        }
        throw new IllegalArgumentException("unsupported dataset: " + config.dataset);
    }

    private static List<OnlineDetector> createDetectors(Config config) {
        List<OnlineDetector> detectors = new ArrayList<>();
        for (String detector : config.detectors) {
            if ("current".equals(detector)) {
                detectors.add(new ExistingLogAnomalyDetector("current_LogAnomalyDetector",
                        config.minimumBaselineBuckets, config.seed, false));
            } else if ("current_update".equals(detector)) {
                detectors.add(new ExistingLogAnomalyDetector("current_LogAnomalyDetector_updateBaseline",
                        config.minimumBaselineBuckets, config.seed, true));
            } else if ("regex_ewma".equals(detector)) {
                detectors.add(new RegexEwmaDetector(config.minimumBaselineBuckets));
            } else if ("drain_ewma".equals(detector)) {
                detectors.add(new DrainEwmaDetector(config.minimumBaselineBuckets, config.drainSimilarity));
            } else if ("drain_signal".equals(detector)) {
                detectors.add(new DrainSignalEwmaDetector(config.minimumBaselineBuckets, config.drainSimilarity));
            } else if ("drain_template".equals(detector)) {
                detectors.add(new DrainTemplateEwmaDetector(config.minimumBaselineBuckets, config.drainSimilarity));
            } else if ("drain_rcf".equals(detector)) {
                detectors.add(new DrainRcfDetector(config.minimumBaselineBuckets, config.drainSimilarity, config.seed));
            } else {
                throw new IllegalArgumentException("unsupported detector: " + detector);
            }
        }
        return detectors;
    }

    private static void writeMetrics(Config config, DatasetSummary summary, Map<String, Metrics> metrics)
            throws IOException {
        Path output = Paths.get(config.output);
        Path parent = output.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
            writer.write("dataset,input,bucket_seconds,buckets,positive_buckets,logs,anomalous_logs");
            writer.newLine();
            writer.write(String.format(Locale.ROOT, "%s,%s,%d,%d,%d,%d,%d", config.dataset, config.input,
                    config.bucketSeconds, summary.bucketCount, summary.positiveBucketCount, summary.logCount,
                    summary.anomalousLogCount));
            writer.newLine();
            writer.newLine();
            writer.write(Metrics.header());
            writer.newLine();
            for (Metrics value : metrics.values()) {
                writer.write(value.toCsv());
                writer.newLine();
            }
        }
    }

    private interface BucketConsumer {
        void accept(Bucket bucket);
    }

    private interface DatasetReader {
        DatasetSummary stream(long bucketSeconds, boolean includeEmptyBuckets, BucketConsumer consumer) throws IOException;
    }

    private static final class BglReader implements DatasetReader {
        private final String input;

        private BglReader(String input) {
            this.input = input;
        }

        @Override
        public DatasetSummary stream(long bucketSeconds, boolean includeEmptyBuckets, BucketConsumer consumer)
                throws IOException {
            return streamParsedLogs(input, bucketSeconds, includeEmptyBuckets, consumer, new ParsedLineFactory() {
                @Override
                public ParsedLine parse(String line) {
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
                    boolean anomaly = !"-".equals(parts[0]);
                    String content = parts.length >= 10 ? parts[9] : "";
                    String message = parts[8] + " " + parts[7] + " " + content;
                    return new ParsedLine(epochSeconds, anomaly, message);
                }
            });
        }
    }

    private static final class HdfsReader implements DatasetReader {
        private final String input;
        private final Set<String> anomalousBlockIds;

        private HdfsReader(String input, String labels) throws IOException {
            this.input = input;
            this.anomalousBlockIds = loadAnomalousBlocks(labels);
        }

        @Override
        public DatasetSummary stream(long bucketSeconds, boolean includeEmptyBuckets, BucketConsumer consumer)
                throws IOException {
            return streamParsedLogs(input, bucketSeconds, includeEmptyBuckets, consumer, new ParsedLineFactory() {
                @Override
                public ParsedLine parse(String line) {
                    String[] parts = line.split("\\s+", 5);
                    if (parts.length < 5) {
                        return null;
                    }
                    long epochSeconds;
                    try {
                        epochSeconds = hdfsEpochSeconds(parts[0], parts[1]);
                    } catch (RuntimeException e) {
                        return null;
                    }
                    boolean anomaly = containsAnomalousBlock(line, anomalousBlockIds);
                    String message = parts[3] + " " + parts[4];
                    return new ParsedLine(epochSeconds, anomaly, message);
                }
            });
        }
    }

    private interface ParsedLineFactory {
        ParsedLine parse(String line);
    }

    private static DatasetSummary streamParsedLogs(String input, long bucketSeconds, boolean includeEmptyBuckets,
            BucketConsumer consumer, ParsedLineFactory factory) throws IOException {
        DatasetSummary summary = new DatasetSummary();
        Bucket current = null;
        long previousBucketKey = Long.MIN_VALUE;
        long ordinal = 0;
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(input), StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                ParsedLine parsed = factory.parse(line);
                if (parsed == null) {
                    continue;
                }
                long bucketKey = Math.floorDiv(parsed.epochSeconds, bucketSeconds);
                if (current == null) {
                    current = new Bucket(bucketKey, bucketKey * bucketSeconds, ordinal++);
                } else if (bucketKey != current.bucketKey) {
                    acceptBucket(current, consumer, summary);
                    if (includeEmptyBuckets) {
                        for (long missing = current.bucketKey + 1; missing < bucketKey; missing++) {
                            Bucket empty = new Bucket(missing, missing * bucketSeconds, ordinal++);
                            acceptBucket(empty, consumer, summary);
                        }
                    }
                    previousBucketKey = current.bucketKey;
                    current = new Bucket(bucketKey, bucketKey * bucketSeconds, ordinal++);
                }
                if (bucketKey < previousBucketKey) {
                    throw new IllegalArgumentException("input is not ordered by timestamp near bucket " + bucketKey);
                }
                current.add(parsed.message, parsed.anomaly);
            }
        }
        if (current != null) {
            acceptBucket(current, consumer, summary);
        }
        return summary;
    }

    private static void acceptBucket(Bucket bucket, BucketConsumer consumer, DatasetSummary summary) {
        consumer.accept(bucket);
        ++summary.bucketCount;
        summary.logCount += bucket.messages.size();
        summary.anomalousLogCount += bucket.anomalousLogCount;
        if (bucket.actualAnomaly) {
            ++summary.positiveBucketCount;
        }
    }

    private static Set<String> loadAnomalousBlocks(String labels) throws IOException {
        Set<String> anomalous = new HashSet<>();
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(labels), StandardCharsets.UTF_8)) {
            String line = reader.readLine();
            while ((line = reader.readLine()) != null) {
                String[] parts = line.trim().split(",", 2);
                if (parts.length == 2 && "Anomaly".equalsIgnoreCase(parts[1].trim())) {
                    anomalous.add(parts[0].trim());
                }
            }
        }
        return anomalous;
    }

    private static boolean containsAnomalousBlock(String line, Set<String> anomalousBlockIds) {
        Matcher matcher = BLOCK_ID.matcher(line);
        while (matcher.find()) {
            if (anomalousBlockIds.contains(matcher.group())) {
                return true;
            }
        }
        return false;
    }

    private static long hdfsEpochSeconds(String date, String time) {
        int year = 2000 + Integer.parseInt(date.substring(0, 2));
        int month = Integer.parseInt(date.substring(2, 4));
        int day = Integer.parseInt(date.substring(4, 6));
        int hour = Integer.parseInt(time.substring(0, 2));
        int minute = Integer.parseInt(time.substring(2, 4));
        int second = Integer.parseInt(time.substring(4, 6));
        return LocalDate.of(year, month, day).toEpochDay() * 86_400L + hour * 3_600L + minute * 60L + second;
    }

    private interface OnlineDetector {
        String name();

        Detection process(Bucket bucket, long timestamp);

        String stateSize();
    }

    private static final class ExistingLogAnomalyDetector implements OnlineDetector {
        private final String name;
        private final LogAnomalyDetector detector;

        private ExistingLogAnomalyDetector(String name, int minimumBaselineBuckets, long seed,
                boolean updateBaselineOnAnomaly) {
            this.name = name;
            this.detector = LogAnomalyDetector.builder().minimumBaselineBuckets(minimumBaselineBuckets)
                    .outputAfter(Math.max(8, minimumBaselineBuckets / 2)).randomSeed(seed)
                    .updateBaselineOnAnomaly(updateBaselineOnAnomaly).build();
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public Detection process(Bucket bucket, long timestamp) {
            LogAnomalyResult result = detector.process(bucket.messages, timestamp);
            return new Detection(result.isAnomaly(), result.getAnomalyGrade());
        }

        @Override
        public String stateSize() {
            return "rcf=50x128,shingle=4";
        }
    }

    private static final class RegexEwmaDetector implements OnlineDetector {
        private final LogBucketVectorizer vectorizer = new LogBucketVectorizer();
        private final EwmaBaseline baseline = new EwmaBaseline(LogVectorSchema.DIMENSIONS, 0.03);
        private final int minimumBaselineBuckets;

        private RegexEwmaDetector(int minimumBaselineBuckets) {
            this.minimumBaselineBuckets = minimumBaselineBuckets;
        }

        @Override
        public String name() {
            return "regex_EWMA_no_RCF";
        }

        @Override
        public Detection process(Bucket bucket, long timestamp) {
            LogBucketStats stats = vectorizer.vectorize(bucket.messages);
            double[] vector = stats.toVector();
            boolean anomaly = baseline.count >= minimumBaselineBuckets && isLogStatAnomaly(stats, baseline);
            if (!anomaly || baseline.count < minimumBaselineBuckets) {
                baseline.update(vector);
            }
            return new Detection(anomaly, anomaly ? 1.0 : 0.0);
        }

        private boolean isLogStatAnomaly(LogBucketStats stats, EwmaBaseline currentBaseline) {
            boolean newPattern = isNewPatternAnomaly(stats, currentBaseline);
            boolean errorBurst = isHigh(LogVectorSchema.ERROR_COUNT, stats.getErrorCount(), 4.0, currentBaseline)
                    || stats.getErrorRatio() >= currentBaseline.mean(LogVectorSchema.ERROR_RATIO) + 0.12
                            && stats.getErrorCount() >= 3;
            boolean warnBurst = isHigh(LogVectorSchema.WARN_COUNT, stats.getWarnCount(), 5.0, currentBaseline);
            boolean exceptionBurst = stats.getUniqueExceptionCount() > currentBaseline
                    .mean(LogVectorSchema.UNIQUE_EXCEPTION_COUNT) && stats.getUniqueExceptionCount() > 0
                    && stats.getErrorCount() >= 2;
            boolean volumeShift = isVolumeShift(stats.getLogCount(), currentBaseline);
            return newPattern || errorBurst || warnBurst || exceptionBurst || volumeShift;
        }

        private boolean isNewPatternAnomaly(LogBucketStats stats, EwmaBaseline currentBaseline) {
            double mean = currentBaseline.mean(LogVectorSchema.NOVEL_TEMPLATE_COUNT);
            double stdDev = currentBaseline.stdDev(LogVectorSchema.NOVEL_TEMPLATE_COUNT);
            boolean noveltyBurst = stats.getNovelTemplateCount() >= mean + Math.max(2.0, 4.0 * stdDev);
            boolean repeatedNewPattern = stats.getLargestNovelTemplateCount() >= Math.max(5.0,
                    0.1 * Math.max(1, stats.getLogCount()));
            return noveltyBurst || repeatedNewPattern && mean < 1.0;
        }

        @Override
        public String stateSize() {
            return "templates=" + vectorizer.getObservedTemplateCount() + ",rcf=none";
        }
    }

    private static final class DrainEwmaDetector implements OnlineDetector {
        private final DrainBucketVectorizer vectorizer;
        private final EwmaBaseline baseline = new EwmaBaseline(DrainBucketVectorizer.DIMENSIONS, 0.03);
        private final int minimumBaselineBuckets;

        private DrainEwmaDetector(int minimumBaselineBuckets, double similarity) {
            this.minimumBaselineBuckets = minimumBaselineBuckets;
            this.vectorizer = new DrainBucketVectorizer(similarity);
        }

        @Override
        public String name() {
            return "Drain_EWMA_no_RCF";
        }

        @Override
        public Detection process(Bucket bucket, long timestamp) {
            DrainBucketStats stats = vectorizer.vectorize(bucket.messages);
            double[] vector = stats.toVector();
            boolean anomaly = baseline.count >= minimumBaselineBuckets && DrainRules.isAnomaly(stats, baseline);
            if (!anomaly || baseline.count < minimumBaselineBuckets) {
                baseline.update(vector);
            }
            return new Detection(anomaly, anomaly ? 1.0 : 0.0);
        }

        @Override
        public String stateSize() {
            return "templates=" + vectorizer.templateCount() + ",rcf=none";
        }
    }

    private static final class DrainTemplateEwmaDetector implements OnlineDetector {
        private final DrainBucketVectorizer vectorizer;
        private final TemplateCountBaseline templateBaseline = new TemplateCountBaseline(0.03);
        private final EwmaBaseline bucketBaseline = new EwmaBaseline(DrainBucketVectorizer.DIMENSIONS, 0.03);
        private final int minimumBaselineBuckets;

        private DrainTemplateEwmaDetector(int minimumBaselineBuckets, double similarity) {
            this.minimumBaselineBuckets = minimumBaselineBuckets;
            this.vectorizer = new DrainBucketVectorizer(similarity);
        }

        @Override
        public String name() {
            return "Drain_template_EWMA_no_RCF";
        }

        @Override
        public Detection process(Bucket bucket, long timestamp) {
            DrainBucketStats stats = vectorizer.vectorize(bucket.messages);
            boolean enoughBaseline = templateBaseline.count >= minimumBaselineBuckets;
            boolean anomaly = enoughBaseline && isTemplateAnomaly(stats);
            templateBaseline.update(stats, vectorizer.templateCount());
            bucketBaseline.update(stats.toVector());
            return new Detection(anomaly, anomaly ? 1.0 : 0.0);
        }

        private boolean isTemplateAnomaly(DrainBucketStats stats) {
            if (stats.logCount == 0) {
                return false;
            }
            if (stats.fatalCount >= bucketBaseline.mean(DrainBucketVectorizer.FATAL_COUNT)
                    + Math.max(3.0, 6.0 * bucketBaseline.stdDev(DrainBucketVectorizer.FATAL_COUNT))) {
                return true;
            }
            for (Map.Entry<Integer, Integer> entry : stats.templateCounts.entrySet()) {
                int templateId = entry.getKey();
                int count = entry.getValue();
                boolean severe = stats.severityTemplateCounts.containsKey(templateId);
                boolean novel = stats.novelTemplates.contains(templateId);
                if (novel && severe && count >= 2) {
                    return true;
                }
                if (novel && count >= Math.max(8, (int) Math.ceil(0.05 * stats.logCount))) {
                    return true;
                }
                TemplateStat baseline = templateBaseline.stats.get(templateId);
                if (baseline != null && severe && count >= baseline.mean + Math.max(8.0, 6.0 * baseline.stdDev())
                        && count >= Math.max(2.0 * baseline.mean, baseline.mean + 5.0)) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public String stateSize() {
            return "templates=" + vectorizer.templateCount() + ",per_template_ewma,rcf=none";
        }
    }

    private static final class DrainSignalEwmaDetector implements OnlineDetector {
        private final DrainBucketVectorizer vectorizer;
        private final EwmaBaseline baseline = new EwmaBaseline(DrainBucketVectorizer.DIMENSIONS, 0.03);
        private final int minimumBaselineBuckets;

        private DrainSignalEwmaDetector(int minimumBaselineBuckets, double similarity) {
            this.minimumBaselineBuckets = minimumBaselineBuckets;
            this.vectorizer = new DrainBucketVectorizer(similarity);
        }

        @Override
        public String name() {
            return "Drain_signal_EWMA_no_RCF";
        }

        @Override
        public Detection process(Bucket bucket, long timestamp) {
            DrainBucketStats stats = vectorizer.vectorize(bucket.messages);
            boolean anomaly = baseline.count >= minimumBaselineBuckets && DrainRules.isSignalAnomaly(stats, baseline);
            baseline.update(stats.toVector());
            return new Detection(anomaly, anomaly ? 1.0 : 0.0);
        }

        @Override
        public String stateSize() {
            return "templates=" + vectorizer.templateCount() + ",rcf=none";
        }
    }

    private static final class DrainRcfDetector implements OnlineDetector {
        private final DrainBucketVectorizer vectorizer;
        private final ThresholdedRandomCutForest forest;
        private final EwmaBaseline baseline = new EwmaBaseline(DrainBucketVectorizer.DIMENSIONS, 0.03);
        private final int minimumBaselineBuckets;

        private DrainRcfDetector(int minimumBaselineBuckets, double similarity, long seed) {
            this.minimumBaselineBuckets = minimumBaselineBuckets;
            this.vectorizer = new DrainBucketVectorizer(similarity);
            this.forest = ThresholdedRandomCutForest.builder().dimensions(DrainBucketVectorizer.DIMENSIONS * 4)
                    .shingleSize(4).sampleSize(128).numberOfTrees(50).randomSeed(seed)
                    .outputAfter(Math.max(8, minimumBaselineBuckets / 2)).transformMethod(TransformMethod.NORMALIZE)
                    .scoringStrategy(ScoringStrategy.MULTI_MODE_RECALL).anomalyRate(0.02).autoAdjust(true).build();
        }

        @Override
        public String name() {
            return "Drain_RCF";
        }

        @Override
        public Detection process(Bucket bucket, long timestamp) {
            DrainBucketStats stats = vectorizer.vectorize(bucket.messages);
            double[] vector = stats.toVector();
            AnomalyDescriptor result = forest.process(vector, timestamp);
            boolean enoughBaseline = baseline.count >= minimumBaselineBuckets;
            boolean modelTriggered = enoughBaseline && result.getAnomalyGrade() >= 0.5;
            boolean statTriggered = enoughBaseline && DrainRules.isAnomaly(stats, baseline);
            boolean anomaly = modelTriggered || statTriggered;
            if (!anomaly || !enoughBaseline) {
                baseline.update(vector);
            }
            return new Detection(anomaly, anomaly ? Math.max(result.getAnomalyGrade(), statTriggered ? 1.0 : 0.0) : 0);
        }

        @Override
        public String stateSize() {
            return "templates=" + vectorizer.templateCount() + ",rcf=50x128,shingle=4";
        }
    }

    private static boolean isHigh(int index, double value, double floorIncrease, EwmaBaseline baseline) {
        double mean = baseline.mean(index);
        double stdDev = baseline.stdDev(index);
        return value >= mean + Math.max(floorIncrease, 4.0 * stdDev);
    }

    private static boolean isVolumeShift(double logCount, EwmaBaseline baseline) {
        double mean = baseline.mean(0);
        if (mean < 1) {
            return false;
        }
        double stdDev = baseline.stdDev(0);
        double absoluteChange = Math.abs(logCount - mean);
        return absoluteChange >= Math.max(20.0, Math.max(4.0 * stdDev, 0.35 * mean));
    }

    private static final class DrainRules {
        private DrainRules() {
        }

        private static boolean isAnomaly(DrainBucketStats stats, EwmaBaseline baseline) {
            boolean errorBurst = isHigh(DrainBucketVectorizer.ERROR_COUNT, stats.errorCount, 4.0, baseline)
                    || stats.errorRatio() >= baseline.mean(DrainBucketVectorizer.ERROR_RATIO) + 0.12
                            && stats.errorCount >= 3;
            boolean warnBurst = isHigh(DrainBucketVectorizer.WARN_COUNT, stats.warnCount, 5.0, baseline);
            boolean fatalBurst = isHigh(DrainBucketVectorizer.FATAL_COUNT, stats.fatalCount, 2.0, baseline);
            boolean uniqueShift = isHigh(DrainBucketVectorizer.UNIQUE_TEMPLATE_COUNT, stats.uniqueTemplateCount, 4.0,
                    baseline);
            boolean newPattern = isNewPattern(stats, baseline);
            boolean volumeShift = isVolumeShift(stats.logCount, baseline);
            return errorBurst || warnBurst || fatalBurst || uniqueShift || newPattern || volumeShift;
        }

        private static boolean isSignalAnomaly(DrainBucketStats stats, EwmaBaseline baseline) {
            boolean errorBurst = isHigh(DrainBucketVectorizer.ERROR_COUNT, stats.errorCount, 8.0, baseline)
                    || stats.errorRatio() >= baseline.mean(DrainBucketVectorizer.ERROR_RATIO) + 0.18
                            && stats.errorCount >= 5;
            boolean warnBurst = isHigh(DrainBucketVectorizer.WARN_COUNT, stats.warnCount, 10.0, baseline);
            boolean fatalBurst = isHigh(DrainBucketVectorizer.FATAL_COUNT, stats.fatalCount, 6.0, baseline);
            boolean newPattern = isNewPattern(stats, baseline);
            return errorBurst || warnBurst || fatalBurst || newPattern;
        }

        private static boolean isNewPattern(DrainBucketStats stats, EwmaBaseline baseline) {
            double mean = baseline.mean(DrainBucketVectorizer.NOVEL_TEMPLATE_COUNT);
            double stdDev = baseline.stdDev(DrainBucketVectorizer.NOVEL_TEMPLATE_COUNT);
            boolean noveltyBurst = stats.novelTemplateCount >= mean + Math.max(2.0, 4.0 * stdDev);
            boolean repeatedNewPattern = stats.largestNovelTemplateCount >= Math.max(5.0,
                    0.1 * Math.max(1, stats.logCount));
            return noveltyBurst || repeatedNewPattern && mean < 1.0;
        }
    }

    private static final class DrainBucketVectorizer {
        private static final int LOG_COUNT = 0;
        private static final int ERROR_COUNT = 1;
        private static final int WARN_COUNT = 2;
        private static final int FATAL_COUNT = 3;
        private static final int UNIQUE_TEMPLATE_COUNT = 4;
        private static final int NOVEL_TEMPLATE_COUNT = 5;
        private static final int LARGEST_NOVEL_TEMPLATE_COUNT = 6;
        private static final int ERROR_RATIO = 7;
        private static final int DOMINANT_TEMPLATE_RATIO = 8;
        private static final int DIMENSIONS = 9;

        private final SimpleDrainParser parser;
        private final Set<Integer> observedTemplates = new LinkedHashSet<>();

        private DrainBucketVectorizer(double similarity) {
            this.parser = new SimpleDrainParser(similarity);
        }

        private DrainBucketStats vectorize(List<String> messages) {
            Map<Integer, Integer> templateCounts = new HashMap<>();
            Map<Integer, Integer> severityTemplateCounts = new HashMap<>();
            Set<Integer> currentTemplates = new LinkedHashSet<>();
            int logCount = 0;
            int errorCount = 0;
            int warnCount = 0;
            int fatalCount = 0;

            for (String message : messages) {
                int templateId = parser.parse(message);
                ++logCount;
                currentTemplates.add(templateId);
                templateCounts.put(templateId, templateCounts.containsKey(templateId) ? templateCounts.get(templateId) + 1
                        : 1);
                Severity severity = Severity.from(message);
                if (severity.error) {
                    ++errorCount;
                }
                if (severity.warn) {
                    ++warnCount;
                }
                if (severity.fatal) {
                    ++fatalCount;
                }
                if (severity.error || severity.warn || severity.fatal) {
                    severityTemplateCounts.put(templateId,
                            severityTemplateCounts.containsKey(templateId) ? severityTemplateCounts.get(templateId) + 1
                                    : 1);
                }
            }

            int novelTemplateCount = 0;
            int largestNovelTemplateCount = 0;
            int dominantTemplateCount = 0;
            Set<Integer> novelTemplates = new LinkedHashSet<>();
            for (Map.Entry<Integer, Integer> entry : templateCounts.entrySet()) {
                dominantTemplateCount = Math.max(dominantTemplateCount, entry.getValue());
                if (!observedTemplates.contains(entry.getKey())) {
                    novelTemplates.add(entry.getKey());
                    ++novelTemplateCount;
                    largestNovelTemplateCount = Math.max(largestNovelTemplateCount, entry.getValue());
                }
            }
            observedTemplates.addAll(currentTemplates);
            return new DrainBucketStats(logCount, errorCount, warnCount, fatalCount, currentTemplates.size(),
                    novelTemplateCount, largestNovelTemplateCount, dominantTemplateCount, templateCounts,
                    severityTemplateCounts, novelTemplates);
        }

        private int templateCount() {
            return parser.templateCount();
        }
    }

    private static final class DrainBucketStats {
        private final int logCount;
        private final int errorCount;
        private final int warnCount;
        private final int fatalCount;
        private final int uniqueTemplateCount;
        private final int novelTemplateCount;
        private final int largestNovelTemplateCount;
        private final int dominantTemplateCount;
        private final Map<Integer, Integer> templateCounts;
        private final Map<Integer, Integer> severityTemplateCounts;
        private final Set<Integer> novelTemplates;

        private DrainBucketStats(int logCount, int errorCount, int warnCount, int fatalCount, int uniqueTemplateCount,
                int novelTemplateCount, int largestNovelTemplateCount, int dominantTemplateCount,
                Map<Integer, Integer> templateCounts, Map<Integer, Integer> severityTemplateCounts,
                Set<Integer> novelTemplates) {
            this.logCount = logCount;
            this.errorCount = errorCount;
            this.warnCount = warnCount;
            this.fatalCount = fatalCount;
            this.uniqueTemplateCount = uniqueTemplateCount;
            this.novelTemplateCount = novelTemplateCount;
            this.largestNovelTemplateCount = largestNovelTemplateCount;
            this.dominantTemplateCount = dominantTemplateCount;
            this.templateCounts = templateCounts;
            this.severityTemplateCounts = severityTemplateCounts;
            this.novelTemplates = novelTemplates;
        }

        private double errorRatio() {
            return logCount == 0 ? 0 : (double) errorCount / logCount;
        }

        private double[] toVector() {
            double[] vector = new double[DrainBucketVectorizer.DIMENSIONS];
            vector[DrainBucketVectorizer.LOG_COUNT] = logCount;
            vector[DrainBucketVectorizer.ERROR_COUNT] = errorCount;
            vector[DrainBucketVectorizer.WARN_COUNT] = warnCount;
            vector[DrainBucketVectorizer.FATAL_COUNT] = fatalCount;
            vector[DrainBucketVectorizer.UNIQUE_TEMPLATE_COUNT] = uniqueTemplateCount;
            vector[DrainBucketVectorizer.NOVEL_TEMPLATE_COUNT] = novelTemplateCount;
            vector[DrainBucketVectorizer.LARGEST_NOVEL_TEMPLATE_COUNT] = largestNovelTemplateCount;
            vector[DrainBucketVectorizer.ERROR_RATIO] = errorRatio();
            vector[DrainBucketVectorizer.DOMINANT_TEMPLATE_RATIO] = logCount == 0 ? 0
                    : (double) dominantTemplateCount / logCount;
            return vector;
        }
    }

    private static final class SimpleDrainParser {
        private static final Pattern UUID = Pattern
                .compile("(?i)\\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\\b");
        private static final Pattern IP = Pattern.compile("\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b");
        private static final Pattern BLOCK = Pattern.compile("\\bblk_-?\\d+\\b");
        private static final Pattern HEX = Pattern.compile("(?i)\\b(?:0x)?[0-9a-f]*[a-f][0-9a-f]{7,}\\b");
        private static final Pattern HAS_DIGIT = Pattern.compile(".*\\d.*");

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
                ++best.count;
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
            String[] rawTokens = message.trim().toLowerCase(Locale.ROOT).split("\\s+");
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
            if (containsDigit(rawToken)) {
                return "<*>";
            }
            String token = rawToken.toLowerCase(Locale.ROOT);
            return trimToken(token);
        }

        private static boolean containsDigit(String value) {
            for (int i = 0; i < value.length(); i++) {
                if (Character.isDigit(value.charAt(i))) {
                    return true;
                }
            }
            return false;
        }

        private static String trimToken(String token) {
            int start = 0;
            int end = token.length() - 1;
            while (start <= end && !isTemplateCharacter(token.charAt(start))) {
                ++start;
            }
            while (end >= start && !isTemplateCharacter(token.charAt(end))) {
                --end;
            }
            return start > end ? "" : token.substring(start, end + 1);
        }

        private static boolean isTemplateCharacter(char value) {
            return value >= 'a' && value <= 'z' || value >= '0' && value <= '9' || value == '<' || value == '>'
                    || value == '*';
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
        private int count;

        private DrainCluster(int id, List<String> template) {
            this.id = id;
            this.template = new ArrayList<>(template);
            this.count = 1;
        }
    }

    private static final class Severity {
        private final boolean error;
        private final boolean warn;
        private final boolean fatal;

        private Severity(boolean error, boolean warn, boolean fatal) {
            this.error = error;
            this.warn = warn;
            this.fatal = fatal;
        }

        private static Severity from(String message) {
            String lower = message == null ? "" : message.toLowerCase(Locale.ROOT);
            boolean fatal = containsWord(lower, "fatal") || containsWord(lower, "critical")
                    || containsWord(lower, "severe") || containsWord(lower, "panic");
            boolean error = fatal || containsWord(lower, "error") || containsWord(lower, "exception")
                    || containsWord(lower, "failed") || containsWord(lower, "failure")
                    || containsWord(lower, "corrupt");
            boolean warn = !error && (containsWord(lower, "warn") || containsWord(lower, "warning")
                    || containsWord(lower, "timeout") || containsWord(lower, "retry")
                    || containsWord(lower, "degraded"));
            return new Severity(error, warn, fatal);
        }

        private static boolean containsWord(String text, String word) {
            int from = 0;
            while (from < text.length()) {
                int index = text.indexOf(word, from);
                if (index < 0) {
                    return false;
                }
                int before = index - 1;
                int after = index + word.length();
                if ((before < 0 || !isWordCharacter(text.charAt(before)))
                        && (after >= text.length() || !isWordCharacter(text.charAt(after)))) {
                    return true;
                }
                from = index + 1;
            }
            return false;
        }

        private static boolean isWordCharacter(char value) {
            return value == '_' || value >= 'a' && value <= 'z' || value >= '0' && value <= '9';
        }
    }

    private static final class EwmaBaseline {
        private final double decay;
        private final double[] mean;
        private final double[] variance;
        private int count;

        private EwmaBaseline(int dimensions, double decay) {
            this.decay = decay;
            this.mean = new double[dimensions];
            this.variance = new double[dimensions];
        }

        private void update(double[] vector) {
            ++count;
            if (count == 1) {
                System.arraycopy(vector, 0, mean, 0, vector.length);
                Arrays.fill(variance, 1.0);
                return;
            }
            for (int i = 0; i < vector.length; i++) {
                double previousMean = mean[i];
                double delta = vector[i] - previousMean;
                mean[i] = previousMean + decay * delta;
                variance[i] = (1 - decay) * (variance[i] + decay * delta * delta);
            }
        }

        private double mean(int index) {
            return mean[index];
        }

        private double stdDev(int index) {
            return Math.sqrt(Math.max(variance[index], 1e-6));
        }
    }

    private static final class TemplateCountBaseline {
        private final double decay;
        private final Map<Integer, TemplateStat> stats = new HashMap<>();
        private int count;

        private TemplateCountBaseline(double decay) {
            this.decay = decay;
        }

        private void update(DrainBucketStats current, int templateCount) {
            ++count;
            for (int templateId = 0; templateId < templateCount; templateId++) {
                TemplateStat stat = stats.get(templateId);
                if (stat == null) {
                    stat = new TemplateStat();
                    stats.put(templateId, stat);
                }
                Integer value = current.templateCounts.get(templateId);
                stat.update(value == null ? 0.0 : value.doubleValue(), decay);
            }
        }
    }

    private static final class TemplateStat {
        private double mean;
        private double variance = 1.0;
        private boolean initialized;

        private void update(double value, double decay) {
            if (!initialized) {
                mean = value;
                variance = 1.0;
                initialized = true;
                return;
            }
            double previousMean = mean;
            double delta = value - previousMean;
            mean = previousMean + decay * delta;
            variance = (1 - decay) * (variance + decay * delta * delta);
        }

        private double stdDev() {
            return Math.sqrt(Math.max(variance, 1e-6));
        }
    }

    private static final class Bucket {
        private final long bucketKey;
        private final long startEpochSeconds;
        private final long ordinal;
        private final List<String> messages = new ArrayList<>();
        private boolean actualAnomaly;
        private long anomalousLogCount;

        private Bucket(long bucketKey, long startEpochSeconds, long ordinal) {
            this.bucketKey = bucketKey;
            this.startEpochSeconds = startEpochSeconds;
            this.ordinal = ordinal;
        }

        private void add(String message, boolean anomaly) {
            messages.add(message);
            if (anomaly) {
                actualAnomaly = true;
                ++anomalousLogCount;
            }
        }
    }

    private static final class ParsedLine {
        private final long epochSeconds;
        private final boolean anomaly;
        private final String message;

        private ParsedLine(long epochSeconds, boolean anomaly, String message) {
            this.epochSeconds = epochSeconds;
            this.anomaly = anomaly;
            this.message = message;
        }
    }

    private static final class Detection {
        private final boolean anomaly;
        private final double grade;

        private Detection(boolean anomaly, double grade) {
            this.anomaly = anomaly;
            this.grade = grade;
        }
    }

    private static final class DatasetSummary {
        private long bucketCount;
        private long positiveBucketCount;
        private long logCount;
        private long anomalousLogCount;
    }

    private static final class Metrics {
        private final String detector;
        private final long warmupBuckets;
        private long evaluatedBuckets;
        private long evaluatedLogs;
        private long positives;
        private long predicted;
        private long truePositive;
        private long falsePositive;
        private long trueNegative;
        private long falseNegative;
        private long nanos;
        private String finalStateSize = "";

        private Metrics(String detector, long warmupBuckets) {
            this.detector = detector;
            this.warmupBuckets = warmupBuckets;
        }

        private void record(Bucket bucket, boolean prediction, long elapsedNanos, String stateSize) {
            nanos += elapsedNanos;
            finalStateSize = stateSize;
            if (bucket.ordinal < warmupBuckets) {
                return;
            }
            ++evaluatedBuckets;
            evaluatedLogs += bucket.messages.size();
            if (bucket.actualAnomaly) {
                ++positives;
            }
            if (prediction) {
                ++predicted;
            }
            if (bucket.actualAnomaly && prediction) {
                ++truePositive;
            } else if (!bucket.actualAnomaly && prediction) {
                ++falsePositive;
            } else if (bucket.actualAnomaly) {
                ++falseNegative;
            } else {
                ++trueNegative;
            }
        }

        private double precision() {
            return truePositive + falsePositive == 0 ? 0 : (double) truePositive / (truePositive + falsePositive);
        }

        private double recall() {
            return truePositive + falseNegative == 0 ? 0 : (double) truePositive / (truePositive + falseNegative);
        }

        private double f1() {
            double precision = precision();
            double recall = recall();
            return precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
        }

        private double avgMsPerBucket() {
            long processedBuckets = evaluatedBuckets + warmupBuckets;
            return processedBuckets == 0 ? 0 : nanos / 1_000_000.0 / processedBuckets;
        }

        private double avgMicrosPerLog() {
            return evaluatedLogs == 0 ? 0 : nanos / 1_000.0 / evaluatedLogs;
        }

        private static String header() {
            return "detector,evaluated_buckets,positive_buckets,predicted_buckets,tp,fp,fn,tn,precision,recall,f1,avg_ms_per_bucket,avg_us_per_log,state";
        }

        private String toCsv() {
            return String.format(Locale.ROOT,
                    "%s,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%s", detector,
                    evaluatedBuckets, positives, predicted, truePositive, falsePositive, falseNegative, trueNegative,
                    precision(), recall(), f1(), avgMsPerBucket(), avgMicrosPerLog(), finalStateSize);
        }
    }

    private static final class Config {
        private String dataset = "bgl";
        private String input;
        private String labels;
        private String output;
        private long bucketSeconds = 600;
        private int minimumBaselineBuckets = 80;
        private int warmupBuckets = 80;
        private long seed = 42L;
        private double drainSimilarity = 0.5;
        private boolean includeEmptyBuckets;
        private List<String> detectors = new ArrayList<>(
                Arrays.asList("current", "current_update", "drain_ewma", "drain_signal", "drain_template",
                        "drain_rcf"));

        private static Config parse(String[] args) {
            Config config = new Config();
            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                if ("--dataset".equals(arg)) {
                    config.dataset = next(args, ++i, arg).toLowerCase(Locale.ROOT);
                } else if ("--input".equals(arg)) {
                    config.input = next(args, ++i, arg);
                } else if ("--labels".equals(arg)) {
                    config.labels = next(args, ++i, arg);
                } else if ("--output".equals(arg)) {
                    config.output = next(args, ++i, arg);
                } else if ("--bucket-seconds".equals(arg)) {
                    config.bucketSeconds = Long.parseLong(next(args, ++i, arg));
                } else if ("--minimum-baseline-buckets".equals(arg)) {
                    config.minimumBaselineBuckets = Integer.parseInt(next(args, ++i, arg));
                } else if ("--warmup-buckets".equals(arg)) {
                    config.warmupBuckets = Integer.parseInt(next(args, ++i, arg));
                } else if ("--seed".equals(arg)) {
                    config.seed = Long.parseLong(next(args, ++i, arg));
                } else if ("--drain-similarity".equals(arg)) {
                    config.drainSimilarity = Double.parseDouble(next(args, ++i, arg));
                } else if ("--detectors".equals(arg)) {
                    config.detectors = parseDetectors(next(args, ++i, arg));
                } else if ("--include-empty-buckets".equals(arg)) {
                    config.includeEmptyBuckets = true;
                } else if ("--help".equals(arg)) {
                    usageAndExit();
                } else {
                    throw new IllegalArgumentException("unknown argument: " + arg);
                }
            }
            if (config.input == null) {
                throw new IllegalArgumentException("--input is required");
            }
            return config;
        }

        private static List<String> parseDetectors(String value) {
            List<String> detectors = new ArrayList<>();
            for (String detector : value.split(",")) {
                String normalized = detector.trim().toLowerCase(Locale.ROOT);
                if (!normalized.isEmpty()) {
                    detectors.add(normalized);
                }
            }
            if (detectors.isEmpty()) {
                throw new IllegalArgumentException("--detectors must include at least one detector");
            }
            return detectors;
        }

        private static String next(String[] args, int index, String flag) {
            if (index >= args.length) {
                throw new IllegalArgumentException(flag + " requires a value");
            }
            return args[index];
        }

        private static void usageAndExit() {
            System.out.println("Usage: LogAnomalyBenchmark --dataset bgl|hdfs --input <raw log> [options]");
            System.out.println("Options:");
            System.out.println("  --labels <csv>                    HDFS anomaly_label.csv");
            System.out.println("  --bucket-seconds <seconds>        default 600");
            System.out.println("  --minimum-baseline-buckets <n>    default 80");
            System.out.println("  --warmup-buckets <n>              default 80");
            System.out.println("  --drain-similarity <x>            default 0.5");
            System.out.println(
                    "  --detectors <csv>                 current,current_update,regex_ewma,drain_ewma,drain_signal,drain_template,drain_rcf");
            System.out.println("  --include-empty-buckets           include empty time buckets");
            System.out.println("  --output <csv>");
            System.exit(0);
        }
    }
}
