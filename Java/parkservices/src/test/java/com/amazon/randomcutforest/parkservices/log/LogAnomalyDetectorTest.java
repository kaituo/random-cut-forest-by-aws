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

package com.amazon.randomcutforest.parkservices.log;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.parkservices.log.LogRecordNormalizer.NormalizedLogRecord;

public class LogAnomalyDetectorTest {

    @Test
    void normalizerProducesStableTemplatesAcrossVariableFields() {
        LogRecordNormalizer normalizer = new LogRecordNormalizer();

        NormalizedLogRecord first = normalizer.normalize(
                "ERROR checkout failed orderId=1234567 userId=7654321 from 10.2.3.4 request=550e8400-e29b-41d4-a716-446655440000");
        NormalizedLogRecord second = normalizer.normalize(
                "ERROR checkout failed orderId=2234567 userId=8654321 from 10.5.6.7 request=650e8400-e29b-41d4-a716-446655440111");

        assertEquals(first.getTemplateId(), second.getTemplateId());
        assertEquals("error", first.getLevel());
        assertTrue(first.isError());
        assertFalse(first.isWarn());
        assertTrue(first.getCanonicalTemplate().contains("<num>"));
        assertTrue(first.getCanonicalTemplate().contains("<ip>"));
        assertTrue(first.getCanonicalTemplate().contains("<uuid>"));
    }

    @Test
    void vectorizerTracksPerStreamNovelTemplates() {
        LogBucketVectorizer vectorizer = new LogBucketVectorizer();

        LogBucketStats first = vectorizer.vectorize(Arrays.asList(
                "level=INFO event=inventory_reserved sku=SKU-1 request_id=550e8400-e29b-41d4-a716-446655440000",
                "level=INFO event=inventory_reserved sku=SKU-2 request_id=550e8400-e29b-41d4-a716-446655440001"));
        LogBucketStats second = vectorizer.vectorize(Collections.singletonList(
                "level=INFO event=inventory_reserved sku=SKU-3 request_id=550e8400-e29b-41d4-a716-446655440002"));

        assertEquals(1, first.getUniqueTemplateCount());
        assertEquals(1, first.getNovelTemplateCount());
        assertEquals(0, second.getNovelTemplateCount());
        assertEquals(LogVectorSchema.DIMENSIONS, first.toVector().length);
    }

    @Test
    void summaryClassifiesLogAnomalyTypes() {
        LogAnomalySummaryBuilder builder = new LogAnomalySummaryBuilder();
        LogBucketSummaryStats baseline = LogBucketSummaryStats.fromVector(new double[] { 80, 1, 1, 4, 0, 1.0 / 80, 0 });

        assertEquals(LogAnomalyType.NEW_PATTERN, builder.build(stats(80, 1, 1, 5, 0, 1, 6), baseline).getAnomalyType());
        assertEquals(LogAnomalyType.VOLUME_SHIFT,
                builder.build(stats(130, 1, 1, 4, 0, 0, 0), baseline).getAnomalyType());
        assertEquals(LogAnomalyType.ERROR_BURST, builder.build(stats(80, 7, 1, 4, 0, 0, 0), baseline).getAnomalyType());
        assertEquals(LogAnomalyType.WARN_BURST, builder.build(stats(80, 1, 7, 4, 0, 0, 0), baseline).getAnomalyType());
        assertEquals(LogAnomalyType.EXCEPTION_BURST,
                builder.build(stats(80, 1, 1, 4, 1, 0, 0), baseline).getAnomalyType());
        assertEquals(LogAnomalyType.MIXED, builder.build(stats(130, 8, 1, 5, 1, 1, 6), baseline).getAnomalyType());
    }

    @Test
    void representativeBenchmarkMeetsPrecisionAndRecallFloor() {
        LogAnomalyDetector detector = LogAnomalyDetector.builder().randomSeed(42L).minimumBaselineBuckets(100)
                .modelAlertGrade(0.8).build();
        Random random = new Random(1729L);

        for (int bucket = 0; bucket < 220; bucket++) {
            detector.process(normalBucket(bucket, random), bucket);
        }

        int truePositive = 0;
        int falsePositive = 0;
        int falseNegative = 0;
        int actualPositive = 0;
        int predictedPositive = 0;

        for (int bucket = 0; bucket < 180; bucket++) {
            SyntheticBucket syntheticBucket = benchmarkBucket(bucket, random);
            LogAnomalyResult result = detector.process(syntheticBucket.messages, 220L + bucket);

            if (syntheticBucket.anomaly) {
                ++actualPositive;
            }
            if (result.isAnomaly()) {
                ++predictedPositive;
            }
            if (syntheticBucket.anomaly && result.isAnomaly()) {
                ++truePositive;
            } else if (!syntheticBucket.anomaly && result.isAnomaly()) {
                ++falsePositive;
            } else if (syntheticBucket.anomaly) {
                ++falseNegative;
            }
        }

        double precision = (double) truePositive / Math.max(1, truePositive + falsePositive);
        double recall = (double) truePositive / Math.max(1, truePositive + falseNegative);

        assertTrue(actualPositive >= 10, "benchmark must contain enough labeled incidents");
        assertTrue(predictedPositive > 0, "detector must emit benchmark alerts");
        assertTrue(precision >= 0.5, "precision=" + precision);
        assertTrue(recall >= 0.5, "recall=" + recall);
    }

    private LogBucketStats stats(int logCount, int errorCount, int warnCount, int uniqueTemplateCount,
            int uniqueExceptionCount, int novelTemplateCount, int largestNovelCount) {
        Map<String, Integer> templateCounts = new LinkedHashMap<>();
        Map<String, String> templateExamples = new LinkedHashMap<>();
        Set<String> novelTemplateIds = new LinkedHashSet<>();
        for (int i = 0; i < uniqueTemplateCount; i++) {
            String templateId = "template-" + i;
            templateCounts.put(templateId, i == 0 ? Math.max(1, largestNovelCount) : 1);
            templateExamples.put(templateId, "template " + i);
            if (i < novelTemplateCount) {
                novelTemplateIds.add(templateId);
            }
        }
        Map<String, Integer> exceptionCounts = new LinkedHashMap<>();
        for (int i = 0; i < uniqueExceptionCount; i++) {
            exceptionCounts.put("Exception" + i, 1);
        }
        return new LogBucketStats(logCount, errorCount, warnCount, uniqueTemplateCount, uniqueExceptionCount,
                novelTemplateCount, templateCounts, templateExamples, novelTemplateIds, exceptionCounts);
    }

    private SyntheticBucket benchmarkBucket(int bucket, Random random) {
        if (bucket % 17 == 4) {
            return new SyntheticBucket(errorBurst(bucket, random), true);
        }
        if (bucket % 29 == 8) {
            return new SyntheticBucket(warnBurst(bucket, random), true);
        }
        if (bucket % 31 == 12) {
            return new SyntheticBucket(newPattern(bucket, random), true);
        }
        if (bucket % 37 == 16) {
            return new SyntheticBucket(exceptionBurst(bucket, random), true);
        }
        if (bucket % 41 == 20) {
            return new SyntheticBucket(volumeShift(bucket, random), true);
        }
        if (bucket % 53 == 24) {
            return new SyntheticBucket(mixedIncident(bucket, random), true);
        }
        return new SyntheticBucket(normalBucket(bucket, random), false);
    }

    private List<String> normalBucket(int bucket, Random random) {
        int count = 80 + (int) Math.round(4 * Math.sin(bucket / 12.0)) + random.nextInt(5) - 2;
        List<String> messages = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            messages.add(normalMessage(bucket, i, random));
        }
        if (bucket % 9 == 0) {
            messages.add("WARN checkout retryable timeout attempt=1 orderId=" + longId(bucket, 900));
        }
        if (bucket % 31 == 0) {
            messages.add("ERROR checkout dependency failed status=503 orderId=" + longId(bucket, 901));
        }
        return messages;
    }

    private String normalMessage(int bucket, int index, Random random) {
        int selector = index % 4;
        if (selector == 0) {
            return "INFO checkout completed orderId=" + longId(bucket, index) + " userId=" + longId(bucket, index + 7)
                    + " status=200 latencyMs=" + (1000 + random.nextInt(9000));
        }
        if (selector == 1) {
            return "{\"level\":\"INFO\",\"message\":\"payment authorized orderId=" + longId(bucket, index)
                    + " accountId=" + longId(bucket, index + 13) + " status=200\"}";
        }
        if (selector == 2) {
            return "level=INFO event=inventory_reserved sku=SKU-" + (100 + random.nextInt(900)) + " request_id="
                    + uuid(bucket, index);
        }
        return "INFO checkout cache hit key=0xabcdefff" + (1000 + random.nextInt(9000));
    }

    private List<String> errorBurst(int bucket, Random random) {
        List<String> messages = normalBucket(bucket, random);
        for (int i = 0; i < 14; i++) {
            messages.add("ERROR checkout dependency failed status=503 orderId=" + longId(bucket, 1000 + i));
        }
        return messages;
    }

    private List<String> warnBurst(int bucket, Random random) {
        List<String> messages = normalBucket(bucket, random);
        for (int i = 0; i < 16; i++) {
            messages.add("WARN checkout retryable timeout attempt=1 orderId=" + longId(bucket, 1100 + i));
        }
        return messages;
    }

    private List<String> newPattern(int bucket, Random random) {
        List<String> messages = normalBucket(bucket, random);
        for (int i = 0; i < 10; i++) {
            messages.add("INFO checkout fraud hold policy=manual_review orderId=" + longId(bucket, 1200 + i));
        }
        return messages;
    }

    private List<String> exceptionBurst(int bucket, Random random) {
        List<String> messages = normalBucket(bucket, random);
        for (int i = 0; i < 8; i++) {
            messages.add("ERROR com.amazon.checkout.PaymentTimeoutException payment provider timeout orderId="
                    + longId(bucket, 1300 + i));
        }
        return messages;
    }

    private List<String> volumeShift(int bucket, Random random) {
        List<String> messages = new ArrayList<>();
        for (int i = 0; i < 145; i++) {
            messages.add(normalMessage(bucket, i, random));
        }
        return messages;
    }

    private List<String> mixedIncident(int bucket, Random random) {
        List<String> messages = volumeShift(bucket, random);
        for (int i = 0; i < 12; i++) {
            messages.add("ERROR com.amazon.checkout.InventoryMismatchException reservation failed orderId="
                    + longId(bucket, 1400 + i));
        }
        for (int i = 0; i < 8; i++) {
            messages.add("INFO checkout fraud hold policy=manual_review orderId=" + longId(bucket, 1500 + i));
        }
        return messages;
    }

    private long longId(int bucket, int offset) {
        return 1_000_000L + 10_000L * bucket + offset;
    }

    private String uuid(int bucket, int index) {
        return String.format("550e8400-e29b-41d4-a716-%012d", 100_000L + bucket * 1_000L + index);
    }

    private static class SyntheticBucket {
        private final List<String> messages;
        private final boolean anomaly;

        private SyntheticBucket(List<String> messages, boolean anomaly) {
            this.messages = messages;
            this.anomaly = anomaly;
        }
    }
}
