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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import com.amazon.randomcutforest.parkservices.log.LogAnomalySummary.Driver;
import com.amazon.randomcutforest.parkservices.log.LogAnomalySummary.NovelPattern;

public class LogAnomalySummaryBuilder {

    private static final int MAX_DRIVERS = 3;
    private static final int MAX_NOVEL_PATTERNS = 5;

    public LogAnomalySummary build(LogBucketStats current, LogBucketSummaryStats baseline) {
        List<Driver> drivers = topDrivers(current, baseline);
        List<NovelPattern> novelPatterns = novelPatterns(current);
        LogAnomalyType anomalyType = classify(current, baseline);
        return new LogAnomalySummary(anomalyType, current, baseline, drivers, novelPatterns);
    }

    private LogAnomalyType classify(LogBucketStats current, LogBucketSummaryStats baseline) {
        boolean newPattern = isNewPattern(current);
        boolean errorBurst = current.getErrorCount() >= baseline.getErrorCount()
                + Math.max(3.0, baseline.getErrorCount())
                || current.getErrorRatio() >= baseline.getErrorRatio() + 0.15 && current.getErrorCount() >= 3;
        boolean warnBurst = current.getWarnCount() >= baseline.getWarnCount() + Math.max(4.0, baseline.getWarnCount());
        boolean exceptionBurst = current.getUniqueExceptionCount() > baseline.getUniqueExceptionCount()
                && current.getUniqueExceptionCount() > 0;
        boolean volumeShift = baseline.getLogCount() > 0 && Math
                .abs(current.getLogCount() - baseline.getLogCount()) >= Math.max(20.0, 0.4 * baseline.getLogCount());

        int count = 0;
        count += newPattern ? 1 : 0;
        count += errorBurst ? 1 : 0;
        count += warnBurst ? 1 : 0;
        count += exceptionBurst ? 1 : 0;
        count += volumeShift ? 1 : 0;
        if (count > 1) {
            return LogAnomalyType.MIXED;
        }
        if (newPattern) {
            return LogAnomalyType.NEW_PATTERN;
        }
        if (errorBurst) {
            return LogAnomalyType.ERROR_BURST;
        }
        if (warnBurst) {
            return LogAnomalyType.WARN_BURST;
        }
        if (exceptionBurst) {
            return LogAnomalyType.EXCEPTION_BURST;
        }
        return LogAnomalyType.VOLUME_SHIFT;
    }

    private boolean isNewPattern(LogBucketStats current) {
        return current.getNovelTemplateCount() > 1
                || current.getNovelTemplateCount() == 1 && current.getLargestNovelTemplateCount() >= 3;
    }

    private List<Driver> topDrivers(LogBucketStats current, LogBucketSummaryStats baseline) {
        double[] currentVector = current.toVector();
        double[] baselineVector = baseline.toVector();
        List<Driver> drivers = new ArrayList<>();
        for (int i = 0; i < LogVectorSchema.DIMENSIONS; i++) {
            double contribution = contribution(i, currentVector[i], baselineVector[i]);
            drivers.add(new Driver(LogVectorSchema.featureName(i), currentVector[i], baselineVector[i], contribution));
        }
        drivers.sort(Comparator.comparingDouble(Driver::getContribution).reversed());
        if (drivers.size() > MAX_DRIVERS) {
            return new ArrayList<>(drivers.subList(0, MAX_DRIVERS));
        }
        return drivers;
    }

    private double contribution(int index, double current, double baseline) {
        if (index == LogVectorSchema.ERROR_RATIO) {
            return Math.abs(current - baseline) * 5;
        }
        return Math.abs(Math.log1p(current) - Math.log1p(baseline));
    }

    private List<NovelPattern> novelPatterns(LogBucketStats current) {
        List<NovelPattern> patterns = new ArrayList<>();
        for (String templateId : current.getNovelTemplateIds()) {
            Integer count = current.getTemplateCounts().get(templateId);
            String template = current.getTemplateExamples().get(templateId);
            patterns.add(new NovelPattern(templateId, template, count == null ? 0 : count));
        }
        patterns.sort((first, second) -> Integer.compare(second.getCount(), first.getCount()));
        if (patterns.size() > MAX_NOVEL_PATTERNS) {
            return new ArrayList<>(patterns.subList(0, MAX_NOVEL_PATTERNS));
        }
        return patterns;
    }
}
