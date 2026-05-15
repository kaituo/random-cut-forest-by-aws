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
import java.util.Collections;
import java.util.List;

public class LogAnomalySummary {

    private final LogAnomalyType anomalyType;
    private final LogBucketStats currentBucketStats;
    private final LogBucketSummaryStats baselineBucketStats;
    private final List<Driver> topDrivers;
    private final List<NovelPattern> novelPatterns;

    LogAnomalySummary(LogAnomalyType anomalyType, LogBucketStats currentBucketStats,
            LogBucketSummaryStats baselineBucketStats, List<Driver> topDrivers, List<NovelPattern> novelPatterns) {
        this.anomalyType = anomalyType;
        this.currentBucketStats = currentBucketStats;
        this.baselineBucketStats = baselineBucketStats;
        this.topDrivers = Collections.unmodifiableList(new ArrayList<>(topDrivers));
        this.novelPatterns = Collections.unmodifiableList(new ArrayList<>(novelPatterns));
    }

    public LogAnomalyType getAnomalyType() {
        return anomalyType;
    }

    public String getAnomalyTypeName() {
        return anomalyType.name().toLowerCase();
    }

    public LogBucketStats getCurrentBucketStats() {
        return currentBucketStats;
    }

    public LogBucketSummaryStats getBaselineBucketStats() {
        return baselineBucketStats;
    }

    public List<Driver> getTopDrivers() {
        return topDrivers;
    }

    public List<NovelPattern> getNovelPatterns() {
        return novelPatterns;
    }

    public static class Driver {
        private final String name;
        private final double currentValue;
        private final double baselineValue;
        private final double contribution;

        Driver(String name, double currentValue, double baselineValue, double contribution) {
            this.name = name;
            this.currentValue = currentValue;
            this.baselineValue = baselineValue;
            this.contribution = contribution;
        }

        public String getName() {
            return name;
        }

        public double getCurrentValue() {
            return currentValue;
        }

        public double getBaselineValue() {
            return baselineValue;
        }

        public double getContribution() {
            return contribution;
        }
    }

    public static class NovelPattern {
        private final String templateId;
        private final String template;
        private final int count;

        NovelPattern(String templateId, String template, int count) {
            this.templateId = templateId;
            this.template = template;
            this.count = count;
        }

        public String getTemplateId() {
            return templateId;
        }

        public String getTemplate() {
            return template;
        }

        public int getCount() {
            return count;
        }
    }
}
