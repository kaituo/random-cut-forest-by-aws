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

import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;

public class LogAnomalyResult {

    private final AnomalyDescriptor rcfResult;
    private final LogBucketStats currentBucketStats;
    private final LogBucketSummaryStats baselineBucketStats;
    private final LogAnomalySummary summary;
    private final boolean anomaly;
    private final double anomalyGrade;
    private final boolean modelTriggered;
    private final boolean logStatTriggered;

    LogAnomalyResult(AnomalyDescriptor rcfResult, LogBucketStats currentBucketStats,
            LogBucketSummaryStats baselineBucketStats, LogAnomalySummary summary, boolean anomaly, double anomalyGrade,
            boolean modelTriggered, boolean logStatTriggered) {
        this.rcfResult = rcfResult;
        this.currentBucketStats = currentBucketStats;
        this.baselineBucketStats = baselineBucketStats;
        this.summary = summary;
        this.anomaly = anomaly;
        this.anomalyGrade = anomalyGrade;
        this.modelTriggered = modelTriggered;
        this.logStatTriggered = logStatTriggered;
    }

    public AnomalyDescriptor getRcfResult() {
        return rcfResult;
    }

    public LogBucketStats getCurrentBucketStats() {
        return currentBucketStats;
    }

    public LogBucketSummaryStats getBaselineBucketStats() {
        return baselineBucketStats;
    }

    public LogAnomalySummary getSummary() {
        return summary;
    }

    public boolean isAnomaly() {
        return anomaly;
    }

    public double getAnomalyGrade() {
        return anomalyGrade;
    }

    public boolean isModelTriggered() {
        return modelTriggered;
    }

    public boolean isLogStatTriggered() {
        return logStatTriggered;
    }
}
