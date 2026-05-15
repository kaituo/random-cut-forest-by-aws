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

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.util.Arrays;

public class LogBucketSummaryStats {

    private final double[] values;

    private LogBucketSummaryStats(double[] values) {
        checkArgument(values != null && values.length == LogVectorSchema.DIMENSIONS, "incorrect log vector length");
        this.values = Arrays.copyOf(values, values.length);
    }

    public static LogBucketSummaryStats fromVector(double[] values) {
        return new LogBucketSummaryStats(values);
    }

    public static LogBucketSummaryStats zero() {
        return new LogBucketSummaryStats(new double[LogVectorSchema.DIMENSIONS]);
    }

    public static LogBucketSummaryStats fromBucketStats(LogBucketStats stats) {
        return new LogBucketSummaryStats(stats.toVector());
    }

    public double getFeatureValue(int index) {
        checkArgument(index >= 0 && index < LogVectorSchema.DIMENSIONS, "invalid log vector feature index");
        return values[index];
    }

    public double[] toVector() {
        return Arrays.copyOf(values, values.length);
    }

    public double getLogCount() {
        return values[LogVectorSchema.LOG_COUNT];
    }

    public double getErrorCount() {
        return values[LogVectorSchema.ERROR_COUNT];
    }

    public double getWarnCount() {
        return values[LogVectorSchema.WARN_COUNT];
    }

    public double getUniqueTemplateCount() {
        return values[LogVectorSchema.UNIQUE_TEMPLATE_COUNT];
    }

    public double getUniqueExceptionCount() {
        return values[LogVectorSchema.UNIQUE_EXCEPTION_COUNT];
    }

    public double getErrorRatio() {
        return values[LogVectorSchema.ERROR_RATIO];
    }

    public double getNovelTemplateCount() {
        return values[LogVectorSchema.NOVEL_TEMPLATE_COUNT];
    }

    @Override
    public String toString() {
        return Arrays.toString(values);
    }
}
