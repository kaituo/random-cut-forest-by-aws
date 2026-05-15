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
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

public class LogBucketStats {

    private final int logCount;
    private final int errorCount;
    private final int warnCount;
    private final int uniqueTemplateCount;
    private final int uniqueExceptionCount;
    private final double errorRatio;
    private final int novelTemplateCount;
    private final Map<String, Integer> templateCounts;
    private final Map<String, String> templateExamples;
    private final Set<String> novelTemplateIds;
    private final Map<String, Integer> exceptionCounts;

    LogBucketStats(int logCount, int errorCount, int warnCount, int uniqueTemplateCount, int uniqueExceptionCount,
            int novelTemplateCount, Map<String, Integer> templateCounts, Map<String, String> templateExamples,
            Set<String> novelTemplateIds, Map<String, Integer> exceptionCounts) {
        this.logCount = logCount;
        this.errorCount = errorCount;
        this.warnCount = warnCount;
        this.uniqueTemplateCount = uniqueTemplateCount;
        this.uniqueExceptionCount = uniqueExceptionCount;
        this.errorRatio = logCount == 0 ? 0 : (double) errorCount / logCount;
        this.novelTemplateCount = novelTemplateCount;
        this.templateCounts = Collections.unmodifiableMap(new LinkedHashMap<>(templateCounts));
        this.templateExamples = Collections.unmodifiableMap(new LinkedHashMap<>(templateExamples));
        this.novelTemplateIds = Collections.unmodifiableSet(new LinkedHashSet<>(novelTemplateIds));
        this.exceptionCounts = Collections.unmodifiableMap(new LinkedHashMap<>(exceptionCounts));
    }

    public static LogBucketStats empty() {
        return new LogBucketStats(0, 0, 0, 0, 0, 0, Collections.emptyMap(), Collections.emptyMap(),
                Collections.emptySet(), Collections.emptyMap());
    }

    public double[] toVector() {
        double[] answer = new double[LogVectorSchema.DIMENSIONS];
        answer[LogVectorSchema.LOG_COUNT] = logCount;
        answer[LogVectorSchema.ERROR_COUNT] = errorCount;
        answer[LogVectorSchema.WARN_COUNT] = warnCount;
        answer[LogVectorSchema.UNIQUE_TEMPLATE_COUNT] = uniqueTemplateCount;
        answer[LogVectorSchema.UNIQUE_EXCEPTION_COUNT] = uniqueExceptionCount;
        answer[LogVectorSchema.ERROR_RATIO] = errorRatio;
        answer[LogVectorSchema.NOVEL_TEMPLATE_COUNT] = novelTemplateCount;
        return answer;
    }

    public double getFeatureValue(int index) {
        checkArgument(index >= 0 && index < LogVectorSchema.DIMENSIONS, "invalid log vector feature index");
        return toVector()[index];
    }

    public int getLargestNovelTemplateCount() {
        int max = 0;
        for (String templateId : novelTemplateIds) {
            Integer count = templateCounts.get(templateId);
            if (count != null && count > max) {
                max = count;
            }
        }
        return max;
    }

    public int getLogCount() {
        return logCount;
    }

    public int getErrorCount() {
        return errorCount;
    }

    public int getWarnCount() {
        return warnCount;
    }

    public int getUniqueTemplateCount() {
        return uniqueTemplateCount;
    }

    public int getUniqueExceptionCount() {
        return uniqueExceptionCount;
    }

    public double getErrorRatio() {
        return errorRatio;
    }

    public int getNovelTemplateCount() {
        return novelTemplateCount;
    }

    public Map<String, Integer> getTemplateCounts() {
        return templateCounts;
    }

    public Map<String, String> getTemplateExamples() {
        return templateExamples;
    }

    public Set<String> getNovelTemplateIds() {
        return novelTemplateIds;
    }

    public Map<String, Integer> getExceptionCounts() {
        return exceptionCounts;
    }

    @Override
    public String toString() {
        return Arrays.toString(toVector());
    }
}
