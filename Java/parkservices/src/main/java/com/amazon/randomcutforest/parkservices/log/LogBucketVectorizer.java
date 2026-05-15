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

import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

import com.amazon.randomcutforest.parkservices.log.LogRecordNormalizer.NormalizedLogRecord;

/**
 * Converts one time bucket of log messages into the fixed numeric vector used
 * by the RCF model. Template novelty is maintained on this object so each
 * detector or entity stream should own its own vectorizer instance.
 */
public class LogBucketVectorizer {

    private final LogRecordNormalizer normalizer = new LogRecordNormalizer();
    private final Set<String> observedTemplateIds = new LinkedHashSet<>();

    public LogBucketStats vectorize(Collection<String> messages) {
        Collection<String> safeMessages = messages == null ? Collections.emptyList() : messages;
        Map<String, Integer> templateCounts = new LinkedHashMap<>();
        Map<String, String> templateExamples = new LinkedHashMap<>();
        Map<String, Integer> exceptionCounts = new LinkedHashMap<>();
        Set<String> currentTemplateIds = new LinkedHashSet<>();
        int logCount = 0;
        int errorCount = 0;
        int warnCount = 0;

        for (String message : safeMessages) {
            NormalizedLogRecord normalized = normalizer.normalize(message);
            String templateId = normalized.getTemplateId();
            ++logCount;
            if (normalized.isError()) {
                ++errorCount;
            }
            if (normalized.isWarn()) {
                ++warnCount;
            }
            currentTemplateIds.add(templateId);
            templateCounts.merge(templateId, 1, Integer::sum);
            templateExamples.putIfAbsent(templateId, normalized.getCanonicalTemplate());
            if (normalized.getExceptionName() != null) {
                exceptionCounts.merge(normalized.getExceptionName(), 1, Integer::sum);
            }
        }

        Set<String> novelTemplateIds = new LinkedHashSet<>();
        for (String templateId : currentTemplateIds) {
            if (!observedTemplateIds.contains(templateId)) {
                novelTemplateIds.add(templateId);
            }
        }
        observedTemplateIds.addAll(currentTemplateIds);

        return new LogBucketStats(logCount, errorCount, warnCount, currentTemplateIds.size(), exceptionCounts.size(),
                novelTemplateIds.size(), templateCounts, templateExamples, novelTemplateIds, exceptionCounts);
    }

    public void clearObservedTemplates() {
        observedTemplateIds.clear();
    }

    public int getObservedTemplateCount() {
        return observedTemplateIds.size();
    }
}
