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

/**
 * Fixed numeric schema used for log anomaly detection. The source text is never
 * sent directly to RCF; every bucket is transformed into this vector first.
 */
public final class LogVectorSchema {

    public static final int LOG_COUNT = 0;
    public static final int ERROR_COUNT = 1;
    public static final int WARN_COUNT = 2;
    public static final int UNIQUE_TEMPLATE_COUNT = 3;
    public static final int UNIQUE_EXCEPTION_COUNT = 4;
    public static final int ERROR_RATIO = 5;
    public static final int NOVEL_TEMPLATE_COUNT = 6;
    public static final int DIMENSIONS = 7;

    private static final String[] FEATURE_NAMES = new String[] { "log_count", "error_count", "warn_count",
            "unique_template_count", "unique_exception_count", "error_ratio", "novel_template_count" };

    private LogVectorSchema() {
    }

    public static String featureName(int index) {
        checkArgument(index >= 0 && index < DIMENSIONS, "invalid log vector feature index");
        return FEATURE_NAMES[index];
    }

    public static String[] featureNames() {
        return Arrays.copyOf(FEATURE_NAMES, FEATURE_NAMES.length);
    }
}
