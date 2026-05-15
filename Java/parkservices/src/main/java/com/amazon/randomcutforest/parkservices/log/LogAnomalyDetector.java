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
import java.util.Collection;
import java.util.Optional;

import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.config.ScoringStrategy;

/**
 * Streaming log anomaly detector that feeds fixed log bucket features into
 * {@link ThresholdedRandomCutForest}. The additional adaptive checks use the
 * same bucket vector and act as guardrails for single-bucket log incidents that
 * are operationally important but may be intentionally damped by the model.
 */
public class LogAnomalyDetector {

    private final ThresholdedRandomCutForest forest;
    private final LogBucketVectorizer vectorizer;
    private final LogAnomalySummaryBuilder summaryBuilder;
    private final AdaptiveBaseline baseline;
    private final int minimumBaselineBuckets;
    private final double modelAlertGrade;
    private final boolean updateBaselineOnAnomaly;

    private LogAnomalyDetector(Builder<?> builder) {
        this.forest = ThresholdedRandomCutForest.builder().dimensions(LogVectorSchema.DIMENSIONS * builder.shingleSize)
                .shingleSize(builder.shingleSize).sampleSize(builder.sampleSize).numberOfTrees(builder.numberOfTrees)
                .randomSeed(builder.randomSeed.orElse(0L)).outputAfter(builder.outputAfter)
                .transformMethod(TransformMethod.NORMALIZE).scoringStrategy(ScoringStrategy.MULTI_MODE_RECALL)
                .anomalyRate(builder.anomalyRate).autoAdjust(true).build();
        this.vectorizer = new LogBucketVectorizer();
        this.summaryBuilder = new LogAnomalySummaryBuilder();
        this.baseline = new AdaptiveBaseline(builder.baselineDecay);
        this.minimumBaselineBuckets = builder.minimumBaselineBuckets;
        this.modelAlertGrade = builder.modelAlertGrade;
        this.updateBaselineOnAnomaly = builder.updateBaselineOnAnomaly;
    }

    public static Builder<?> builder() {
        return new Builder<>();
    }

    public LogAnomalyResult process(Collection<String> bucketMessages, long timestamp) {
        LogBucketStats currentStats = vectorizer.vectorize(bucketMessages);
        LogBucketSummaryStats baselineStats = baseline.snapshot();
        AnomalyDescriptor rcfResult = forest.process(currentStats.toVector(), timestamp);

        boolean enoughBaseline = baseline.getCount() >= minimumBaselineBuckets;
        boolean modelTriggered = enoughBaseline && rcfResult.getAnomalyGrade() >= modelAlertGrade;
        boolean logStatTriggered = enoughBaseline && isLogStatAnomaly(currentStats);
        boolean anomaly = modelTriggered || logStatTriggered;
        double anomalyGrade = anomaly ? Math.max(rcfResult.getAnomalyGrade(), logStatTriggered ? 1.0 : 0.0) : 0.0;

        LogAnomalySummary summary = summaryBuilder.build(currentStats, baselineStats);
        if (!anomaly || updateBaselineOnAnomaly || !enoughBaseline) {
            baseline.update(currentStats.toVector());
        }
        return new LogAnomalyResult(rcfResult, currentStats, baselineStats, summary, anomaly, anomalyGrade,
                modelTriggered, logStatTriggered);
    }

    public int getBaselineBucketCount() {
        return baseline.getCount();
    }

    public ThresholdedRandomCutForest getForest() {
        return forest;
    }

    private boolean isLogStatAnomaly(LogBucketStats currentStats) {
        boolean newPattern = isNewPatternAnomaly(currentStats);
        boolean errorBurst = isHigh(LogVectorSchema.ERROR_COUNT, currentStats.getErrorCount(), 4.0)
                || currentStats.getErrorRatio() >= baseline.mean(LogVectorSchema.ERROR_RATIO) + 0.12
                        && currentStats.getErrorCount() >= 3;
        boolean warnBurst = isHigh(LogVectorSchema.WARN_COUNT, currentStats.getWarnCount(), 5.0);
        boolean exceptionBurst = currentStats.getUniqueExceptionCount() > baseline
                .mean(LogVectorSchema.UNIQUE_EXCEPTION_COUNT) && currentStats.getUniqueExceptionCount() > 0
                && currentStats.getErrorCount() >= 2;
        boolean volumeShift = isVolumeShift(currentStats.getLogCount());
        return newPattern || errorBurst || warnBurst || exceptionBurst || volumeShift;
    }

    private boolean isNewPatternAnomaly(LogBucketStats currentStats) {
        double mean = baseline.mean(LogVectorSchema.NOVEL_TEMPLATE_COUNT);
        double stdDev = baseline.stdDev(LogVectorSchema.NOVEL_TEMPLATE_COUNT);
        boolean noveltyBurst = currentStats.getNovelTemplateCount() >= mean + Math.max(2.0, 4.0 * stdDev);
        boolean repeatedNewPattern = currentStats.getLargestNovelTemplateCount() >= Math.max(5.0,
                0.1 * Math.max(1, currentStats.getLogCount()));
        return noveltyBurst || repeatedNewPattern && mean < 1.0;
    }

    private boolean isHigh(int index, double value, double floorIncrease) {
        double mean = baseline.mean(index);
        double stdDev = baseline.stdDev(index);
        return value >= mean + Math.max(floorIncrease, 4.0 * stdDev);
    }

    private boolean isVolumeShift(double logCount) {
        double mean = baseline.mean(LogVectorSchema.LOG_COUNT);
        if (mean < 1) {
            return false;
        }
        double stdDev = baseline.stdDev(LogVectorSchema.LOG_COUNT);
        double absoluteChange = Math.abs(logCount - mean);
        return absoluteChange >= Math.max(20.0, Math.max(4.0 * stdDev, 0.35 * mean));
    }

    public static class Builder<T extends Builder<T>> {
        private int shingleSize = 4;
        private int sampleSize = 128;
        private int numberOfTrees = 50;
        private Optional<Long> randomSeed = Optional.empty();
        private int outputAfter = 32;
        private double anomalyRate = 0.02;
        private double baselineDecay = 0.03;
        private int minimumBaselineBuckets = 80;
        private double modelAlertGrade = 0.5;
        private boolean updateBaselineOnAnomaly = false;

        public LogAnomalyDetector build() {
            validate();
            return new LogAnomalyDetector(this);
        }

        private void validate() {
            checkArgument(shingleSize > 0, "shingle size must be positive");
            checkArgument(sampleSize > 1, "sample size must be greater than one");
            checkArgument(numberOfTrees > 0, "number of trees must be positive");
            checkArgument(outputAfter > 0, "outputAfter must be positive");
            checkArgument(anomalyRate > 0 && anomalyRate < 1, "anomaly rate must be in (0, 1)");
            checkArgument(baselineDecay > 0 && baselineDecay <= 1, "baseline decay must be in (0, 1]");
            checkArgument(minimumBaselineBuckets >= 0, "minimum baseline buckets cannot be negative");
            checkArgument(modelAlertGrade >= 0 && modelAlertGrade <= 1, "model alert grade must be in [0, 1]");
        }

        public T shingleSize(int shingleSize) {
            this.shingleSize = shingleSize;
            return (T) this;
        }

        public T sampleSize(int sampleSize) {
            this.sampleSize = sampleSize;
            return (T) this;
        }

        public T numberOfTrees(int numberOfTrees) {
            this.numberOfTrees = numberOfTrees;
            return (T) this;
        }

        public T randomSeed(long randomSeed) {
            this.randomSeed = Optional.of(randomSeed);
            return (T) this;
        }

        public T outputAfter(int outputAfter) {
            this.outputAfter = outputAfter;
            return (T) this;
        }

        public T anomalyRate(double anomalyRate) {
            this.anomalyRate = anomalyRate;
            return (T) this;
        }

        public T baselineDecay(double baselineDecay) {
            this.baselineDecay = baselineDecay;
            return (T) this;
        }

        public T minimumBaselineBuckets(int minimumBaselineBuckets) {
            this.minimumBaselineBuckets = minimumBaselineBuckets;
            return (T) this;
        }

        public T modelAlertGrade(double modelAlertGrade) {
            this.modelAlertGrade = modelAlertGrade;
            return (T) this;
        }

        public T updateBaselineOnAnomaly(boolean updateBaselineOnAnomaly) {
            this.updateBaselineOnAnomaly = updateBaselineOnAnomaly;
            return (T) this;
        }
    }

    private static class AdaptiveBaseline {
        private final double decay;
        private final double[] mean = new double[LogVectorSchema.DIMENSIONS];
        private final double[] variance = new double[LogVectorSchema.DIMENSIONS];
        private int count;

        private AdaptiveBaseline(double decay) {
            this.decay = decay;
        }

        private void update(double[] vector) {
            checkArgument(vector != null && vector.length == LogVectorSchema.DIMENSIONS, "incorrect log vector length");
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

        private LogBucketSummaryStats snapshot() {
            if (count == 0) {
                return LogBucketSummaryStats.zero();
            }
            return LogBucketSummaryStats.fromVector(mean);
        }

        private double mean(int index) {
            return mean[index];
        }

        private double stdDev(int index) {
            return Math.sqrt(Math.max(variance[index], 1e-6));
        }

        private int getCount() {
            return count;
        }
    }
}
