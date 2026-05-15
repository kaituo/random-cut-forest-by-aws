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

import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Converts raw log lines into stable template identities. The parser
 * intentionally stays dependency-free for parkservices and handles the common
 * JSON and key=value shapes before falling back to free-form canonicalization.
 */
public class LogRecordNormalizer {

    private static final Pattern JSON_FIELD = Pattern.compile(
            "\\\"([^\\\"\\\\]*(?:\\\\.[^\\\"\\\\]*)*)\\\"\\s*:\\s*(\\\"([^\\\"\\\\]*(?:\\\\.[^\\\"\\\\]*)*)\\\"|-?\\d+(?:\\.\\d+)?|true|false|null)");
    private static final Pattern KEY_VALUE_FIELD = Pattern
            .compile("([A-Za-z0-9_@.-]+)=(\\\"[^\\\"]*\\\"|'[^']*'|\\S+)");
    private static final Pattern UUID = Pattern
            .compile("\\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\\b");
    private static final Pattern IP = Pattern.compile("\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b");
    private static final Pattern QUOTED_STRING = Pattern.compile("\\\"[^\\\"]*\\\"|'[^']*'");
    private static final Pattern HEX = Pattern.compile("\\b(?:0x)?[0-9a-fA-F]*[a-fA-F][0-9a-fA-F]{7,}\\b");
    private static final Pattern LONG_NUMBER = Pattern.compile("\\b\\d{4,}\\b");
    private static final Pattern WHITESPACE = Pattern.compile("\\s+");
    private static final Pattern FULL_EXCEPTION = Pattern
            .compile("\\b(?:[A-Za-z_$][\\w$]*\\.)+[A-Za-z_$][\\w$]*(?:Exception|Error)\\b");
    private static final Pattern SIMPLE_EXCEPTION = Pattern.compile("\\b[A-Za-z_$][\\w$]*(?:Exception|Error)\\b");

    private static final Set<String> MESSAGE_FIELDS = new HashSet<>();
    private static final Set<String> LEVEL_FIELDS = new HashSet<>();
    private static final Set<String> VARIABLE_FIELDS = new HashSet<>();

    static {
        Collections.addAll(MESSAGE_FIELDS, "message", "msg", "log", "event.message", "error.message",
                "exception.message");
        Collections.addAll(LEVEL_FIELDS, "level", "severity", "log.level", "status", "priority");
        Collections.addAll(VARIABLE_FIELDS, "id", "requestid", "request_id", "traceid", "trace_id", "spanid", "span_id",
                "userid", "user_id", "orderid", "order_id", "sessionid", "session_id", "sku", "request", "latency",
                "latencyms", "latency_ms", "duration", "durationms", "duration_ms", "timestamp", "@timestamp", "time",
                "ts", "date", "host", "hostname", "ip", "clientip", "client_ip");
    }

    public NormalizedLogRecord normalize(String rawMessage) {
        String raw = rawMessage == null ? "" : rawMessage.trim();
        Map<String, String> fields = parseJsonFields(raw);
        if (fields.isEmpty()) {
            fields = parseKeyValueFields(raw);
        }

        String templateSource = chooseTemplateSource(raw, fields);
        String canonicalTemplate = canonicalize(templateSource);
        String level = normalizeLevel(fields, raw);
        String exceptionName = extractException(raw);
        boolean error = isError(level, canonicalTemplate, exceptionName);
        boolean warn = isWarn(level, canonicalTemplate) && !error;

        return new NormalizedLogRecord(raw, canonicalTemplate, stableTemplateId(canonicalTemplate), level, error, warn,
                exceptionName);
    }

    public String canonicalize(String value) {
        String answer = value == null ? "" : value;
        answer = UUID.matcher(answer).replaceAll("<uuid>");
        answer = IP.matcher(answer).replaceAll("<ip>");
        answer = QUOTED_STRING.matcher(answer).replaceAll("<str>");
        answer = HEX.matcher(answer).replaceAll("<hex>");
        answer = LONG_NUMBER.matcher(answer).replaceAll("<num>");
        answer = answer.toLowerCase(Locale.ROOT);
        answer = WHITESPACE.matcher(answer).replaceAll(" ").trim();
        return answer;
    }

    private Map<String, String> parseJsonFields(String raw) {
        if (!looksLikeJsonObject(raw)) {
            return Collections.emptyMap();
        }
        Matcher matcher = JSON_FIELD.matcher(raw);
        Map<String, String> fields = new LinkedHashMap<>();
        while (matcher.find()) {
            String key = normalizeKey(unescapeJson(matcher.group(1)));
            String rawValue = matcher.group(2);
            String value = rawValue;
            if (rawValue.length() >= 2 && rawValue.charAt(0) == '"' && rawValue.charAt(rawValue.length() - 1) == '"') {
                value = unescapeJson(rawValue.substring(1, rawValue.length() - 1));
            }
            fields.put(key, value);
        }
        return fields;
    }

    private boolean looksLikeJsonObject(String raw) {
        return raw.length() >= 2 && raw.charAt(0) == '{' && raw.charAt(raw.length() - 1) == '}';
    }

    private Map<String, String> parseKeyValueFields(String raw) {
        Matcher matcher = KEY_VALUE_FIELD.matcher(raw);
        Map<String, String> fields = new LinkedHashMap<>();
        int firstStart = -1;
        while (matcher.find()) {
            if (firstStart < 0) {
                firstStart = matcher.start();
            }
            String key = normalizeKey(matcher.group(1));
            String value = stripQuotes(matcher.group(2));
            fields.put(key, value);
        }
        if (firstStart > 0 && raw.substring(0, firstStart).trim().length() > 0) {
            return Collections.emptyMap();
        }
        return fields.size() >= 2 ? fields : Collections.emptyMap();
    }

    private String chooseTemplateSource(String raw, Map<String, String> fields) {
        if (fields.isEmpty()) {
            return raw;
        }
        for (String field : MESSAGE_FIELDS) {
            String value = fields.get(field);
            if (value != null && !value.isEmpty()) {
                return value;
            }
        }

        StringBuilder builder = new StringBuilder();
        fields.entrySet().stream().filter(entry -> !VARIABLE_FIELDS.contains(entry.getKey()))
                .sorted(Map.Entry.comparingByKey()).forEach(entry -> {
                    if (builder.length() > 0) {
                        builder.append(' ');
                    }
                    builder.append(entry.getKey()).append('=').append(entry.getValue());
                });
        return builder.length() == 0 ? raw : builder.toString();
    }

    private String normalizeLevel(Map<String, String> fields, String raw) {
        for (String field : LEVEL_FIELDS) {
            String level = fields.get(field);
            if (level != null) {
                return canonicalize(level);
            }
        }
        String lower = raw.toLowerCase(Locale.ROOT);
        if (lower.startsWith("error ") || lower.startsWith("fatal ") || lower.startsWith("critical ")
                || lower.startsWith("severe ")) {
            return lower.substring(0, lower.indexOf(' '));
        }
        if (lower.startsWith("warn ") || lower.startsWith("warning ")) {
            return lower.substring(0, lower.indexOf(' '));
        }
        if (lower.startsWith("info ") || lower.startsWith("debug ") || lower.startsWith("trace ")) {
            return lower.substring(0, lower.indexOf(' '));
        }
        return "";
    }

    private boolean isError(String level, String canonicalTemplate, String exceptionName) {
        if (level.equals("error") || level.equals("fatal") || level.equals("critical") || level.equals("severe")) {
            return true;
        }
        if (exceptionName != null) {
            return true;
        }
        return containsWord(canonicalTemplate, "error") || containsWord(canonicalTemplate, "failed")
                || containsWord(canonicalTemplate, "failure") || containsWord(canonicalTemplate, "fatal")
                || containsWord(canonicalTemplate, "critical") || containsWord(canonicalTemplate, "exception")
                || containsWord(canonicalTemplate, "panic");
    }

    private boolean isWarn(String level, String canonicalTemplate) {
        if (level.equals("warn") || level.equals("warning")) {
            return true;
        }
        return containsWord(canonicalTemplate, "warn") || containsWord(canonicalTemplate, "warning")
                || containsWord(canonicalTemplate, "retry") || containsWord(canonicalTemplate, "retryable")
                || containsWord(canonicalTemplate, "degraded") || containsWord(canonicalTemplate, "throttle")
                || containsWord(canonicalTemplate, "timeout");
    }

    private boolean containsWord(String text, String word) {
        return Pattern.compile("(^|[^a-z0-9_])" + Pattern.quote(word) + "([^a-z0-9_]|$)").matcher(text).find();
    }

    private String extractException(String raw) {
        Matcher matcher = FULL_EXCEPTION.matcher(checkNotNull(raw, "raw must not be null"));
        if (matcher.find()) {
            return matcher.group().toLowerCase(Locale.ROOT);
        }
        matcher = SIMPLE_EXCEPTION.matcher(raw);
        if (matcher.find()) {
            return matcher.group().toLowerCase(Locale.ROOT);
        }
        return null;
    }

    private String stableTemplateId(String canonicalTemplate) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = digest.digest(canonicalTemplate.getBytes(StandardCharsets.UTF_8));
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < 8; i++) {
                builder.append(String.format(Locale.ROOT, "%02x", bytes[i] & 0xff));
            }
            return builder.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is required for stable log template ids", e);
        }
    }

    private String stripQuotes(String value) {
        if (value.length() >= 2) {
            char first = value.charAt(0);
            char last = value.charAt(value.length() - 1);
            if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
                return value.substring(1, value.length() - 1);
            }
        }
        return value;
    }

    private String normalizeKey(String key) {
        return key == null ? "" : key.trim().toLowerCase(Locale.ROOT).replace("-", "_");
    }

    private String unescapeJson(String value) {
        return value.replace("\\\"", "\"").replace("\\\\", "\\");
    }

    public static final class NormalizedLogRecord {
        private final String rawMessage;
        private final String canonicalTemplate;
        private final String templateId;
        private final String level;
        private final boolean error;
        private final boolean warn;
        private final String exceptionName;

        private NormalizedLogRecord(String rawMessage, String canonicalTemplate, String templateId, String level,
                boolean error, boolean warn, String exceptionName) {
            this.rawMessage = rawMessage;
            this.canonicalTemplate = canonicalTemplate;
            this.templateId = templateId;
            this.level = level;
            this.error = error;
            this.warn = warn;
            this.exceptionName = exceptionName;
        }

        public String getRawMessage() {
            return rawMessage;
        }

        public String getCanonicalTemplate() {
            return canonicalTemplate;
        }

        public String getTemplateId() {
            return templateId;
        }

        public String getLevel() {
            return level;
        }

        public boolean isError() {
            return error;
        }

        public boolean isWarn() {
            return warn;
        }

        public String getExceptionName() {
            return exceptionName;
        }
    }
}
