// Host-side sampling over logits (copied from GPU). Not perf-critical, so it
// lives on the CPU and stays simple + testable. PLAN sec 6 (per-token loop).
#pragma once

// Greedy: index of the maximum logit.
int sample_greedy(const float* logits, int n);

// Temperature sampling via inverse-CDF. rand01 in [0,1). temp <= 0 => greedy.
int sample_temperature(const float* logits, int n, float temp, float rand01);
