// Host-side logit sampling.
#include "sampling.h"
#include <cmath>

int sample_greedy(const float* logits, int n) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
}

int sample_temperature(const float* logits, int n, float temp, float rand01) {
    if (temp <= 0.0f) return sample_greedy(logits, n);

    // Softmax with temperature (max-subtracted for stability).
    float m = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > m) m = logits[i];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += exp((double)(logits[i] - m) / temp);

    // Inverse CDF: smallest i whose prefix probability exceeds rand01.
    double target = (double)rand01 * sum;
    double cum = 0.0;
    for (int i = 0; i < n; ++i) {
        cum += exp((double)(logits[i] - m) / temp);
        if (cum > target) return i;
    }
    return n - 1;
}
