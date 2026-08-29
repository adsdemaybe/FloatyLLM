// Sampling self-test (CPU, no GPU). Include impl to compile standalone.
#include "../src/sampling.cpp"
#include <cstdio>
#include <vector>

static int g_fail = 0;
static void check(bool c, const char* what) {
    printf(c ? "[PASS] %s\n" : "[FAIL] %s\n", what);
    if (!c) ++g_fail;
}

int main() {
    printf("[INFO] Sampling algorithm test\n");

    // Greedy picks the argmax.
    {
        std::vector<float> l = {1.0f, 3.0f, 2.0f, -5.0f};
        check(sample_greedy(l.data(), 4) == 1, "greedy picks argmax (index 1)");
    }

    // temp <= 0 falls back to greedy.
    {
        std::vector<float> l = {0.1f, 0.2f, 5.0f};
        check(sample_temperature(l.data(), 3, 0.0f, 0.9f) == 2, "temp=0 => greedy");
    }

    // Two equal logits, temp=1 -> probs 0.5/0.5. Inverse-CDF boundaries.
    {
        std::vector<float> l = {0.0f, 0.0f};
        check(sample_temperature(l.data(), 2, 1.0f, 0.4f) == 0, "equal probs, r=0.4 -> index 0");
        check(sample_temperature(l.data(), 2, 1.0f, 0.6f) == 1, "equal probs, r=0.6 -> index 1");
    }

    // Dominant logit is picked for almost all r.
    {
        std::vector<float> l = {10.0f, 0.0f, 0.0f};
        bool always0 = true;
        for (int k = 0; k < 9; ++k) {
            float r = 0.1f * (float)k;
            if (sample_temperature(l.data(), 3, 1.0f, r) != 0) always0 = false;
        }
        check(always0, "dominant logit picked across r in [0,0.8]");
    }

    // Low temperature sharpens toward greedy; high temperature spreads.
    {
        std::vector<float> l = {1.0f, 1.2f, 0.9f};
        check(sample_temperature(l.data(), 3, 0.01f, 0.99f) == 1, "low temp -> greedy-like (index 1)");
    }

    if (g_fail == 0) { printf("[INFO] ALL CHECKS PASSED\n"); return 0; }
    printf("[INFO] %d CHECK(S) FAILED\n", g_fail);
    return 1;
}
