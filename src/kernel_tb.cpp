/* kernel_tb.cpp — HLS testbench for src/kernel.cpp (used by csim and cosim).
 *
 * Self-contained: random A, B -> golden accumulated in double -> call gemm() ->
 * check max|err| <= 1e-2. Returns 0 on pass (HLS treats nonzero as a failed test).
 *
 *   kernel_tb full   (csim)  : the four target shapes at N = 1 and 32, plus 768x768x768.
 *   kernel_tb        (cosim) : tiny shapes only (RTL simulation is slow), still covering
 *                              both K values and N > 32.
 * hls_config.cfg passes "full" to csim (csim.argv) and nothing to cosim.
 *
 * COSIM MEMORY SIZES: cosim copies exactly `depth` elements of each m_axi port per call,
 * so in cosim mode every buffer is padded to COSIM_{A,B,C}_ELEMS and the m_axi depth=
 * values in kernel.cpp must EQUAL these. If you add bigger cosim cases, raise both.
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" void gemm(const float *A, const float *B, float *C, int M, int K, int N);

static const size_t COSIM_A_ELEMS = 6144;    /* = depth of A in kernel.cpp (8*768)  */
static const size_t COSIM_B_ELEMS = 11520;   /* = depth of B in kernel.cpp (288*40) */
static const size_t COSIM_C_ELEMS = 256;     /* = depth of C in kernel.cpp          */

static float frand(unsigned &st) {
    st = st * 1664525u + 1013904223u;
    return (float)((st >> 8) * (1.0 / 16777216.0) * 2.0 - 1.0);
}

static size_t atleast(size_t n, size_t pad) { return n > pad ? n : pad; }

static int run_case(int M, int K, int N, bool cosim) {
    size_t sA = (size_t)M * K, sB = (size_t)K * N, sC = (size_t)M * N;
    if (cosim && (sA > COSIM_A_ELEMS || sB > COSIM_B_ELEMS || sC > COSIM_C_ELEMS)) {
        std::printf("  M=%d K=%d N=%d  too big for the cosim depths — raise COSIM_*_ELEMS and depth=\n", M, K, N);
        return 1;
    }
    std::vector<float> A(cosim ? atleast(sA, COSIM_A_ELEMS) : sA, 0.0f);
    std::vector<float> B(cosim ? atleast(sB, COSIM_B_ELEMS) : sB, 0.0f);
    std::vector<float> C(cosim ? atleast(sC, COSIM_C_ELEMS) : sC, 0.0f);
    unsigned st = 1234u + (unsigned)(M * 7 + K * 13 + N * 31);
    for (size_t i = 0; i < sA; i++) A[i] = frand(st);
    for (size_t i = 0; i < sB; i++) B[i] = frand(st);

    gemm(A.data(), B.data(), C.data(), M, K, N);

    int bad = 0; double maxerr = 0.0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) acc += (double)A[(size_t)i * K + k] * (double)B[(size_t)k * N + j];
            double e = std::fabs((double)C[(size_t)i * N + j] - acc);
            if (e > maxerr) maxerr = e;
            if (!(e <= 1e-2)) bad++;
        }
    std::printf("  M=%-6d K=%-4d N=%-4d %s  max|err|=%.2e\n", M, K, N, bad ? "FAIL" : "PASS", maxerr);
    return bad ? 1 : 0;
}

int main(int argc, char **argv) {
    bool full = argc > 1 && std::strcmp(argv[1], "full") == 0;
    int fails = 0;
    if (full) {
        std::printf("=== kernel_tb (full: target shapes) ===\n");
        const int shapes[4][2] = {{288, 288}, {768, 288}, {288, 768}, {32000, 288}};
        for (auto &s : shapes)
            for (int N : {1, 32}) fails += run_case(s[0], s[1], N, false);
        fails += run_case(768, 768, 768, false);   /* the "square" score case */
    } else {
        std::printf("=== kernel_tb (tiny shapes, cosim-sized buffers) ===\n");
        fails += run_case(16, 288, 1, true);
        fails += run_case(8, 768, 1, true);
        fails += run_case(4, 288, 40, true);
    }
    std::printf("=== %s ===\n", fails ? "FAIL" : "ALL PASS");
    return fails ? 1 : 0;
}
