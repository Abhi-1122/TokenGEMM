/* kernel.cpp — YOUR U50 GEMM kernel. This file is what you build into the .xclbin you submit.
 *
 * THE HARDWARE CONTRACT (the grader's host calls exactly this — do not change it):
 *
 *   extern "C" void gemm(const float *A, const float *B, float *C, int M, int K, int N);
 *
 *   C[M*N] = A[M*K] · B[K*N]     row-major, float32, contiguous
 *   - top function name   : gemm
 *   - argument order      : A, B, C, M, K, N   (A,B,C are m_axi pointers; M,K,N scalars)
 *   - dims arrive at runtime; one bitstream must handle every shape the grader runs:
 *       (M,K) in { (288,288), (768,288), (288,768), (32000,288) } at N = 1 and N = 32,
 *       plus the square case M = K = N = 768
 *     so K can be up to 768, M up to 32000, and N up to 768 (any N >= 1 must work).
 *
 * The body below is a deliberately NAIVE placeholder — correct, builds, and slow.
 * Replace it with your optimized kernel (tiling, on-chip buffers, wide memory ports...).
 * Keep the signature and the interface pragmas' port names.
 */
#define F 2
#define P 6
#define W 32

static void load_row(const float *A, int i, int K, float arow[768])
{
#pragma HLS ARRAY_PARTITION variable = arow type = cyclic factor = W dim = 1
    for (int c = 0; c < K; c++)
    {
#pragma HLS PIPELINE II = 1
        arow[c] = A[i * K + c];
    }
}

static void compute_row(float arow[768], float Bbuf[768][768], float *C, int i, int K, int N)
{
#pragma HLS ARRAY_PARTITION variable = arow type = cyclic factor = W dim = 1
    float acc[800];
#pragma HLS ARRAY_PARTITION variable = acc type = cyclic factor = F dim = 1

    float shift[800][P];
#pragma HLS ARRAY_PARTITION variable = shift type = cyclic factor = F dim = 1
#pragma HLS ARRAY_PARTITION variable = shift type = complete dim = 2
    for (int j = 0; j < N; j++)
    {
        for (int s = 0; s < P; s++)
            shift[j][s] = 0;
    }

    for (int k = 0; k < K; k += W)
    {
#pragma HLS PIPELINE II = 1
        for (int j = 0; j < N; j++)
        {
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL factor = F
            float prod[W];
            for (int w = 0; w < W; w++)
            {
#pragma HLS UNROLL
                prod[w] = (k + w < K) ? (arow[k + w] * Bbuf[(k + w)][j]) : 0.0f;
            }
            for (int half = W / 2; half > 0; half /= 2)
            {
#pragma HLS UNROLL
                for (int t = 0; t < half; t++)
                {
#pragma HLS UNROLL
                    prod[t] += prod[t + half];
                }
            }
            // literal-indexed shift register: every access below uses a
            // compile-time constant index, so HLS can prove the true
            // P-iteration dependency distance instead of assuming distance=1.
            float sum = shift[j][0] + prod[0];
            for (int s = 0; s < P - 1; s++)
            {
#pragma HLS UNROLL
                shift[j][s] = shift[j][s + 1];
            }
            shift[j][P - 1] = sum;
        }
    }
    for (int j = 0; j < N; j++)
#pragma HLS PIPELINE II = 1
        acc[j] = 0;

    for (int s = 0; s < P; s++)
    {
        for (int j = 0; j < N; j++)
        {
#pragma HLS PIPELINE II = 1
            acc[j] += shift[j][s];
        }
    }
    for (int j = 0; j < N; j++)
    {
        C[i * N + j] = acc[j];
    }
}

extern "C" void gemm(const float *A, const float *B, float *C, int M, int K, int N)
{
    // depth= only sizes cosim's memory models and must EQUAL COSIM_{A,B,C}_ELEMS in
    // kernel_tb.cpp (cosim copies exactly `depth` elements per call). No effect on hw.
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem0 depth = 6144 max_widen_bitwidth = 512
#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmem1 depth = 11520 max_widen_bitwidth = 512
#pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmem2 depth = 256 max_widen_bitwidth = 512
#pragma HLS INTERFACE s_axilite port = M
#pragma HLS INTERFACE s_axilite port = K
#pragma HLS INTERFACE s_axilite port = N
#pragma HLS INTERFACE s_axilite port = return

    float Bbuf[768][768];
#pragma HLS ARRAY_PARTITION variable = Bbuf type = cyclic factor = W dim = 1
#pragma HLS ARRAY_RESHAPE variable = Bbuf type = block factor = F dim = 2
    for (int y = 0; y < K; y++)
    {
        for (int z = 0; z < N; z++)
#pragma HLS PIPELINE II = 1
            Bbuf[y][z] = B[y * N + z];
    }

    // DATAFLOW: row i+1's arow load overlaps row i's compute instead of
    // running fully before it. load_row produces arow, compute_row consumes
    // it — Vitis infers a ping-pong (double) buffer for arow across iterations.
    for (int i = 0; i < M; i++)
    {
#pragma HLS DATAFLOW
        float arow[768];
#pragma HLS ARRAY_PARTITION variable = arow type = cyclic factor = W dim = 1
        load_row(A, i, K, arow);
        compute_row(arow, Bbuf, C, i, K, N);
    }
}
