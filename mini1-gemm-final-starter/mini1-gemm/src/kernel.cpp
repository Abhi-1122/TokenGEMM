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
 * Two engines share one bitstream:
 *   - N == 1 (and K % 16 == 0): gemv() streams A 16 floats per cycle in blocks of GEMV_R rows.
 *   - everything else: gemm_tiled() computes TROWS x TPJ tiles of C, reusing every A and B
 *     value read across many multiply-adds.
 * Wide memory accesses index as (word * 16 + e) so HLS can prove 64-byte alignment and
 * widen the AXI ports to 512 bits. Sizes are derived with unsigned shifts to keep the
 * setup logic short (signed division by a power of two becomes a slow add/sub/select chain).
 */
#define KMAX 768
#define KWMAX (KMAX / 16)

// N = 1 engine
#define GEMV_R 32 // rows per block; also the distance between updates of one row's sum

// N > 1 engine
#define TPI 4            // tile rows computed per cycle
#define TPJ 32           // tile columns computed per cycle
#define TG 12            // row groups interleaved in time; distance between updates of one sum
#define TROWS (TPI * TG) // rows of A per tile (48; divides 288 and 768)
#define BPC 128          // columns of B per on-chip panel
#define BPW (BPC / 16)   // 16-float words per panel row

// ---------------------------------------------------------------- N = 1 (matrix-vector)

static void gemv_load_x(const float *B, float xbuf[KWMAX][16], int kwn)
{
#pragma HLS ARRAY_RESHAPE variable = xbuf type = complete dim = 2
    for (int w = 0; w < kwn; w++)
    {
#pragma HLS PIPELINE II = 1
        for (int e = 0; e < 16; e++)
        {
#pragma HLS UNROLL
            xbuf[w][e] = B[w * 16 + e];
        }
    }
}

static void gemv_load(const float *A, float abuf[GEMV_R][KWMAX][16], int b, int M, int kwn)
{
#pragma HLS ARRAY_RESHAPE variable = abuf type = complete dim = 3
    int rv = M - b * GEMV_R;
    if (rv > GEMV_R)
        rv = GEMV_R;
    int base = b * GEMV_R * kwn;
    int r = 0, w = 0;
    for (int t = 0; t < rv * kwn; t++)
    {
#pragma HLS PIPELINE II = 1
        for (int e = 0; e < 16; e++)
        {
#pragma HLS UNROLL
            abuf[r][w][e] = A[(base + t) * 16 + e];
        }
        if (++w == kwn)
        {
            w = 0;
            r++;
        }
    }
}

static void gemv_compute(float abuf[GEMV_R][KWMAX][16], float xbuf[KWMAX][16], float res[GEMV_R], int kwn)
{
#pragma HLS ARRAY_RESHAPE variable = abuf type = complete dim = 3
#pragma HLS ARRAY_RESHAPE variable = xbuf type = complete dim = 2
    float acc[GEMV_R];
    int r = 0, w = 0;
    // Order: for each 16-float chunk, for each row. A row's sum is revisited every GEMV_R
    // iterations, which is longer than the float adder's latency.
    for (int t = 0; t < kwn * GEMV_R; t++)
    {
#pragma HLS PIPELINE II = 1 style = frp
#pragma HLS DEPENDENCE variable = acc type = inter direction = RAW dependent = true distance = GEMV_R
        float p[16];
#pragma HLS ARRAY_PARTITION variable = p type = complete
        for (int e = 0; e < 16; e++)
        {
#pragma HLS UNROLL
            p[e] = abuf[r][w][e] * xbuf[w][e];
        }
        for (int h = 8; h > 0; h /= 2)
        {
#pragma HLS UNROLL
            for (int e = 0; e < h; e++)
            {
#pragma HLS UNROLL
                p[e] += p[e + h];
            }
        }
        float v = ((w == 0) ? 0.0f : acc[r]) + p[0];
        acc[r] = v;
        if (w == kwn - 1)
            res[r] = v;
        if (++r == GEMV_R)
        {
            r = 0;
            w++;
        }
    }
}

static void gemv_store(float res[GEMV_R], float *C, int b, int M)
{
    int rv = M - b * GEMV_R;
    if (rv > GEMV_R)
        rv = GEMV_R;
    for (int r = 0; r < rv; r++)
    {
#pragma HLS PIPELINE II = 1
        C[b * GEMV_R + r] = res[r];
    }
}

static void gemv_blocks(const float *A, float xbuf[KWMAX][16], float *C, int M, int kwn, int nb)
{
#pragma HLS ARRAY_RESHAPE variable = xbuf type = complete dim = 2
    // Block b+1 loads while block b computes and block b-1 stores.
    for (int b = 0; b < nb; b++)
    {
#pragma HLS DATAFLOW
        float abuf[GEMV_R][KWMAX][16];
#pragma HLS ARRAY_RESHAPE variable = abuf type = complete dim = 3
#pragma HLS BIND_STORAGE variable = abuf type = ram_s2p impl = bram latency = 2
        float res[GEMV_R];
        gemv_load(A, abuf, b, M, kwn);
        gemv_compute(abuf, xbuf, res, kwn);
        gemv_store(res, C, b, M);
    }
}

static void gemv(const float *A, const float *B, float *C, int M, int K)
{
    float xbuf[KWMAX][16];
#pragma HLS ARRAY_RESHAPE variable = xbuf type = complete dim = 2
    int kwn = (int)((unsigned)K >> 4);
    int nb = (int)(((unsigned)M + (GEMV_R - 1)) >> 5);
    gemv_load_x(B, xbuf, kwn);
    gemv_blocks(A, xbuf, C, M, kwn, nb);
}

// ---------------------------------------------------------------- N > 1 (tiled)

static void load_bpanel(const float *B, float bp[KMAX][BPW][16], int jp, int K, int N)
{
#pragma HLS ARRAY_PARTITION variable = bp type = cyclic factor = 2 dim = 2
#pragma HLS ARRAY_RESHAPE variable = bp type = complete dim = 3
    int ncols = N - jp * BPC;
    if (ncols > BPC)
        ncols = BPC;
    if ((N & 15) == 0)
    {
        int n16 = (int)((unsigned)N >> 4);
        int cw = (int)((unsigned)ncols >> 4);
        int k = 0, w = 0;
        for (int t = 0; t < K * cw; t++)
        {
#pragma HLS PIPELINE II = 1
            for (int e = 0; e < 16; e++)
            {
#pragma HLS UNROLL
                bp[k][w][e] = B[(k * n16 + jp * BPW + w) * 16 + e];
            }
            if (++w == cw)
            {
                w = 0;
                k++;
            }
        }
    }
    else
    {
        int k = 0, c = 0;
        for (int t = 0; t < K * ncols; t++)
        {
#pragma HLS PIPELINE II = 1
            bp[k][(unsigned)c / 16][(unsigned)c % 16] = B[k * N + jp * BPC + c];
            if (++c == ncols)
            {
                c = 0;
                k++;
            }
        }
    }
}

static void tile_load(const float *A, float abuf[TROWS][KWMAX][16], int it, int M, int K)
{
#pragma HLS ARRAY_PARTITION variable = abuf type = cyclic factor = TPI dim = 1
#pragma HLS ARRAY_RESHAPE variable = abuf type = complete dim = 3
    int rv = M - it * TROWS;
    if (rv > TROWS)
        rv = TROWS;
    if ((K & 15) == 0)
    {
        int kwn = (int)((unsigned)K >> 4);
        int base = it * TROWS * kwn;
        int r = 0, w = 0;
        for (int t = 0; t < rv * kwn; t++)
        {
#pragma HLS PIPELINE II = 1
            for (int e = 0; e < 16; e++)
            {
#pragma HLS UNROLL
                abuf[r][w][e] = A[(base + t) * 16 + e];
            }
            if (++w == kwn)
            {
                w = 0;
                r++;
            }
        }
    }
    else
    {
        int base = it * TROWS * K;
        int r = 0, k = 0;
        for (int t = 0; t < rv * K; t++)
        {
#pragma HLS PIPELINE II = 1
            abuf[r][(unsigned)k / 16][(unsigned)k % 16] = A[base + t];
            if (++k == K)
            {
                k = 0;
                r++;
            }
        }
    }
}

static void tile_compute(float abuf[TROWS][KWMAX][16], float bp[KMAX][BPW][16], float *C,
                         int it, int jp, int M, int K, int N)
{
#pragma HLS ARRAY_PARTITION variable = abuf type = cyclic factor = TPI dim = 1
#pragma HLS ARRAY_RESHAPE variable = abuf type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = bp type = cyclic factor = 2 dim = 2
#pragma HLS ARRAY_RESHAPE variable = bp type = complete dim = 3
    float acc[TG][TPI][TPJ];
#pragma HLS ARRAY_PARTITION variable = acc type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = acc type = complete dim = 3
    int rv = M - it * TROWS;
    if (rv > TROWS)
        rv = TROWS;
    int ncols = N - jp * BPC;
    if (ncols > BPC)
        ncols = BPC;
    int njt = (int)(((unsigned)ncols + (TPJ - 1)) / TPJ);
    for (int jt = 0; jt < njt; jt++)
    {
        // Order: for each k, for each row group g. Each cycle does TPI x TPJ multiply-adds:
        // every A value feeds TPJ columns and every B value feeds TPI rows. A given sum
        // acc[g][pi][pj] is revisited every TG iterations.
        int g = 0, k = 0, kw = 0, e = 0;
        for (int t = 0; t < K * TG; t++)
        {
#pragma HLS PIPELINE II = 1 style = frp
#pragma HLS DEPENDENCE variable = acc type = inter direction = RAW dependent = true distance = TG
            float a[TPI], b[TPJ];
#pragma HLS ARRAY_PARTITION variable = a type = complete
#pragma HLS ARRAY_PARTITION variable = b type = complete
            for (int pi = 0; pi < TPI; pi++)
            {
#pragma HLS UNROLL
                a[pi] = abuf[g * TPI + pi][kw][e];
            }
            for (int pj = 0; pj < TPJ; pj++)
            {
#pragma HLS UNROLL
                float x = bp[k][jt * 2 + pj / 16][pj % 16];
                b[pj] = (jp * BPC + jt * TPJ + pj < N) ? x : 0.0f;
            }
            for (int pi = 0; pi < TPI; pi++)
            {
#pragma HLS UNROLL
                for (int pj = 0; pj < TPJ; pj++)
                {
#pragma HLS UNROLL
                    float prev = (k == 0) ? 0.0f : acc[g][pi][pj];
                    acc[g][pi][pj] = prev + a[pi] * b[pj];
                }
            }
            if (++g == TG)
            {
                g = 0;
                k++;
                if (++e == 16)
                {
                    e = 0;
                    kw++;
                }
            }
        }

        int cols = ncols - jt * TPJ;
        if (cols > TPJ)
            cols = TPJ;
        if ((N & 15) == 0)
        {
            int n16 = (int)((unsigned)N >> 4);
            int cw = (int)((unsigned)cols >> 4);
            int r = 0, q = 0;
            if (cw == n16)
            {
                // The tile covers whole rows of C (N <= TPJ), so its rows are contiguous in
                // memory: write them as one sequential burst.
                int base = it * TROWS * n16;
                for (int t = 0; t < rv * cw; t++)
                {
#pragma HLS PIPELINE II = 1
                    for (int e2 = 0; e2 < 16; e2++)
                    {
#pragma HLS UNROLL
                        C[(base + t) * 16 + e2] = acc[(unsigned)r / TPI][(unsigned)r % TPI][q * 16 + e2];
                    }
                    if (++q == cw)
                    {
                        q = 0;
                        r++;
                    }
                }
            }
            else
            {
                for (int t = 0; t < rv * cw; t++)
                {
#pragma HLS PIPELINE II = 1
                    int row = it * TROWS + r;
                    for (int e2 = 0; e2 < 16; e2++)
                    {
#pragma HLS UNROLL
                        C[(row * n16 + jp * BPW + jt * 2 + q) * 16 + e2] =
                            acc[(unsigned)r / TPI][(unsigned)r % TPI][q * 16 + e2];
                    }
                    if (++q == cw)
                    {
                        q = 0;
                        r++;
                    }
                }
            }
        }
        else
        {
            int r = 0, c = 0;
            for (int t = 0; t < rv * cols; t++)
            {
#pragma HLS PIPELINE II = 1
                C[(it * TROWS + r) * N + jp * BPC + jt * TPJ + c] = acc[(unsigned)r / TPI][(unsigned)r % TPI][c];
                if (++c == cols)
                {
                    c = 0;
                    r++;
                }
            }
        }
    }
}

static void tile_rows(const float *A, float bp[KMAX][BPW][16], float *C, int jp, int M, int K, int N, int nt)
{
#pragma HLS ARRAY_PARTITION variable = bp type = cyclic factor = 2 dim = 2
#pragma HLS ARRAY_RESHAPE variable = bp type = complete dim = 3
    // Tile it+1 loads while tile it computes.
    for (int it = 0; it < nt; it++)
    {
#pragma HLS DATAFLOW
        float abuf[TROWS][KWMAX][16];
#pragma HLS ARRAY_PARTITION variable = abuf type = cyclic factor = TPI dim = 1
#pragma HLS ARRAY_RESHAPE variable = abuf type = complete dim = 3
#pragma HLS BIND_STORAGE variable = abuf type = ram_s2p impl = bram latency = 2
        tile_load(A, abuf, it, M, K);
        tile_compute(abuf, bp, C, it, jp, M, K, N);
    }
}

static void gemm_tiled(const float *A, const float *B, float *C, int M, int K, int N)
{
    float bp[KMAX][BPW][16];
#pragma HLS ARRAY_PARTITION variable = bp type = cyclic factor = 2 dim = 2
#pragma HLS ARRAY_RESHAPE variable = bp type = complete dim = 3
#pragma HLS BIND_STORAGE variable = bp type = ram_s2p impl = bram latency = 2
    int npan = (int)(((unsigned)N + (BPC - 1)) >> 7);
    int nt = (int)(((unsigned)M + (TROWS - 1)) / TROWS);
    for (int jp = 0; jp < npan; jp++)
    {
        load_bpanel(B, bp, jp, K, N);
        tile_rows(A, bp, C, jp, M, K, N, nt);
    }
}

extern "C" void gemm(const float *A, const float *B, float *C, int M, int K, int N)
{
    // depth= only sizes cosim's memory models and must EQUAL COSIM_{A,B,C}_ELEMS in
    // kernel_tb.cpp (cosim copies exactly `depth` elements per call). No effect on hw.
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem0 depth = 6144 max_widen_bitwidth = 512 max_read_burst_length = 64 num_read_outstanding = 32
#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmem1 depth = 11520 max_widen_bitwidth = 512
#pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmem2 depth = 256 max_widen_bitwidth = 512
#pragma HLS INTERFACE s_axilite port = M
#pragma HLS INTERFACE s_axilite port = K
#pragma HLS INTERFACE s_axilite port = N
#pragma HLS INTERFACE s_axilite port = return

    if (N == 1 && (K & 15) == 0)
        gemv(A, B, C, M, K);
    else
        gemm_tiled(A, B, C, M, K, N);
}
