/* kernel.cpp — single-precision GEMM kernel for the AMD Alveo U50 (Vitis HLS 2025.1).
 *
 * Interface (the host calls exactly this):
 *
 *   extern "C" void gemm(const float *A, const float *B, float *C, int M, int K, int N);
 *
 *   C[M*N] = A[M*K] · B[K*N]     row-major, float32, contiguous
 *   - top function name   : gemm
 *   - argument order      : A, B, C, M, K, N   (A,B,C are m_axi pointers; M,K,N scalars)
 *   - dims arrive at runtime; one bitstream must handle every shape the host runs:
 *       (M,K) in { (288,288), (768,288), (288,768), (32000,288) } at N = 1 and N = 32,
 *       plus the square case M = K = N = 768
 *     so K can be up to 768, M up to 32000, and N up to 768 (any N >= 1 must work).
 *
 * Two engines share one bitstream:
 *   - N == 1 (and K % 16 == 0): gemv() — persistent dataflow tasks. A reader streams all of A
 *     as 512-bit words, a packer fills GEMV_R-row blocks into a stream_of_blocks, the compute
 *     task consumes blocks, and a store task writes the results. No per-block restarts.
 *   - everything else: gemm_tiled() — NE copies of a TPI x TPJ multiply-add engine work on
 *     different row tiles at the same time; one task loads A tiles and one task writes C.
 * Wide memory accesses index as (word * 16 + e) so HLS can prove 64-byte alignment and
 * widen the AXI ports to 512 bits. Sizes are derived with unsigned shifts to keep the
 * setup logic short (signed division by a power of two becomes a slow add/sub/select chain).
 */
#include "hls_stream.h"
#include "hls_streamofblocks.h"

#define KMAX 768
#define KWMAX (KMAX / 16)

// N = 1 engine
#define GEMV_R 32 // rows per block; also the distance between updates of one row's sum

// N > 1 engine
#define NE 2 // engine copies (4 copies did not meet timing on the U50)
#define TPI 4            // tile rows computed per cycle per engine
#define TPJ 32           // tile columns computed per cycle per engine
#define TG 12            // row groups interleaved in time; distance between updates of one sum
#define TROWS (TPI * TG) // rows of A per tile (48; divides 288 and 768)
#define BPC 128          // columns of B per on-chip panel
#define BPW (BPC / 16)   // 16-float words per panel row

#define PRAGMA_SUB(x) _Pragma(#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)
// B panel: two word banks (each engine reads 2 words per cycle), 16 floats per word.
#define BP_SHAPE(v)                                                  \
    DO_PRAGMA(HLS ARRAY_PARTITION variable = v type = cyclic factor = 2 dim = 2) \
    DO_PRAGMA(HLS ARRAY_RESHAPE variable = v type = complete dim = 3)
// A tile: TPI row banks (each engine reads TPI rows per cycle), 16 floats per word.
#define AB_SHAPE(v)                                                  \
    DO_PRAGMA(HLS ARRAY_PARTITION variable = v type = cyclic factor = TPI dim = 1) \
    DO_PRAGMA(HLS ARRAY_RESHAPE variable = v type = complete dim = 3)
// C tile: 16 floats per word.
#define CO_SHAPE(v) DO_PRAGMA(HLS ARRAY_RESHAPE variable = v type = complete dim = 3)
#define IN_BRAM(v) DO_PRAGMA(HLS BIND_STORAGE variable = v type = ram_s2p impl = bram latency = 2)

struct w16
{
    float v[16];
};
typedef w16 gblk_t[GEMV_R][KWMAX];

// ---------------------------------------------------------------- N = 1 (matrix-vector)

static void gemv_reader(const float *A, hls::stream<w16> &out, int nwords)
{
    // One sequential pass over all of A: a single long burst stream.
    for (int t = 0; t < nwords; t++)
    {
#pragma HLS PIPELINE II = 1
        w16 x;
        for (int e = 0; e < 16; e++)
        {
#pragma HLS UNROLL
            x.v[e] = A[t * 16 + e];
        }
        out.write(x);
    }
}

static void gemv_pack(hls::stream<w16> &in, hls::stream_of_blocks<gblk_t, 2> &blocks, int M, int kwn, int nb)
{
    for (int b = 0; b < nb; b++)
    {
        hls::write_lock<gblk_t> abuf(blocks);
        int rv = M - b * GEMV_R;
        if (rv > GEMV_R)
            rv = GEMV_R;
        int r = 0, w = 0;
        for (int t = 0; t < rv * kwn; t++)
        {
#pragma HLS PIPELINE II = 1
            abuf[r][w] = in.read();
            if (++w == kwn)
            {
                w = 0;
                r++;
            }
        }
    }
}

static void gemv_compute(hls::stream_of_blocks<gblk_t, 2> &blocks, w16 xbuf[KWMAX], hls::stream<float> &res,
                         int M, int kwn, int nb)
{
    float acc[GEMV_R];
    // r and w wrap back to 0 at the end of every block, so they need no per-block reset
    // (a reset adds a select in front of the counters once HLS flattens the two loops).
    int r = 0, w = 0;
    for (int b = 0; b < nb; b++)
    {
        hls::read_lock<gblk_t> abuf(blocks);
        int rv = M - b * GEMV_R;
        if (rv > GEMV_R)
            rv = GEMV_R;
        // Order: for each 16-float chunk, for each row. A row's sum is revisited every GEMV_R
        // iterations, which is longer than the float adder's latency.
        for (int t = 0; t < kwn * GEMV_R; t++)
        {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc type = inter direction = RAW dependent = true distance = GEMV_R
            w16 a = abuf[r][w];
            w16 x = xbuf[w];
            float p[16];
#pragma HLS ARRAY_PARTITION variable = p type = complete
            for (int e = 0; e < 16; e++)
            {
#pragma HLS UNROLL
                p[e] = a.v[e] * x.v[e];
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
            if (w == kwn - 1 && r < rv)
                res.write(v);
            if (++r == GEMV_R)
            {
                r = 0;
                if (++w == kwn)
                    w = 0;
            }
        }
    }
}

static void gemv_store(hls::stream<float> &res, float *C, int M)
{
    for (int i = 0; i < M; i++)
    {
#pragma HLS PIPELINE II = 1
        C[i] = res.read();
    }
}

static void gemv_df(const float *A, w16 xbuf[KWMAX], float *C, int M, int kwn, int nb, int nwords)
{
#pragma HLS DATAFLOW
    hls::stream<w16> afifo;
#pragma HLS STREAM variable = afifo depth = 64
    hls::stream_of_blocks<gblk_t, 2> blocks;
    hls::stream<float> res;
#pragma HLS STREAM variable = res depth = 64
    gemv_reader(A, afifo, nwords);
    gemv_pack(afifo, blocks, M, kwn, nb);
    gemv_compute(blocks, xbuf, res, M, kwn, nb);
    gemv_store(res, C, M);
}

static void gemv(const float *A, const float *B, float *C, int M, int K)
{
    w16 xbuf[KWMAX];
    int kwn = (int)((unsigned)K >> 4);
    for (int w = 0; w < kwn; w++)
    {
#pragma HLS PIPELINE II = 1
        for (int e = 0; e < 16; e++)
        {
#pragma HLS UNROLL
            xbuf[w].v[e] = B[w * 16 + e];
        }
    }
    int nb = (int)(((unsigned)M + (GEMV_R - 1)) >> 5);
    gemv_df(A, xbuf, C, M, kwn, nb, M * kwn);
}

// ---------------------------------------------------------------- N > 1 (tiled, NE engines)

static void load_bpanel(const float *B, float bp0[KMAX][BPW][16], float bp1[KMAX][BPW][16], int jp, int K, int N)
{
    BP_SHAPE(bp0)
    BP_SHAPE(bp1)
    // Every engine gets its own copy of the panel so no copy is read by more than one engine.
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
                float x = B[(k * n16 + jp * BPW + w) * 16 + e];
                bp0[k][w][e] = x;
                bp1[k][w][e] = x;
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
            float x = B[k * N + jp * BPC + c];
            unsigned cw = (unsigned)c / 16, ce = (unsigned)c % 16;
            bp0[k][cw][ce] = x;
            bp1[k][cw][ce] = x;
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
    AB_SHAPE(abuf)
    int rv = M - it * TROWS;
    if (rv > TROWS)
        rv = TROWS;
    if (rv <= 0)
        return;
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

static void tile_load_round(const float *A, float ab0[TROWS][KWMAX][16], float ab1[TROWS][KWMAX][16],
                            int round, int M, int K)
{
    AB_SHAPE(ab0)
    AB_SHAPE(ab1)
    tile_load(A, ab0, round * NE + 0, M, K);
    tile_load(A, ab1, round * NE + 1, M, K);
}

static void tile_compute(float abuf[TROWS][KWMAX][16], float bp[KMAX][BPW][16], float cout[TROWS][BPW][16],
                         int round, int eng, int jp, int M, int K, int N)
{
    AB_SHAPE(abuf)
    BP_SHAPE(bp)
    CO_SHAPE(cout)
    float acc[TG][TPI][TPJ];
#pragma HLS ARRAY_PARTITION variable = acc type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = acc type = complete dim = 3
    int it = round * NE + eng;
    int rv = M - it * TROWS;
    if (rv <= 0)
        return;
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
        // Copy this column tile's sums into the C tile buffer (two 16-float words per row).
        for (int t = 0; t < TROWS * 2; t++)
        {
#pragma HLS PIPELINE II = 1
            unsigned r = (unsigned)t >> 1, q = (unsigned)t & 1;
            for (int e2 = 0; e2 < 16; e2++)
            {
#pragma HLS UNROLL
                cout[r][jt * 2 + q][e2] = acc[r / TPI][r % TPI][q * 16 + e2];
            }
        }
    }
}

static void tile_store(float *C, float cout[TROWS][BPW][16], int it, int jp, int M, int N)
{
    CO_SHAPE(cout)
    int rv = M - it * TROWS;
    if (rv > TROWS)
        rv = TROWS;
    if (rv <= 0)
        return;
    int ncols = N - jp * BPC;
    if (ncols > BPC)
        ncols = BPC;
    if ((N & 15) == 0)
    {
        int n16 = (int)((unsigned)N >> 4);
        int cw = (int)((unsigned)ncols >> 4);
        int r = 0, q = 0;
        if (cw == n16)
        {
            // The panel covers whole rows of C, so this tile's rows are contiguous in memory.
            int base = it * TROWS * n16;
            for (int t = 0; t < rv * cw; t++)
            {
#pragma HLS PIPELINE II = 1
                for (int e2 = 0; e2 < 16; e2++)
                {
#pragma HLS UNROLL
                    C[(base + t) * 16 + e2] = cout[r][q][e2];
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
                    C[(row * n16 + jp * BPW + q) * 16 + e2] = cout[r][q][e2];
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
        for (int t = 0; t < rv * ncols; t++)
        {
#pragma HLS PIPELINE II = 1
            C[(it * TROWS + r) * N + jp * BPC + c] = cout[r][(unsigned)c / 16][(unsigned)c % 16];
            if (++c == ncols)
            {
                c = 0;
                r++;
            }
        }
    }
}

static void tile_store_round(float *C, float co0[TROWS][BPW][16], float co1[TROWS][BPW][16], int round, int jp,
                             int M, int N)
{
    CO_SHAPE(co0)
    CO_SHAPE(co1)
    // The only process that writes C.
    tile_store(C, co0, round * NE + 0, jp, M, N);
    tile_store(C, co1, round * NE + 1, jp, M, N);
}

static void tile_panel(const float *A, float bp0[KMAX][BPW][16], float bp1[KMAX][BPW][16], float *C, int jp, int M,
                       int K, int N, int nrounds)
{
    BP_SHAPE(bp0)
    BP_SHAPE(bp1)
    // Round r: engine e works on row tile r*NE + e. The next round's tiles load while this
    // round computes, and this round's results are written while the next one computes.
    for (int round = 0; round < nrounds; round++)
    {
#pragma HLS DATAFLOW
        float ab0[TROWS][KWMAX][16];
        AB_SHAPE(ab0)
        IN_BRAM(ab0)
        float ab1[TROWS][KWMAX][16];
        AB_SHAPE(ab1)
        IN_BRAM(ab1)
        float co0[TROWS][BPW][16];
        CO_SHAPE(co0)
        IN_BRAM(co0)
        float co1[TROWS][BPW][16];
        CO_SHAPE(co1)
        IN_BRAM(co1)
        tile_load_round(A, ab0, ab1, round, M, K);
        tile_compute(ab0, bp0, co0, round, 0, jp, M, K, N);
        tile_compute(ab1, bp1, co1, round, 1, jp, M, K, N);
        tile_store_round(C, co0, co1, round, jp, M, N);
    }
}

static void gemm_tiled(const float *A, const float *B, float *C, int M, int K, int N)
{
    float bp0[KMAX][BPW][16];
    BP_SHAPE(bp0)
    IN_BRAM(bp0)
    float bp1[KMAX][BPW][16];
    BP_SHAPE(bp1)
    IN_BRAM(bp1)
    int npan = (int)(((unsigned)N + (BPC - 1)) >> 7);
    int nt = (int)(((unsigned)M + (TROWS - 1)) / TROWS);
    int nrounds = (nt + NE - 1) / NE;
    for (int jp = 0; jp < npan; jp++)
    {
        load_bpanel(B, bp0, bp1, jp, K, N);
        tile_panel(A, bp0, bp1, C, jp, M, K, N, nrounds);
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
