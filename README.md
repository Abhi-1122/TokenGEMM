# TokenGEMM

**An FP32 matrix-multiply accelerator for LLM inference on the AMD Alveo U50 FPGA.**

TokenGEMM is a single Vitis HLS kernel that runs every matrix multiply of a Llama-architecture language model (llama2.c, TinyStories-15M) on an FPGA, at **105 tokens/s**. That is **50× faster** than a naive HLS kernel on the same card. On compute-bound shapes it sustains **116 GFLOP/s**, **87% of theoretical peak**.

![FPGA](https://img.shields.io/badge/FPGA-AMD%20Alveo%20U50-ED1C24)
![HLS](https://img.shields.io/badge/Vitis%20HLS-2025.1-blue)
![Precision](https://img.shields.io/badge/precision-FP32-green)
![Memory](https://img.shields.io/badge/memory-HBM2-orange)

| | |
|---|---|
| **105.5 tok/s** | end-to-end LLM generation, all matmuls on the FPGA |
| **55 µs** | per 288 × 288 attention-projection call |
| **1,713×** | faster than naive on a 768³ GEMM (13.35 s → 7.80 ms) |
| **10.5 GB/s** | sustained weight streaming from a single HBM pseudo-channel |
| **87%** | of theoretical compute peak on dense GEMM |

---

## The problem

Generating one token of an LLM means multiplying every weight matrix by a single activation vector. For this model that's 43 matrix–vector products per token:

| Layer | Shape (M × K) | Calls per token |
|---|:-:|:-:|
| Attention Q / K / V / O | 288 × 288 | 24 |
| FFN up / gate | 768 × 288 | 12 |
| FFN down | 288 × 768 | 6 |
| Vocabulary classifier | 32000 × 288 | 1 |

The two cases stress different hardware:

- **Decode (N = 1)** reads each weight once and never reuses it. HBM bandwidth sets the speed: it is **memory bound**.
- **Batched / prefill (N > 1)** reuses each weight N times. Multiply-adds per clock set the speed: it is **compute bound**.

TokenGEMM puts both datapaths in one bitstream and picks one per call, at runtime, from the shape.

## Results

All times are medians of 5 runs on an Alveo U50. Every output stays within 5.7 × 10⁻⁵ of an exact reference.

| Workload | Naive HLS | **TokenGEMM** | Speedup |
|---|---:|---:|---:|
| Dense GEMM, 768 × 768 × 768 | 13,353 ms | **7.80 ms** (116 GFLOP/s) | **1,713×** |
| Batched, N = 32 (4 shapes, total) | 9,432 ms | **5.20 ms** | **1,815×** |
| Decode, N = 1 (4 shapes, total) | 295.3 ms | **3.75 ms** | **79×** |
| LLM generation (TinyStories-15M) | 2.12 tok/s | **105.5 tok/s** | **50×** |

### Per-shape breakdown

| Shape (M × K × N) | Time | Throughput | Bound by |
|---|---:|---:|---|
| 288 × 288 × 1 | 55 µs | 3.0 GFLOP/s | per-call overhead |
| 768 × 288 × 1 | 101 µs | 4.4 GFLOP/s | memory |
| 288 × 768 × 1 | 100 µs | 4.4 GFLOP/s | memory |
| 32000 × 288 × 1 | 3.50 ms | 5.3 GFLOP/s, **10.5 GB/s** | memory (HBM channel) |
| 288 × 288 × 32 | 86 µs | 61.9 GFLOP/s | compute |
| 768 × 288 × 32 | 153 µs | 92.4 GFLOP/s | compute |
| 288 × 768 × 32 | 178 µs | 79.7 GFLOP/s | compute |
| 32000 × 288 × 32 | 4.78 ms | 123.4 GFLOP/s | compute |
| 768 × 768 × 768 | 7.80 ms | 116.2 GFLOP/s | compute |

**Per-token cost.** One token takes about 6.6 ms of kernel time. The 32000-row vocabulary classifier accounts for 53% of it and streams 37 MB of weights per call, so the decode path targets that shape first.

### Version history

| Version | Key change | 768³ GEMM | Decode N = 1 | tok/s |
|---|---|---:|---:|---:|
| v1 | Partial accumulators, tree reduction | 417.5 ms | 3,852 ms | < 1.5 |
| v2 | Burst-friendly row buffering of A | 181.4 ms | 214.0 ms | 2.9 |
| v3 | Shift-register accumulators, row-level dataflow | 248.6 ms | 48.6 ms | 11.8 |
| v4 | Rewrite: dedicated GEMV + 128-MAC tiled engine, 512-bit AXI | 12.9 ms | 7.16 ms | 65.7 |
| **v5** | **Persistent streaming GEMV, dual engines, HBM channel mapping** | **7.80 ms** | **3.75 ms** | **105.5** |

---

## Architecture

```
Decode path (N = 1, memory bound)

  HBM --> Reader --> Packer --> Dot-product --> Writer --> HBM
          512-bit    32-row     16 MAC/cycle
          bursts     blocks

Batched path (N > 1, compute bound)

                          +--> Engine 0 (4x32 MAC/cycle) --+
  HBM --> A-tile loader --+                                +--> C-tile writer --> HBM
          (48 rows, x2)   +--> Engine 1 (4x32 MAC/cycle) --+
                                        ^
  HBM --> B-panel loader ---------------+
          (768 x 128, one copy per engine)
```

### Decode path: streaming at memory speed

The decode path keeps the HBM read port saturated.

- **Persistent dataflow.** TokenGEMM launches a reader, packer, compute unit and writer once per call, and all four run in parallel until it returns. With no per-block pipeline fill or drain, the reader issues back-to-back 512-bit bursts across the whole weight matrix.
- **`hls::stream_of_blocks` handoff.** The packer fills 32-row blocks while the compute unit drains the previous one. Load and compute overlap with no shared-buffer stalls.
- **FP-adder latency, hidden by loop order.** Compute walks each block chunk-outer, row-inner, so it returns to a row's running sum once every 32 cycles. The floating-point adder finishes well inside that window, and the loop issues a new 16-wide dot-product chunk **every cycle (II = 1)** with one accumulator per row.

### Batched path: 256 multiply-adds per cycle

- **Outer-product engines.** Each engine computes a 4 × 32 block of C per cycle: every A value feeds 32 columns and every B value feeds 4 rows. That is 128 MACs per cycle from 36 operand reads.
- **Latency-hiding by interleaving.** Twelve row groups rotate through each engine, so each accumulator comes up once every 12 cycles. The adder pipeline never stalls and the design needs no partial-sum trees.
- **Two 128-MAC engines.** A single 256-MAC engine broadcast each operand to too many multipliers and failed timing. Splitting it into two independent engines, each with a private copy of the B panel, closed routing at 260 MHz.
- **Overlapped memory traffic.** Double-buffered A tiles let the next tiles load while both engines compute and the previous results drain to HBM.

### Memory system

- **512-bit AXI on all ports.** The kernel indexes each wide access as `ptr[word * 16 + e]`. The compiler can then prove 64-byte alignment and widens all three ports to full-width bursts.
- **HBM channel isolation.** `connectivity.cfg` pins A, B and C to separate HBM pseudo-channels, so weight streams never compete with activation reads or result writes.

### Hardware footprint

Routed results, as a share of the user-available fabric:

| LUT | FF | BRAM | DSP | Clock |
|---:|---:|---:|---:|---:|
| 111k (15%) | 166k (11%) | 586 (50%) | 1,407 (24%) | 260 MHz |

---

## Engineering notes: what limits the next 2×

TokenGEMM ran into four limits:

| Bottleneck | Where TokenGEMM stands | Why |
|---|---|---|
| **Compute** | 116 of 133 GFLOP/s peak (87%) | Peak = 256 MACs × 260 MHz × 2 FLOPs. |
| **Routing** | 512 MACs/cycle failed timing (~196 MHz before adding the platform shell) | Broadcasting operands to hundreds of multipliers runs into wire delay before it runs out of DSPs. |
| **On-chip memory** | A 4-engine build needed 97% of available BRAM | Each engine needs its own B-panel copy. URAM (98% free) is the next lever. |
| **HBM bandwidth** | Decode streams at 10.5 of 14.4 GB/s (73% of one pseudo-channel) | Going past that needs weights striped across channels. |

The next step is a **systolic array**. Passing operands neighbour-to-neighbour instead of broadcasting them removes the routing wall, and moving tiles into URAM removes the memory one.

---

## Repository layout

```
src/kernel.cpp      TokenGEMM kernel (Vitis HLS C++)
src/kernel_tb.cpp   C-simulation / RTL co-simulation testbench (all 9 benchmark shapes)
connectivity.cfg    HBM pseudo-channel assignment for A, B and C
hls_config.cfg      HLS settings: part, 300 MHz target, top function
Makefile            csim · csynth · cosim · hw_emu · hw
env.sh              toolchain paths (Vitis 2025.1, XRT, U50 platform)
slurm.sh            run any make target as a Slurm job
```

## Build

Requires **Vitis 2025.1**, **XRT** and the `xilinx_u50_gen3x16_xdma_5_202210_1` platform. Point `env.sh` at your install.

```bash
make csim        # functional check on all 9 shapes           (seconds)
make csynth      # HLS synthesis + II / timing / resource report (minutes)
make cosim       # cycle-accurate RTL co-simulation              (minutes)
make hw          # full bitstream -> build/hw/gemm.xclbin        (~3.5 h, ~96 GB RAM)
```

On a Slurm cluster, `./slurm.sh hw` runs the bitstream build as a job with the right memory and time limits.

### Kernel interface

```cpp
extern "C" void gemm(const float *A, const float *B, float *C, int M, int K, int N);
// C[M×N] = A[M×K] · B[K×N], row-major FP32. Shapes are runtime arguments (K ≤ 768, any M, N ≥ 1).
```

## Acknowledgements

The end-to-end benchmark runs [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy with the TinyStories-15M checkpoint.
