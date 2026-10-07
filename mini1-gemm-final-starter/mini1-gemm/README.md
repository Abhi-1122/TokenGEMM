# Mini-Project 1 — a GEMM kernel on the U50

You write **one FPGA kernel: a matrix multiply (GEMM)**, `C = A · B`. You build it for the
Alveo U50 card and submit it. The grader measures how fast it is and checks that it's correct.
It also uses your kernel to run a small language model that writes a short story, and reports
how many tokens per second it reaches.

## Why these matrix sizes

A language model spends nearly all its time on matrix multiplies. The grader runs
**TinyStories** (llama2.c, 15 M parameters). To write each word-piece (token), it does 43 of them:

| Name | M | K | Where in the model | Per token |
|------|:-:|:-:|---|:-:|
| attn-proj | 288 | 288 | attention: query, key, value, output | 24 |
| ffn-up | 768 | 288 | feed-forward, widen | 12 |
| ffn-down | 288 | 768 | feed-forward, shrink | 6 |
| classifier | 32000 | 288 | a score for each of the 32000 possible next tokens | 1 |

The model writes one token at a time, so it calls your kernel with **N = 1** (a matrix times one
vector). The grader also tests **N = 32** (32 vectors at once) and one **square** GEMM,
**768 × 768 × 768**. Background: [slm-background.md](slm-background.md) (optional).

## The kernel contract

The grader calls exactly this function. **Don't change its name, argument order, or types:**

```c
extern "C" void gemm(const float *A, const float *B, float *C, int M, int K, int N);
// C[M×N] = A[M×K] · B[K×N]   row-major float32. A, B, C are memory pointers; M, K, N arrive at run time.
```

One bitstream must handle all the shapes above, with any `N ≥ 1` (K goes up to 768, M up to 32000).

## How to use this repo

```
src/kernel.cpp       ← your kernel. Ships with the contract and a slow but correct placeholder
src/kernel_tb.cpp    testbench used by csim and cosim
Makefile             all the build steps (hls_config.cfg and env.sh hold its settings)
slurm.sh             submits a make step with sbatch (for the long hw build)
scripts/             hw-submit.sh (used by make submit), hw-status.sh
```

## ⚠️ Running on RCE: accounts (important)

> **Never build on the login node.** Everything runs on a compute node, under the course's
> Slurm **account `cs2401`** (`-A cs2401`).

- **Quick things** (csim, cosim, trying commands): get a shell on a compute node, then run `make csim` etc. there.
  ```bash
  sinteractive -c 8 -A cs2401
  ```
  `source env.sh` if you want to run the Vitis tools by hand.
- **Long things** (the ~1 h `hw` build): `./slurm.sh hw` submits it with `sbatch`.
  Check it with `squeue -u $USER`; the log is `build/logs/hw-<jobid>.log`.
  If `sbatch` asks for an account, run it as `SBATCH_ACCOUNT=cs2401 ./slurm.sh hw`.
- **If it doesn't start:**
  - try **with and without** `-A cs2401` (or `SBATCH_ACCOUNT=cs2401`), since the account setup may change;
  - if neither works, **ask on the course forum**.

## Steps

1. **Write your kernel** in `src/kernel.cpp`.
2. **`make csim`** (seconds): runs your C++ on all shapes. Every case must say `PASS`.
3. **`make cosim`** (minutes): synthesizes the hardware and simulates it on small inputs.
   Look for `C/RTL co-simulation finished: PASS`. Check the loop II, and that the clock meets 300 MHz, in
   `build/hls/hls/syn/report/gemm_csynth.rpt`.
4. **`./slurm.sh hw`** (about 1 hour): builds the bitstream, `build/hw/gemm.xclbin`.
5. **Submit** from the login node: `HW_TOKEN=<your-token> make submit`.
6. **See your results** in a browser on the campus network:
   - your runs: `http://10.4.25.39:18080/?token=<your-token>`
   - leaderboard: `http://10.4.25.39:18080/leaderboard`

   Reload the page for updates. From a terminal: `HW_TOKEN=<token> ./scripts/hw-status.sh`.

Other steps:

- `make csynth`: synthesis only. `make hw_emu`: emulation image (about 5 min). `make clean`: removes `build/`.
- Any step can also go to `sbatch`: `./slurm.sh <step>`.
- Run one of csim, csynth or cosim at a time, because they share `build/hls/`.

Submitting:

- Your token comes from the instructor. It is your identity, so keep it private.
- Submissions run on the card one at a time, and each takes about a minute.
- **If you submit again while your earlier one is still waiting, the new one replaces it.**

## Scoring

The grader runs 9 cases: the 4 shapes at N = 1, the 4 shapes at N = 32, and the square case.
Every output must be within `1e-2` of an exact result. **If all 9 pass**, you get three scores, in
milliseconds of kernel time (lower is better):

| Score | Adds up |
|---|---|
| **square** | the 768 × 768 × 768 case (the leaderboard ranks by this) |
| **N=1** | the 4 shapes at N = 1 |
| **N=32** | the 4 shapes at N = 32 |

Then TinyStories writes a new 256-token story with every matrix multiply on your kernel, and you get the
story and its **tokens/sec**. Every run writes the same number of tokens, so tok/s compares fairly.

Time limits: 5 minutes for the 9 cases, 3 minutes for the story. The shipped placeholder needs
about 45 s + 110 s (2 tok/s). A simple tiled kernel reaches about 7 tok/s, and a well-optimized one
can go much higher.

## Making it fast

- **N = 1** reads every element of A once and uses it once, so the speed limit is memory bandwidth.
  Read A with wide ports and long bursts, and keep many multiply-adds running at once.
- **N > 1** reuses each element of A across N columns. Buffer tiles on-chip and compute many outputs in parallel.
- The placeholder's inner loop runs at II = 8 because each add waits for the previous one. Splitting the sum
  into several partial accumulators brings it to II = 1.

## Troubleshooting

| Symptom | Fix |
|---|---|
| cosim `SIGSEGV` or memory errors | The `depth=` values in `kernel.cpp` must equal `COSIM_*_ELEMS` in `kernel_tb.cpp` |
| result says `ERROR … kernel` | The kernel's name or arguments don't match the contract |
| FAIL only for K = 768 or large N | Your buffers or tiling don't cover K = 768, or N beyond your tile size |
| all cases PASS but the story is nonsense | A bug the test shapes don't catch. Re-check csim |
