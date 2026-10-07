# Background (optional): how your GEMM kernel runs a language model

*You don't need this to do the project. It explains what the grader does with your
kernel after the shape tests, and why those four shapes.*

## A language model is mostly matrix × vector

The grader runs **llama2.c** with the **TinyStories 15M** checkpoint, a small language
model that writes children's stories. It writes **one token (about one word-piece) at a time**.
For each token it runs the whole network once, and nearly all of that work is a sequence of
matrix-vector products `y = W · x`. `x` is the current 288-number hidden state and `W` is a
learned weight matrix.

Per token the model runs 6 layers of 7 products, plus 1 final classifier:

| Where | Matrix | Shape (`out ← in`) | Count per token |
|-------|--------|:------------------:|:-----:|
| attention (query/key/value/output) | `Wq, Wk, Wv, Wo` | 288 ← 288 | 4 × 6 layers |
| feed-forward, widen | `W1, W3` | 768 ← 288 | 2 × 6 layers |
| feed-forward, shrink | `W2` | 288 ← 768 | 1 × 6 layers |
| classifier (a score for each of 32000 vocabulary tokens) | `Wcls` | 32000 ← 288 | 1 |

That's **43 matrix-vector products per token**, and these are exactly the four `(M, K)` shapes in
the kernel contract. Everything else (normalization, softmax, attention scores) is cheap in
comparison.

## How your kernel is dropped in

llama2.c does every one of those products through a single C function:

```c
void matmul(float *xout, float *x, float *w, int n, int d);   // xout[d] = W[d×n] · x[n]
```

The grader compiles llama2.c with that function replaced by a small host routine that
calls **your** kernel on the U50:

```c
matmul(xout, x, w, n, d)   →   gemm(A = w, B = x, C = xout, M = d, K = n, N = 1)
```

- Each weight matrix `W` is copied to the card's memory **once**, the first time it is seen.
  Weights never change while the model writes.
- Per call, only `x` (up to 768 floats) goes to the card and `xout` (up to 32000 floats) comes
  back.
- The model then writes a 256-token story from the prompt *"Once upon a time"*. Each run samples
  a different story, but always exactly 256 tokens, so every run does the same work. The grader
  reports the text and the achieved **tokens/sec**.

If your kernel is correct, the story reads like a normal children's story. If it's wrong, the
model doesn't crash; it writes nonsense. A garbled story is a correctness bug.

## Why N = 1 here, and what N > 1 would buy

The model makes one token at a time, so every call is `N = 1`, a matrix-*vector* product.
That case is **memory-bound**: the kernel reads all `M × K` weights to produce only `M`
outputs, and each weight is used once. The classifier alone reads 32000 × 288 ≈ 9.2 M
weights per token. So the tok/s you get back mostly measures **how fast your kernel streams
A**.

Real inference systems raise `N` by processing a prompt's tokens together or generating
several stories at once. The same weights then serve many vectors per read, which is the
`N > 1` case the shape tests also measure. That's why the contract requires one kernel
that handles any `N`.
