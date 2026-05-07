# ICNN Research

Benchmarking input-convex neural networks (ICNNs) against statistical baselines. Planned extensions: FALCON, COMONet, DCiNN.

## Structure

```
src/
  models/       — model definitions (FICNN + stubs for FALCON, COMONet, DCiNN)
  training/     — training loop
  data/         — dataset generators
benchmarks/     — comparison harness and baseline stubs
notebooks/      — exploratory experiments
```

## Models

| Model | Status | Description |
|-------|--------|-------------|
| FICNN | ✓ | Fully Input-Convex NN (Amos et al., 2017) |
| FALCON | stub | — |
| COMONet | stub | — |
| DCiNN | stub | — |

## FICNN Weight Constraints

Three strategies enforce W_z ≥ 0 (required for convexity):

- **clip** — project onto non-negative orthant after each gradient step
- **exp** — reparametrize W_z = exp(W̃_z); regularize W̃_z to prevent divergence
- **softplus** — reparametrize W_z = softplus(W̃_z); `FICNNSoftplusShifted` initializes W̃_z at −2 for better expressivity early in training
