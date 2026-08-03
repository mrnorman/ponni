# Generated Kokkos performance baseline

Measured 2026-08-02 on `thatchroof` with `machines/thatchroof/thatchroof_gpu_fast.env`, `float`, Kokkos CUDA
`AMPERE86`, and `-O3 --use_fast_math -DNDEBUG`. Each model is a generated `I -> I -> I -> 3` MLP with tanh after
the first two dense layers. Every region was warmed and fenced. The full benchmark covers `I = 4, 8, 16, 32, 64,
128`; batches 10,000, 100,000, and 1,000,000; and hierarchical batch tiles 1, 2, 4, 8, 16, and 32.

At batch 1,000,000, the five generated families are shown below; the hierarchical family is reported both at tile 1
and at its best measured tile:

```text
width  SArray ms  View batch ms  hierarchical tile 1 ms  hierarchical best ms (tile)  raw CUDA WMMA best ms (warps)  half2 ms
4       0.04445     0.04558          1.66739                 0.08363 (32)                 0.14275 (4)              0.04526
8       0.11099     0.11226          1.84401                 0.20709 (32)                 0.15421 (4)              0.08560
16      0.36973     0.36868          3.72621                 0.70729 (16)                 0.30558 (2)              0.24138
32      1.36044     1.36904         20.43290                 2.67258 (32)                 1.06182 (4)              0.72283
64      5.01728     5.03499         75.20280                10.00230 (32)                 3.06994 (2)              2.65347
128    31.64890    33.24880        287.77700                38.20510 (32)                16.41240 (1)             12.86840
```

Tile 32 is the robust hierarchical choice. Tiling substantially reduces team-per-sample overhead, but both
sample-local paths remain faster than hierarchical at every tested width. A legal three-dense streaming tail
therefore selects sample-local; the 64-neuron threshold applies only to non-streamable graphs.

Both sample-local forms now load each input exactly once into fixed local storage and materialize only the first
I-neuron activation; the second I-neuron layer is produced one scalar at a time and accumulated directly into three
scalar outputs. This removes the previous View rereads and reduces activation storage from `2I` to `I`. The paths are
within 0.2% through I=64 and within 5.4% at I=128. At I=128 the optimized kernels use essentially identical resources:
SArray uses 167 registers and 512 stack bytes per thread, while View-batch uses 168 registers and 512 stack bytes.

All comparisons had zero maximum absolute difference between the portable paths. `Kokkos::RandomAccess` remains disabled:
these small, regular dense traversals provide no measured reason to add it.

The fourth generated API, `infer_batch_tensorcore`, launches a raw CUDA WMMA TF32 kernel with a fixed 16-sample batch
tile and no Kokkos `TeamPolicy`. It accumulates in FP32, caches the input and first hidden activation in aligned
dynamic shared memory, and streams second-layer tiles into the final output fragment. Shared memory ranges from 2,560
bytes per warp at I=4/8 to 17,920 bytes at I=128. Sweeping every legal choice among 1, 2, 4, and 8 warps per block
selected 4, 2, 2, 4, 2, and 1 warps for I=4 through 128 at one million samples. The emitted defaults use those
large-batch choices. Tensor Core becomes fastest at I=16 in this run and reaches a 1.94x speedup
over SArray at I=128. Maximum absolute differences ranged from `1.79e-5` to `1.63e-3`; portable paths matched exactly.
For these reasons, the Tensor Core target is explicit-only and `auto` never selects its approximate TF32 semantics.

The fifth generated API, `infer_batch_half2`, uses a Kokkos `RangePolicy` over adjacent sample pairs and persistent
scalar-FP16 weights. CUDA device assembly contains 947 `HFMA2` instructions across the six generated model
instantiations, confirming that packed arithmetic was emitted rather than scalarized. It is approximately neutral at
I=4, then beats the FP32 SArray/View paths from I=8 onward; at I=128 it is 2.46x faster than SArray and 1.28x faster
than the WMMA target in this benchmark. Maximum absolute difference from the FP32 View result grows from `1.19e-4`
at I=4 to `1.10e-2` at I=128 because products, weights, intermediates, and accumulation are FP16. It is therefore
explicit-only and `auto` does not select it.

Follow-up accumulation-chain measurements tested 0, 2, 4, 8, 16, and 32 partials. The generated default
`infer_batch_half2_heuristic` now selects 2 partials for dot lengths through 24, 4 through 80, and 16 above 80;
lengths below 2 retain the baseline chain. The count is selected independently for every dense node. At I=128,
16 partials reduced the repeated-run mean time to about 0.71 of baseline and maximum absolute error from `1.10e-2`
to `4.39e-3`; 32 partials reduced error further to `3.97e-3` but increased mean time to about 1.20 of baseline.
For a streamed consumer, the count is conservatively reduced until `output_size * accumulators <= 48`, the largest
live-output-partial point covered by these measurements. Users can request one global or one-per-dense measured count
with `--half2-accumulators`.

Reproduce with:

```bash
cd unit/build
source machines/thatchroof/thatchroof_gpu_fast.env
./cmakescript.sh
make -j generator_gpu_scale
ctest -V -R generator_gpu_scale_test
```
