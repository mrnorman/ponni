# Generated Kokkos performance baseline

## NVIDIA Ampere baseline

Measured 2026-08-02 on `thatchroof` with `machines/thatchroof/thatchroof_gpu_fast.env`, `float`, Kokkos CUDA
`AMPERE86`, and `-O3 --use_fast_math -DNDEBUG`. Each model is a generated `I -> I -> I -> 3` MLP with tanh after
the first two dense layers. Every region was warmed and fenced. The full benchmark covers `I = 4, 8, 16, 32, 64,
128`; batches 10,000, 100,000, and 1,000,000; and hierarchical batch tiles 1, 2, 4, 8, 16, and 32.

At batch 1,000,000, the four families available at the time of this baseline are shown below; the hierarchical family is reported both at tile 1
and at its best measured tile:

```text
width  SArray ms  View batch ms  hierarchical tile 1 ms  hierarchical best ms (tile)  half2 ms
4       0.04445     0.04558          1.66739                 0.08363 (32)              0.04526
8       0.11099     0.11226          1.84401                 0.20709 (32)              0.08560
16      0.36973     0.36868          3.72621                 0.70729 (16)              0.24138
32      1.36044     1.36904         20.43290                 2.67258 (32)              0.72283
64      5.01728     5.03499         75.20280                10.00230 (32)              2.65347
128    31.64890    33.24880        287.77700                38.20510 (32)             12.86840
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

The fourth generated API, `infer_batch_half2`, uses a Kokkos `RangePolicy` over adjacent sample pairs and persistent
scalar-FP16 weights. CUDA device assembly contains 947 `HFMA2` instructions across the six generated model
instantiations, confirming that packed arithmetic was emitted rather than scalarized. It is approximately neutral at
I=4, then beats the FP32 SArray/View paths from I=8 onward; at I=128 it is 2.46x faster than SArray. Maximum absolute
difference from the FP32 View result grows from `1.19e-4`
at I=4 to `1.10e-2` at I=128 because products, weights, intermediates, and accumulation are FP16. It is therefore
explicit-only and `auto` does not select it.

Follow-up accumulation-chain measurements tested 0, 2, 4, 8, 16, and 32 partials. At I=128,
16 partials reduced the repeated-run mean time to about 0.71 of baseline and maximum absolute error from `1.10e-2`
to `4.39e-3`; 32 partials reduced error further to `3.97e-3` but increased mean time to about 1.20 of baseline.
These NVIDIA results favored multiple partials at long dot lengths, but the cross-vendor default also accounts for
the optimized MI250X results below. Users can request one global or one-per-dense measured count with
`--half2-accumulators`.

Reproduce the NVIDIA run with:

```bash
cd unit/build
source machines/thatchroof/thatchroof_gpu_fast.env
./cmakescript.sh
make -j4 generator_gpu_scale
ctest -V -R generator_gpu_scale_test
```

## NVIDIA batch-team architecture study

Measured 2026-08-04 on the same RTX 3090 fast environment with batch size 1,000,000. This study separates simple
sequential chains from short residual blocks, one long-lived skip, and four independent branches. Each row compares
`infer_batch` with the fastest of the 64/128/256/512/1024 fixed batch-team candidates:

```text
model             batch ms  best batch-team ms (team)  speedup  best scratch B/sample
SeqW16D2             0.367          0.390 (128)           0.942          64
SeqW32D2             1.372          1.364 ( 64)           1.006           0
SeqW64D2             5.168          5.152 (128)           1.003           0
SeqW128D2           33.310         50.198 (128)           0.664           0
SeqW32D8             5.774          6.034 ( 64)           0.957           0
SeqW64D8            25.700         25.328 (128)           1.015           0
ResidualW32D4        2.722          2.794 ( 64)           0.974           0
ResidualW64D4       16.022         15.432 (128)           1.038           0
ResidualW32D8        7.859          7.064 (128)           1.113           0
ResidualW64D8       43.476         30.037 (128)           1.447           0
ResidualW128D8     193.792        187.815 (128)           1.032           0
LongSkipW32D8        5.896          6.437 (128)           0.916           0
LongSkipW64D8       41.772         29.569 (128)           1.413           0
Branch4W32D2         5.487          6.436 (256)           0.853           0
Branch4W64D2        30.630         32.406 (512)           0.945           0
```

The important result is that this did not demonstrate a shared-memory benefit. The occupancy-constrained planner
placed workspace in scratch only for the width-16 sequential model; it filled 64 bytes per sample (4--32 KiB per
team), and its best candidate was 5.8% slower than direct batch. Every speedup in the table used zero scratch. Those
gains instead arise from the fixed team launch and its resulting compiler/resource behavior. Depth-eight width-64
residual and long-skip graphs are the clearest candidates on Ampere, both favoring team size 128 and gaining about
41--45%. Sequential graphs are essentially neutral through width 64 and lose substantially at width 128; four-branch
graphs also lose. Team sizes 64 or 128 are normally safest, while 256 and above are architecture-specific and often
reduce occupancy sharply.

The portable CUDA/HIP CTest prints every candidate with its local bytes, scratch placement, scratch budget, timing,
speedup, and correctness result. This is the preferred test for collecting the corresponding Frontier data:

```bash
make -j generator_batch_team_architecture_experiment
ctest -V -R '^generator_batch_team_architecture_test$'
```

## Frontier MI250X ROCm baseline

Measured 2026-08-03 on Frontier with `machines/frontier/frontier_gpu_fast.env`, `PONNI_DEBUG=OFF`, `float`, Kokkos
HIP `AMD_GFX90A`, ROCm 6.2.4, and `-O3 -ffast-math -munsafe-fp-atomics -DNDEBUG`. The formatted performance CTest
passed in 26.30 seconds. At batch 1,000,000:

```text
width  SArray ms  View batch ms  hierarchical tile 1 ms  hierarchical best ms (tile)  half2 baseline ms
4       0.04454      0.04447             3.15952                 0.13659 (32)                0.04626
8       0.11888      0.11983             3.60855                 0.19663 (32)                0.08728
16      0.31392      0.31554             5.78168                 0.45920 (32)                0.19824
32      1.03137      1.03259            14.23600                 2.00713 (32)                0.43569
64      3.92900      3.91174            42.86200                10.24620 (32)                1.68161
128    70.71230     72.00980           166.43600                44.09270 (32)                8.50097
```

The SArray and View paths remain closely matched. Packed half2 is fastest from I=8 onward and at I=128 is 8.32x
faster than SArray and 8.47x faster than View. The maximum half2 absolute difference remains within the benchmark
tolerance, increasing from `1.19e-4` at I=4 to `1.10e-2` at I=128. Disassembly of the release gfx90a executable
contains 3,014 `v_pk_fma_f16` instructions, confirming packed FP16 FMAs in the HIP code objects.

The best hierarchical tile depends on both width and batch:

```text
width  best tile at 10,000  best tile at 100,000  best tile at 1,000,000  cross-vendor default
4              32                    32                     32                    32
8              32                    32                     32                    32
16             32                    32                     32                    32
32             32                    32                     32                    32
64             16                    16                     32                    16
128             8                     8                     32                     8
```

NVIDIA favored tile 32, while MI250X favored smaller tiles for I=64 and I=128 at the two smaller batches. The
generated default therefore uses the conservative smaller cross-vendor choice: 32 through I=32, 16 at I=64, and 8
above I=64, subject to the device scratch limit. The large-batch I=128 result still benefits from an explicit tile 32.

The release half2 policy sweep produced these one-million-sample results. The measured heuristic used two partials
through I=16 and four partials from I=32 onward; the explicit policy used four throughout.

```text
width  baseline ms (error)       measured heuristic ms (error)  explicit-4 ms (error)
4       0.04626 (1.19e-4)         0.04735 (6.22e-5)              0.04968 (6.22e-5)
8       0.08728 (3.41e-4)         0.10624 (1.82e-4)              0.11301 (2.12e-4)
16      0.19824 (6.95e-4)         0.23686 (5.15e-4)              0.21162 (3.25e-4)
32      0.43569 (1.01e-3)         0.52203 (7.96e-4)              0.52190 (7.96e-4)
64      1.68161 (4.11e-3)         1.86212 (3.79e-3)              1.86217 (3.79e-3)
128     8.50097 (1.10e-2)        18.55390 (5.91e-3)             18.54720 (5.91e-3)
```

Additional partials improve error but lose performance on MI250X, most sharply at I=128. NVIDIA favored 16 partials
there, so the generated cross-vendor heuristic now conservatively retains the baseline single dependency chain.
Target-specific multi-partial policies remain available explicitly when their accuracy or performance tradeoff is
preferred.

The newer fixed batch-team family is intentionally absent from these historical tables. Its 64/128/256/512/1024
team-size candidates and generator-planned scratch placements are included in each newly generated autotuner and need
fresh measurements on both architectures before a default can be selected.

Reproduce the Frontier run with:

```bash
cd unit/build
source machines/frontier/frontier_gpu_fast.env
./cmakescript.sh
make -j4 generator_gpu_scale
ctest -V -R '^generator_gpu_scale_test$'
```
