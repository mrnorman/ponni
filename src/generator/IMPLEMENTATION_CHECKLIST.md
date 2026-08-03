# PONNI ahead-of-time generator checklist

- [x] Inspect `ponni::DeviceSpace`, `ponni::SArray`, CMake, and unit-test conventions.
- [x] Export fixed-shape PyTorch models through the dynamo ONNX exporter.
- [x] Validate ONNX and import it into a framework-neutral per-sample IR.
- [x] Canonicalize dense operations, constants, shapes, and tensor orientation.
- [x] Run deterministic, individually disableable optimization passes.
- [x] Schedule the graph and reuse activation storage using tensor liveness.
- [x] Generalize dense-chain scheduling across deep graphs with materialize, stream, retain, and bounded-recompute decisions.
- [x] Emit readable Kokkos C++ with batched and inline inference APIs.
- [x] Emit and benchmark Kokkos CUDA/HIP packed-half2 inference over adjacent batch samples.
- [x] Write and validate a versioned, checksummed external weight blob.
- [x] Compare PyTorch, ONNX Runtime, IR, optimized IR, and generated C++.
- [x] Cover residual DAGs, invalid models, storage reuse, and fusion rules.
- [x] Build and run generated code with the repository's GPU-debug profile.
- [x] Run the end-to-end example, benchmark, and complete documentation.

The compiler is deliberately not an ONNX runtime. Unsupported domains, operators, dynamic non-batch dimensions,
runtime shape operations, unsafe broadcasting, and training/stateful behavior are rejected before C++ is emitted.
