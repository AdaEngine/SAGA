# SAGA

SAGA is an independent shader compiler and WebGPU-style GPU runtime written in Swift. It lets Swift applications share shader source and GPU submission code across native GPU backends and browser WebGPU.

## Supported platforms

| Platform | What runs today |
| --- | --- |
| macOS 14+ with Metal | Compute, offscreen rendering, and presentation through `CAMetalLayer` on supported devices |
| Windows and Linux with Vulkan | Native compute and rendering |
| Windows with Direct3D 12 | Native compute, single-target rendering, and window presentation |
| Web browsers with WebGPU | Swift/Wasm compute, rendering, and canvas presentation through a JavaScript WebGPU host |

The package requires Swift 6.2 or newer and has no third-party package dependencies.

## Compiler

The compiler accepts WGSL, GLSL, and SPIR-V. It emits WGSL, GLSL, SPIR-V, MSL, HLSL, and diagnostic DOT for supported combinations of features and targets.

## Vulkan benchmarks

Release builds on Windows with an NVIDIA GeForce RTX 4070 and driver 610.88. SAGA was built
from `95418a3` with the compute SPIR-V and Vulkan command-buffer reuse changes, and wgpu from
`0a8aefdad`. The table shows median and minimum wall-clock time in milliseconds, with three
warm-up runs and 15 measured samples per case.

| Case | SAGA median / min (ms) | wgpu median / min (ms) |
| --- | ---: | ---: |
| `compile`, per pipeline | 0.752 / 0.448 | 0.118 / 0.117 |
| `dispatch-batch`, 1,000 empty dispatches | 1.046 / 0.888 | 0.394 / 0.349 |
| `dispatch-submit`, per submit and wait | 0.074 / 0.065 | 0.061 / 0.049 |
| `alu` | 2.330 / 2.286 | 10.393 / 9.659 |
| `stream` | 3.600 / 3.545 | 3.493 / 3.467 |
| `roundtrip`, 16 MiB | 4.282 / 4.037 | 4.387 / 4.196 |

In a separate SAGA-only benchmark, both paths use a prebuilt Swift `CommandList`. The first
submission records a Vulkan command buffer, and the second reuses that recording after the
first completes. Each value is the median of three run medians, with 100 samples per run.

| List of 1,000 dispatches | First submission (ms) | Reused Vulkan recording (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Empty shader | 0.545 | 0.179 | 3.0× |
| Atomic increments in a storage buffer | 5.379 | 2.854 | 1.9× |

Each run verified 206,000 increments by reading the counter back from the GPU. See the
[benchmark workloads and run commands](Benchmarks/README.md) to reproduce the measurements.

## Planned and unsupported

- Complete the [pinned Naga compiler scope](docs/adr/port-full-naga-on-swift.md), including remaining language features, transforms, CLI tooling, and IR interchange.
- Expand runtime API coverage and establish WebGPU conformance with a SAGA-routed CTS suite.
- A native OpenGL runtime backend is unavailable.

[API documentation source](Sources/SAGA/SAGA.docc/SAGA.md) · [Build the documentation](docs/build-documentation.md) · [Delivery roadmap](docs/saga-delivery-slices.md)
