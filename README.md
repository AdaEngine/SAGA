# SAGA

SAGA is an independent shader compiler and WebGPU-style GPU runtime written in Swift. It lets Swift applications share shader source and GPU submission code across native Metal and browser WebGPU.

## Supported platforms

| Platform | What runs today |
| --- | --- |
| macOS 14+ with Metal | Compute, offscreen rendering, and presentation through `CAMetalLayer` on supported devices |
| Web browsers with WebGPU | Swift/Wasm compute, rendering, and canvas presentation through a JavaScript WebGPU host |

The package requires Swift 6.2 or newer and has no third-party package dependencies.

## Compiler

The compiler accepts WGSL, GLSL, and SPIR-V. It emits WGSL, GLSL, SPIR-V, MSL, HLSL, and diagnostic DOT for supported combinations of features and targets.

## Planned and unsupported

- Complete the [pinned Naga compiler scope](docs/adr/port-full-naga-on-swift.md), including remaining language features, transforms, CLI tooling, and IR interchange.
- Expand runtime API coverage and establish WebGPU conformance with a SAGA-routed CTS suite.
- Native Vulkan, Direct3D, and OpenGL runtime backends are unavailable. The verified native platform is macOS.

[API documentation source](Sources/SAGA/SAGA.docc/SAGA.md) · [Build the documentation](docs/build-documentation.md) · [Delivery roadmap](docs/saga-delivery-slices.md)
