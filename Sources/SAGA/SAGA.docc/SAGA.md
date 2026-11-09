# ``SAGA``

Compile shaders and submit GPU work from Swift on Metal and browser WebGPU.

## Overview

SAGA provides a backend-neutral device and command API, with a separate shader
compiler and explicit GPU backends. A device owns its resources and validates
work before handing it to the backend. Await a submission ticket to observe
completion, and close the device when its owner shuts down.

Use the `SAGACompiler` module to translate supported WGSL, GLSL, and SPIR-V
programs. Choose `SAGAMetal` for native Metal or `SAGAWebGPU` for an injected
browser WebGPU host. The compiler reflects the entry point and bindings that the
runtime uses to create a pipeline.

Native compute shaders can use the `wgpu_cooperative_matrix` WGSL extension.
For Metal execution, request the `cooperativeMatrix` device feature and use
complete 32-lane subgroups. Metal supports 8×8 tiles and rectangular 8×16,
16×8 and 16×16 matrices made from those tiles, including f16 inputs with an
f32 accumulator. HLSL supports these shapes using wave operations and supports
f16 loads and stores with f32 accumulation. The compiler also emits native WGSL, GLSL with KHR
cooperative matrices, and SPIR-V for supported combinations.
Browser WebGPU compilation rejects this native extension.

## Topics

### Essentials

- <doc:RunningComputeOnMetal>

### Device API

- ``GPUDevice``
- ``GPUBufferDescriptor``
- ``GPUBackendDevice``
