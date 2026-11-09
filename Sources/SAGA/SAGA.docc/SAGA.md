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
complete 32-lane subgroups. Inspect `device.capabilities.cooperativeMatrixProperties`
for the supported input, accumulator and matrix dimensions. Metal supports 8×8 tiles and rectangular 8×16,
16×8 and 16×16 matrices made from those tiles, including f16 inputs with an
f32 accumulator. HLSL supports these shapes using wave operations and supports
f16 loads and stores with f32 accumulation. The compiler also emits native WGSL, GLSL with KHR
cooperative matrices, and SPIR-V for supported combinations.
Browser WebGPU compilation rejects this native extension.

Check `device.capabilities.features` for `rayQuery` before using the native Metal
ray-query path. On supported devices, `createAccelerationStructure` allocates
a bottom-level resource and `buildAccelerationStructure` builds it from a
float32 triangle vertex buffer, optionally indexed by 16- or 32-bit indices.
Out-of-range indices turn their triangle into a degenerate triangle. A top-level resource holds instances
of bottom-level structures, with an affine transform, mask and instance ID. Await
the submission before using the build result, and destroy the resources when
finished. Native Metal compute pipelines can bind a top-level structure to a WGSL
`acceleration_structure` for ray queries. This path accepts opacity, face and
geometry culling, and terminate-on-first-hit flags as literals or dynamic
values. Invalid flag combinations do not initialize the query. Vertex return is not supported. Browser WebGPU
does not provide this native API.

## Topics

### Essentials

- <doc:RunningComputeOnMetal>

### Device API

- ``GPUDevice``
- ``GPUBufferDescriptor``
- ``GPUBackendDevice``
