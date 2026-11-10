# ``SAGA``

Compile shaders and submit GPU work from Swift on Metal and browser WebGPU.

## Overview

SAGA provides a backend-neutral device and command API, with a separate shader
compiler and explicit GPU backends. A device owns its resources and validates
work before handing it to the backend. Await a submission ticket to observe
completion, and close the device when its owner shuts down.

Configure sampler addressing independently for U, V and W coordinates.
`SamplerDescriptor` also accepts `lodMinClamp` and `lodMaxClamp` to constrain
the selected mip level; their defaults are 0 and 32, respectively. The device
rejects non-finite values, negative minimums and inverted ranges.
Set `maxAnisotropy` above 1 to request anisotropic filtering; minification,
magnification and mipmap filters must all be linear in that case. The default
is 1, and the backend clamps larger values to its supported maximum.

Buffer-to-texture and texture-to-buffer copies can span multiple layers of a
2D texture array. Each layer occupies a full image stride in the buffer; the
last layer needs only the bytes covered by the copied rows. Set `rowsPerImage`
to add padding between images; for compressed formats the value counts block
rows. When omitted, images are packed using the copied height in block rows.
Use `copyTextureRegionToBuffer` to read a rectangular region rather than a
whole mip level, including selected array layers or 3D slices.
Texture-to-texture copies may cross dimensions, such as between 2D array
layers and 3D slices, when the source and destination regions are valid.
Copies between disjoint slices of the same 3D mip are supported; the browser
backend uses an intermediate texture when its WebGPU implementation rejects
copying a 3D mip to itself.

Vertex buffer layouts may contain empty slots. Bind the active slots using
`VertexBufferBinding.slot` when a binding's slot differs from its position in
the bindings array. Additional unused slots within the device limit are valid
and do not affect vertex fetch limits.

Use the `SAGACompiler` module to translate supported WGSL, GLSL, and SPIR-V
programs. Choose `SAGAMetal` for native Metal or `SAGAWebGPU` for an injected
browser WebGPU host. The compiler reflects the entry point and bindings that the
runtime uses to create a pipeline.

For native shader translation, `WGSLSourceProfile.nagaNative` enables selected
compatibility behavior of the pinned Naga frontend, including its fallback
from abstract integers to `u32` or `f32` when `i32` cannot represent a value.
Browser WebGPU compilation keeps the stricter WGSL concretization rule.

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
The compiler can also translate supported ray queries to HLSL and Vulkan GLSL;
the GLSL output requires `GL_EXT_ray_query` and primitive-culling support.
Vertex-position return in GLSL additionally requires
`GL_EXT_ray_tracing_position_fetch`.

For native mesh pipelines, `IndependentShaderCompiler.compileTaskStage` and
`compileMeshStage` translate supported WGSL entry points to MSL, HLSL and GLSL.
The HLSL profile includes subgroups, workgroup memory, texture resources and
binding arrays in both stages. The GLSL profile uses `GL_EXT_mesh_shader`, including task payloads, uniform and
storage buffers, textures, samplers, binding arrays, workgroup memory and subgroup
operations in both stages. Runtime-sized binding arrays require supplied lengths;
task, mesh and fragment stages from the same module share their buffer and
texture-unit allocation. If a module has both vertex and mesh entry points,
select `GLSLRenderPipelineFamily.vertex` or `.mesh` when compiling its fragment
stage, and use the same family for a linked vertex stage.
Additional resource shapes remain unsupported. Metal supports an
optional task stage, uniform and storage buffers
(including runtime-sized arrays), sampled textures and samplers, task/mesh
binding arrays, workgroup memory,
per-vertex and per-primitive varyings, including sample interpolation, direct and indirect draws, multiple color
targets, primitive culling, depth/stencil and multisampling with resolve. Fragment depth and sample
mask outputs, viewport and scissor, culling, timestamp and occlusion queries are
supported. Task, mesh and fragment stages can bind buffers, sampled textures,
samplers and 1D, 2D, 2D-array and 3D storage textures, including binding arrays.
Texture atomic operations require the corresponding device features. Task and mesh numeric
types and 64-bit atomics are reflected and checked against device features before pipeline
allocation. Metal also checks each stage's estimated threadgroup allocation against its
device limit; native compilation can add overhead. Subgroup operations require the device feature.
Metal task shaders bound their generated mesh grid to the device-family limits. Standalone MSL
defaults to the pinned wgpu baseline of 1024 and can override `SAGA_MESH_GRID_LIMIT` at compilation.
Direct and indirect mesh draws use the device's separate task or mesh dispatch limits,
including the total workgroup count.
Metal task payloads support validated nested structures, fixed arrays and 16-, 32- and
64-bit numeric scalars and vectors, including tightly packed 16- and 32-bit `vec3` followed by a scalar.
Task and mesh payloads are linked by byte size, as in wgpu. Metal generation preserves
supported explicit `@align` and `@size` padding; unrepresentable array strides are rejected.
Pipeline creation also checks task and mesh workgroup invocation limits, task payload size,
and mesh output vertex and primitive counts against device capabilities.
Explicit `@per_primitive` fragment inputs are translated to MSL, HLSL and
GLSL; the GLSL output requires `GL_EXT_mesh_shader`. Metal mesh pipelines check
that these inputs match primitive outputs.
Storage textures require device features appropriate to their format and access
mode. Additional resource shapes and interpolation modes remain unsupported for
this runtime profile.

## Topics

### Essentials

- <doc:RunningComputeOnMetal>

### Device API

- ``GPUDevice``
- ``GPUBufferDescriptor``
- ``GPUBackendDevice``
