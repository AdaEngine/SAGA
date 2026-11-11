# Add Vulkan (Windows, Linux) and Direct3D 12 (Windows) native backends

- Status: accepted by explicit user request ("support Windows, Web, Linux and Apple"), 2026-11-10
- Amends: [Metal and browser WebGPU](use-metal-and-browser-webgpu.md) and the last paragraph of
  [full Naga on Swift](port-full-naga-on-swift.md), which said SPIR-V/HLSL/GLSL output does not add
  Vulkan or Direct3D runtime backends. That limit is lifted.

## Context

The compiler already emits MSL, HLSL, GLSL, SPIR-V and WGSL, and the runtime has two execution
paths: Metal (Apple) and an injected browser WebGPU host (Web). Windows and Linux have no native
GPU path. The package builds on Windows with Swift 6.3, but `SAGAMetal` compiles to stubs there.

## Decision

| Platform | Runtime backend | Shader artifact | Status |
|---|---|---|---|
| macOS, iOS | Metal (`SAGAMetal`) | MSL | existing |
| Web | Browser WebGPU host (`SAGAWebGPU`) | WGSL | existing |
| Windows, Linux (and Android later) | Vulkan 1.1+ (`SAGAVulkan`) | SPIR-V 1.3 | new |
| Windows | Direct3D 12 (`SAGAD3D12`) | HLSL / DXIL | planned, after Vulkan reaches compute+render parity |

Vulkan comes first because one backend serves both new operating systems and the SPIR-V writer is
already verified with SPIRV-Tools. Direct3D 12 follows as the Windows-native alternative and reuses
the same `SAGABackend` protocols.

Constraints that carry over:

- No SwiftPM dependency, no Vulkan headers and no SDK needed to build. `SAGAVulkan` declares the
  small part of the Vulkan C ABI it uses in Swift and resolves it at run time from the system
  loader (`vulkan-1.dll`, `libvulkan.so.1`). A machine without a driver gets
  `GPUError.adapterUnavailable`, never a CPU fallback.
- Platform objects stay inside the backend. Ownership, validation, error scopes and queue ordering
  stay in `SAGACore`; the backend revalidates the trusted command SPI like Metal does.
- A backend advertises only features it has verified on hardware. Unsupported commands and
  pipeline layouts fail with `GPUError.unsupported`.
- Evidence stays separate per backend: Vulkan results do not count toward Metal or browser claims.
  Linux is verified only on a Linux host; Windows results here do not prove Linux.

## Slice order

1. Vulkan loader, adapter, device, buffers, queue, SPIR-V compute pipeline, direct dispatch, copy,
   clear and readback (this change).
2. Vulkan indirect dispatch, immediates, binding ranges, timestamp queries, buffer mapping.
3. Vulkan textures, samplers, copies, storage textures.
4. Vulkan render passes, vertex/index/indirect draws, depth/stencil, MSAA.
5. Vulkan presentation (Win32 and Xlib/Wayland surfaces, swapchain).
6. Direct3D 12 following the same order.
7. Feature parity items from the delivery slices that Metal has: atomics tiers, subgroups,
   cooperative matrices, mesh shaders, ray queries, external textures.

## Status of slices (2026-11-10, verified on a Windows Vulkan 1.4 GPU)

| Slice | State | Evidence |
|---|---|---|
| 1. Compute, buffers, readback | done | `swift test --filter vulkan` |
| 2. Uniform buffers, immediates (push constants), indirect dispatch with GPU-side limit validation | done | same |
| 3z. Linux host: build and full suite on Ubuntu 24.04 (WSL2) with lavapipe; Xlib presentation with swapchain rebuild and Wayland presentation to a bare wl_surface on the WSLg compositor (both through a run-time loaded client library); portability fixes (libm link, expressions the Swift 6.2.3 type checker rejects). The Vulkan tests also ran on the RTX 4070 through a locally built Mesa Dozen (Vulkan over Direct3D 12 in WSL2): every test passed or skipped for missing features, and the experimental driver then segfaults on process exit. Dozen cannot render into 3D texture slices, so that test skips there. Running the suite on Dozen found two real defects, fixed here: Vulkan adapter name and type were read from the wrong bytes of the properties, and query results were copied without the wait flag | done on lavapipe and Dozen | Linux run of the suite, Xlib and Wayland presentation tests |
| 7a. wgpu feature parity, small set: clamp-to-border samplers (SamplerAddressMode.clampToBorder with transparent black, opaque black or opaque white), polygon modes (line on Vulkan and Direct3D 12, point on Vulkan only, as in wgpu), and multi-draw-indirect with a fixed count or a GPU count word (`renderMultiIndirect`; Vulkan validates every argument record in a compute pass, Direct3D 12 uses ExecuteIndirect). Each is feature-gated and verified on the RTX 4070 with the validation and debug layers clean; Metal and the browser host do not advertise them, and the Metal edits are unbuilt | done on Vulkan and Direct3D 12 | border, polygon-mode and multi-draw tests |
| 3a. 2D textures, buffer/texture copies and uploads, offscreen render (vertex buffers, instances, uniforms and storage buffers, blending, viewport, scissor) | done | triangle and vertex-buffer pixel readback tests |
| 3b. Sampled 2D textures, samplers, compute storage textures | done | compute sample-and-store and fragment sampling tests |
| 4a. Indexed draws (uint16, uint32, signed base vertex), robustBufferAccess enabled | done | indexed draw test |
| 4b. Depth-only attachments (depth16Unorm, depth32Float): compare, write mask, bias, load/clear/store, readback, sampling | done | depth test |
| 4c. 4x MSAA color targets with resolve (store or discard), alpha-to-coverage, sample mask | done | resolve test |
| 4d. Indirect draws (indexed and non-indexed, GPU-validated arguments) | done | indirect draw test |
| 4e. Stencil and combined depth-stencil formats (device-probed D24S8/D32S8/S8), stencil ops, reference, aspect copies | done | stencil test |
| 4f. Texture views (mip range, aspect) for sampling and attachments | done | mip-view test |
| 4g. 2D array and 3D textures, cube-compatible images, layer-range views, array/cube/3D sampling bindings, layer and slice copies | done | array and volume test |
| 4h. Block-compressed textures (BC, ETC2, ASTC LDR; probed), MSAA with depth, 1D textures | done | BC, MSAA-depth and 1D tests |
| 4i. Render to 3D slices, resolve-less MSAA (works on the tested driver; render pass compatibility not checked by validation layers) | done | slice and MSAA-keep tests |
| 4j. ASTC HDR (VK_EXT_texture_compression_astc_hdr: feature enabled and SFLOAT block formats mapped when the device offers it). Unit tests pin the 14 format values, the structure type, the extension name and the feature chain linkage to vulkan_core.h; storing real textures has not run, because no reachable device (RTX 4070, AMD iGPU, lavapipe, Dozen) reports the feature and the Khronos profiles layer cannot add features a device lacks | partial (plumbing verified, storage unverified) | mapping, chain and enablement tests |
| 5a. Buffer mapping (read, write, mapped at creation), timestamp and occlusion queries with resolve | done | mapping and query tests |
| 5b. Presentation: Win32 swapchain (verified, including resize rebuild); Xlib and Wayland surfaces implemented but not run | partial | Win32 present test |
| 5c. Optional device features probed and enabled (f16 with 16-bit storage, int16, int64, clip distance, anisotropy, fragment stores); f16 verified | done | f16 compute test |
| 5d. Subgroups (basic, vote, arithmetic, ballot, shuffle, quad; probed) | done | subgroup add test |
| 5e. Binding arrays (buffers, textures, samplers; static and dynamic indexing features enabled when present) | done | storage buffer array test |
| 5f. Multiview (render pass view masks, array attachment views, view-index shaders) | done | multiview test |
| 5g. Cooperative matrices (VK_KHR_cooperative_matrix; subgroup-scope f16 and f32 shapes read from the driver) | done | cooperative matrix multiply test |
| 5h. Mesh shaders (VK_EXT_mesh_shader): SPIR-V mesh stage (triangles, position and location outputs, no task stage), reflected SPIR-V mesh artifacts, mesh graphics pipelines and drawMesh; verified with validation layers | partial | mesh draw test |
| 5i. Line and point mesh outputs (verified); mesh vertex positions now negate Y like vertex stages, fixing a vertical flip found by the point test | done | line and point mesh tests |
| 5j. Indirect mesh draws with GPU-side group-count validation (verified); whole Vulkan suite passes with the validation layer clean after fixing buffer usage flags and a per-submission helper pipeline leak | done | indirect mesh test |
| 5k. Task stages (TaskEXT with a `TaskPayloadWorkgroupEXT` payload, dispatch size clamped to the portable limits, merged task bindings, task-aware indirect limits); verified with validation layers | done | task payload and oversized dispatch tests |
| 5l. External textures (planes as three sampled images and a uniform block of parameter words at binding numbers derived from the WGSL binding; sample, load and dimensions expanded inline in SPIR-V); verified in compute and render with validation layers | done | external texture tests |
| 5m. Ray queries in compute (VK_KHR_acceleration_structure and ray_query: bottom-level builds from triangle geometry with GPU-sanitized indices, top-level builds from instances, acceleration structure descriptors, SPIR-V ray query lowering that mirrors the GLSL writer's state handling); results match the Metal tests with validation layers clean. Vertex positions of hits work through VK_KHR_ray_tracing_position_fetch (verified); Ray queries also run in fragment and vertex stages (acceleration structures are bound through the render command texture map, verified in a fragment shader on Vulkan and D3D12; Metal render pipelines report them unsupported). Procedural box geometry builds through a new `buildBoxAccelerationStructure` command and works with candidate hits confirmed by `rayQueryGenerateIntersection` (verified on Vulkan and D3D12; Metal and the browser host report it unsupported) | partial | ray query tests |
| 6k. Direct3D 12 planar external textures in compute and render pipelines (three plane SRVs plus a root CBV of parameter words; HLSL helpers made DXC-compatible) | done | `d3d12` external texture test |
| 6l. Direct3D 12 mesh and task shaders (HLSL ms_6_5/as_6_5, ID3D12Device2 pipeline-state streams, DispatchMesh; payload copied member by member because DXC crashes on struct copies); GPU-authored mesh group counts validated into a scratch buffer, then ExecuteIndirect | done | mesh, task+mesh and indirect mesh tests |
| 6a. Direct3D 12 compute: HLSL through DXC (dxcompiler.dll, dxil.dll found at runtime), root-descriptor buffer bindings (CBV, SRV, UAV), copies, fills, direct dispatch, readback | done | `d3d12` compute test |
| 6b. Direct3D 12 immediates and workgroup counts as root constants, indirect dispatch with GPU-side limit validation (ExecuteIndirect) | done | immediates and indirect test |
| 6c. Direct3D 12 2D/array/3D color textures, buffer-texture copies and uploads, samplers, sampled and storage textures in compute (descriptor tables; debug layer clean) | done | texture compute test |
| 6d. Direct3D 12 render pipelines (one color target, blend, cull, topology, vertex and index buffers, instances, buffers/textures/samplers/immediates); fragment inputs now list system values first so Direct3D stage signatures link | done | render test |
| 6e. Direct3D 12 depth and stencil (D16, D32, D24S8, D32S8), 4x MSAA colour and depth with resolve; vertex-index draw bases fall back to root constants when the DXIL validator lacks shader model 6.8 | done | depth and MSAA test |
| 6f. Direct3D 12 buffer mapping (snapshot reads, batched writes), timestamp and occlusion queries with resolve | done | mapping and query tests |
| 6g. Direct3D 12 indirect draws (ExecuteIndirect, indexed and not) and window presentation through a DXGI flip swap chain that follows the presented texture size | done | indirect draw and present tests |
| 6h. Direct3D 12 texture views (mip range, array layers, dimension) for sampling, storage and attachments; compute binding arrays of buffers, textures and samplers | done | view and binding-array tests |
| 6i. Direct3D 12 binding arrays in render shaders (textures, samplers, storage buffers) and depth/stencil aspect views for sampling | done | render array and depth-view tests |
| 6j. Direct3D 12 ray queries in compute (DXR tier 1.1: bottom-level builds with GPU-sanitized indices, top-level builds from instances, structures bound as root SRVs, HLSL ray queries through DXC cs_6_5); same results as the Metal and Vulkan tests. Cooperative matrices run through the HLSL wave-intrinsic lowering (no new shader model) on devices with a fixed 32-lane wave, which also lets the device advertise subgroups and subgroup size control; only f32 shapes are advertised, and native SM 6.9 cooperative matrices are not used | partial | ray query tests |

Compiler additions: `compileCompute(to: .spirv)` and `compileRenderStage(to: .spirv)` take reflection from the GLSL
writer over the same IR and the binary from the SPIR-V writer, so layout and code come from one module. They inherit
the GLSL writer's binding capacities and capability checks. Every image stays in `VK_IMAGE_LAYOUT_GENERAL`.
Linux: the Vulkan backend builds with Swift 6.2.3 on Ubuntu 24.04 (WSL2) and the whole suite passes there, 1550 tests,
with Vulkan running on Mesa's lavapipe software driver. Xlib presentation was exercised against a real X server
(WSLg) with a swapchain rebuild after a resize, and Wayland presentation to a surface on the WSLg compositor. Not
covered: a hardware Linux GPU driver (lavapipe reports mesh shaders, ray queries, subgroups and 16-bit shaders, so those tests run there; it lacks cooperative matrices, ASTC and ASTC HDR, so those tests skip).

## Mesh shader plan for Vulkan (survey, 2026-11-10)

The mesh path is Metal-only end to end today, so Vulkan needs four pieces, in this order. The development GPU exposes
`VK_EXT_mesh_shader`, so each piece can be checked on hardware, and `spirv-val` ships with the Vulkan SDK.

1. **SPIR-V mesh and task stages.** `SPIRVEntryPlan` only knows vertex, fragment and compute. The mesh output variable is a
   workgroup-space structure (as in `MeshGLSLCodeGenerator`), so the writer needs an epilogue after the user body: a
   barrier, `OpSetMeshOutputsEXT` with the clamped counts, then strided copy loops into the `MeshVerticesEXT` per-vertex
   block, the `PrimitiveTriangleIndicesEXT` array and the user varyings (decorated `PerPrimitiveEXT` for primitive outputs).
   Required modules details: execution model `MeshEXT` (5365) or `TaskEXT` (5364), modes `OutputVertices`,
   `OutputPrimitivesEXT`, `OutputTrianglesEXT`, capability `MeshShadingEXT`, extension `SPV_EXT_mesh_shader`, and SPIR-V 1.4,
   which also requires every referenced global variable in the entry-point interface. Early `return` inside nested
   loops cannot branch to the epilogue in structured control flow and should be rejected or flag-lowered.
2. **Reflected artifacts.** `compileReflectedMeshStage` and the task equivalent accept only MSL. Vulkan needs the same
   reflection (bindings, payload, topology, limits) paired with the SPIR-V words, as done for compute and render stages.
3. **Vulkan pipelines.** Enable `VkPhysicalDeviceMeshShaderFeaturesEXT`, create a graphics pipeline with mesh (and optional
   task) plus fragment stages and no vertex-input or input-assembly state, and advertise `.meshShaders` with the device limits.
4. **Draws.** Plan `drawMesh` like `render` and record `vkCmdDrawMeshTasksEXT`; indirect and counted variants follow.

Direct3D 12 mesh shaders follow the same shape using HLSL from `MeshHLSLCodeGenerator`, DXC `ms_6_5`, and a pipeline-state
stream, which this backend does not create yet.
