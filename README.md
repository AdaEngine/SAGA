# SAGA

Independent Swift WebGPU runtime and full Naga-family shader compiler, pinned to
wgpu/Naga `a55352911bb26a1b33bc8d5fc03bdebb0a967f5a` as read-only reference.
**The Swift queue runs validated WGSL compute and offscreen rendering on Metal and browser WebGPU.**

- Native runtime direction: Apple / Metal.
- Web direction: Swift / Wasm with an explicit browser WebGPU host boundary.
- Compiler scope: WGSL, GLSL, SPIR-V inputs; WGSL, GLSL, SPIR-V, MSL, HLSL, DOT outputs;
  shared IR, validation, transforms, reflection, robustness, diagnostics, CLI and IR interchange.
- No Dawn, wgpu or Rust Naga implementation/runtime dependencies. No third-party package dependencies.

## Build and test

Requires Swift toolchain 6.2 or newer. Package tools-version is **6.2**; language mode is **6**,
not a fictional `6.2` language mode. Local verified toolchain versions are recorded separately.

```
swift build -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
swift test -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
```

The XCTest compatibility runner can print `Executed 0 tests`; the Swift Testing result must
show a nonzero executed count. Do not use the XCTest line as evidence of test execution.

Run the native diagnostic on a Metal-capable Mac:

```
/usr/bin/swift run --scratch-path .build-apple SAGAMetalDiagnostic --count 4099
```

It stages input, dispatches real GPU work, copies to readback and compares **every** UInt32
with `input*3+7` (wrapping). Missing adapter and timeouts are errors, never CPU fallback.
Direct MSL is backend diagnostics only, **not WGSL/compiler/WebGPU conformance**.

`IndependentShaderCompiler.compile` validates WGSL and emits MSL, HLSL (Shader
Model 6 profile), GLSL 4.50 core or WGSL for one compute entry point in the supported
profile. It supports `global_invocation_id`, `local_invocation_id`, `local_invocation_index`,
`workgroup_id` and `num_workgroups` as direct arguments or fields of entry-point
structures. When HLSL uses `num_workgroups`, bind the dispatch dimensions as
four words `[x, y, z, 0]` at the reflected `hlslWorkgroupCountBinding`
(`b0, space4`). Compute supports signed and unsigned 32-bit scalar
arithmetic, `bool`, `i32` and `u32` locals, `if/else`, and structured loops with `continue`, `break` and
runtime `break if`. `switch` supports signed and unsigned constant selectors,
grouped labels and default clauses. It preserves switch-local breaks and
continues that target the enclosing loop.
Use `compileCompute(_:entryPoint:to:constants:bindingArrayLengths:)` or
`compileWebGPUCompute(_:entryPoint:constants:bindingArrayLengths:)` to select a
named compute entry point. With the name omitted, the compiler selects the sole
compute entry point, even if the module contains render entry points. Compiling
a named entry reflects its reachable bindings.
Compute supports up to eight storage bindings across groups 0–3
with binding indices 0–999. HLSL uses the WGSL group as its register space.
MSL and OpenGL GLSL use the reflected `flatBindingIndex`. Low group-zero
bindings retain their indices and remaining bindings occupy free slots in group
and binding order. Storage types include
`u32`, `i32`, `f32` and their vectors, including fixed and runtime-sized arrays.
A runtime array can occupy the entire binding or follow scalar, vector and fixed-array
fields in a storage structure. Array strides preserve WGSL alignment and padding.
`arrayLength` counts elements in the bound range after the array's
byte offset. Use
`dispatchComputeRanges` to bind slices with 256-byte aligned offsets and an optional
size in bytes. Metal passes runtime lengths to the shader at buffer slot 8.
Compute also accepts up to twelve uniform bindings of at most 64 KiB each.
Bind a buffer with `.uniform` usage using `ComputeBufferBinding` and the same
dispatch API. Uniform layout is reflected separately from storage and follows
WGSL offsets for supported scalars, vectors, fixed arrays and flat structures.
HLSL reads 16-byte constant-buffer registers from the reflected binding and group.
Vertex and fragment shaders use the same uniform layouts. Pass sliced uniform
buffers with the `uniforms` argument of `render` or `renderIndexed`. Stages that
use the same group and binding share one buffer slice.
Array accesses with supported indices return
zero on out-of-bounds reads and discard out-of-bounds writes. The generated
entry point is `saga_main` in MSL, HLSL and WGSL and `main` in GLSL.
`compileCompute` also returns the entry point, workgroup dimensions and minimum
buffer sizes needed to dispatch the shader. The compiler reports unsupported features explicitly.
Reachable helper functions can take and return supported scalar and vector
values, and can access the supported storage buffers. MSL passes captured buffers
as explicit arguments. Compute and render preserve call order and store each
call result before later writes.
Workgroup memory supports fixed-size arrays, scalars and vectors of `u32`, `i32`
and `f32` within 16 KiB. Native writers zero it before user code runs.
`workgroupBarrier()` synchronizes workgroup memory, including through helper
calls in uniform control flow. `storageBarrier()` synchronizes storage-buffer
accesses among invocations of the same workgroup. The GLSL writer declares
writable storage blocks coherent when a storage barrier is reachable.
`workgroupUniformLoad` supports workgroup scalar and vector values and atomic
integers. The compiler requires a uniform address and call site. Native targets
emit a workgroup barrier before and after the read.
Runtime `select` supports scalar conditions and per-component boolean masks.
Both value operands are evaluated before selection. For indexed
`workgroupUniformLoad`, WGSL selects a valid address before reading and masks
the result to zero on an out-of-bounds index, preserving result uniformity.
Compile-time `select` expressions can initialize module and local constants
and determine fixed array lengths and workgroup sizes. Compute and render
writers emit their validated constant values.
WGSL `bitcast<T>` preserves bits between `i32`, `u32` and `f32` scalars or
vectors of the same width. Module and local constants fold the same overloads
and reject non-finite floating-point results.
The integer bit builtins support `i32` and `u32` scalars and vectors in runtime
and constant expressions. `extractBits` and `insertBits` check constant ranges
at shader creation and clamp dynamic offset and count to the word width.
WGSL packing and unpacking cover 8- and 16-bit normalized channels, binary16
float pairs and four packed integer bytes. Constant expressions use the same
lane order as runtime shaders. Normalized signed packing emits the WGSL
rounding formula explicitly to preserve the result at negative ties.
Geometric builtins cover `dot`, `cross`, `length`, `distance`, `normalize`,
`faceForward`, `reflect` and `refract`. Integer `dot` wraps at 32 bits.
`dot4I8Packed` and `dot4U8Packed` multiply four byte lanes from two `u32`
words with signed or unsigned interpretation.
Scalar and vector numeric builtins cover signed absolute value and sign,
min/max/clamp, rounding, fractional parts, saturation and square roots.
Trigonometric and hyperbolic functions, angle conversions, exponentials,
logarithms and `pow` also support scalar and vector calls. These functions
fold in constant expressions with domain and finite-range validation.
Runtime `f32` matrices support column construction and indexing, addition,
subtraction, scaling and matrix or vector products. HLSL represents each WGSL
column as a row and reverses the operands of `mul` to preserve the product.
`transpose` swaps matrix dimensions, and `determinant` accepts square matrices
from 2×2 through 4×4. Both also evaluate in WGSL constant expressions.
Storage and uniform buffers can hold `f32` matrices in the supported scalar,
vector and flat-structure layouts. Storage writes preserve padding between
columns. GLSL reads uniform matrices with two rows from packed `uvec4`
registers to preserve WGSL's eight-byte column stride.
`copyBufferToTexture` uploads whole or partial RGBA8 and BGRA8 textures
through the queue. Use a copy-source buffer, a copy-destination texture,
a source offset aligned to four bytes and a row pitch aligned to 256 bytes
for multi-row copies.
Single-row copies can omit the row pitch.

Compute, vertex and fragment shaders can read a bound `texture_2d<f32>` using `textureLoad`,
`textureDimensions` and `textureNumLevels`. Create the RGBA8 or BGRA8 texture
with `.textureBinding` and supply its resource ID in the dispatch or draw's
`textures` dictionary using the shader's group and binding. A draw can share
one texture binding between both stages. Sampled and attachment mip ranges
on the same texture must be disjoint. The compiler returns a zero texel for
invalid coordinates or mip levels across MSL, HLSL, GLSL and WGSL.

Set `TextureDescriptor.mipLevelCount` to allocate a mip chain on Metal and
WebGPU. It defaults to one level. Texture copies select a level with `mipLevel`
and render commands select an attachment level with `targetMipLevel`. Both
default to level zero. Binding a texture directly exposes the full chain and
each uninitialized level reads as zero.

Use `Device.createTextureView` with `TextureViewDescriptor` to select a mip
range for shader bindings. The selected base is LOD zero in the view. A
single-level view can serve as a render attachment. Copies and presentation
take texture resources. Destroying a texture invalidates its views for future
submissions, while accepted work retains the resources it uses.

A render pipeline can declare a `RenderDepthState` and each draw can supply a
`RenderDepthAttachment` selecting a `depth32Float` mip or single-level view.
Draws can clear or load depth and retain the result for later passes. Depth
testing and depth writes are controlled by the pipeline. Set
`RenderDepthState.bias` to configure constant, slope-scaled and clamped depth
bias for triangle lists and strips. A `depth24PlusStencil8` render attachment
supports independent stencil loading and clearing. Set `RenderDepthState.stencil`
for front and back comparisons, operations and masks, and pass a stencil
reference in the draw's depth attachment. Use `.readOnly` as the depth or
stencil load action to retain and test that aspect with a compatible pipeline.
A `depth32Float` mip can be sampled by the same pass when its depth attachment
uses `.readOnly`. For a `depth24PlusStencil8` texture with `.textureBinding`
usage, create a `.depthOnly` view for depth shaders or a `.stencilOnly` view
for unsigned texture shaders. Read stencil from the first component of
`textureLoad`. Bind an `.all` view as the render attachment and mark the
sampled aspect read-only when using both in the same pass. Buffer copies of
a `depth24PlusStencil8` texture use `aspect: .stencilOnly` and transfer a
full mip with one byte per texel. Use four-byte-aligned buffer offsets and
256-byte-aligned row strides. A fragment shader can
return `@builtin(frag_depth)` by itself or alongside its color and sample mask
outputs. The pipeline requires a depth attachment for that shader. A draw
with only a depth output uses its color attachment for the pass clear. A depth
texture can be copied to a buffer as raw f32 values. With `.textureBinding` usage,
`texture_depth_2d` shaders can read scalar depth, query dimensions and count mip
levels. Shader bindings and writable attachments must use disjoint mip ranges.
Create a comparison sampler by setting `SamplerDescriptor.compare`. Use it
with `textureSampleCompare` in uniform fragment control flow or with
`textureSampleCompareLevel` at the view's base mip in any supported stage.
`textureGatherCompare` returns four base-level depth comparisons in WGSL
texel order, with an optional constant offset.

For raw depth sampling, bind a non-comparison sampler with nearest
minification, magnification and mip filtering. `textureSample` returns a scalar
in uniform fragment control flow. `textureSampleLevel` takes an `i32` or `u32`
mip level relative to the bound view and returns a scalar in any supported
stage. `textureGather` returns four base-level depth values in WGSL texel order.
All three accept an optional constant texel offset.

For 2D float color textures, compute, vertex and fragment shaders can use
`textureSampleLevel` with a float mip level and a non-comparison `sampler`. Create a sampler with `Device.createSampler`
and bind its resource ID in the draw or dispatch `samplers` dictionary. Samplers
support nearest and linear filtering plus clamp-to-edge, repeat and mirror-repeat
addressing in U and V. For GLSL, `glslTextureSamplerBindings` reflects the texture
unit for each pair, so one texture can be used with different sampler states.
Vertex and fragment stages compiled from the same WGSL module use the same unit
for a shared pair.

Fragment shaders can use `textureSample` and `textureSampleBias` in uniform
control flow. `textureSampleGrad` accepts explicit `vec2<f32>` gradients in
compute, vertex and fragment shaders. These 2D float sampling overloads support
the normalized coordinate and sampler modes above. The optional texel offset is
a constant two-component signed integer vector with each component in -8...7.
`textureGather` reads one constant color component from each of the four
neighboring texels at a 2D texture view's base mip. It preserves WGSL's result
order and accepts the same texel offset. Integer textures return `vec4<i32>`
or `vec4<u32>` and use a sampler with nearest filters.
For `texture_2d<f32>`, `textureSampleBaseClampToEdge` uses the view's base mip
and clamps coordinates to the centers of the outer texels, even with repeat
addressing. It preserves the sampler's filtering.

For multisampled color or depth, set `sampleCount: 4` on the texture and the
render pipeline. Multisampled attachments have one mip level. Bind them as
`texture_multisampled_2d<f32>` or `texture_depth_multisampled_2d` to read an
individual sample with `textureLoad`. Use `textureNumSamples` and
`textureDimensions` to query the attachment. Out-of-bounds loads return zero.
Fragment shaders can read `@builtin(sample_index)` and `@builtin(sample_mask)`.
Return a structure with a color at location zero and a `sample_mask` output
to select the covered samples that the fragment writes.
A pipeline's `sampleMask` further restricts which samples are written.
Use `alphaToCoverageEnabled: true` with a multisampled, blendable color
attachment and a fragment alpha output. In this mode the fragment's alpha
produces the coverage mask, which is intersected with the pipeline mask.
Explicit fragment `sample_mask` output and alpha-to-coverage are mutually
exclusive.
Pass `resolve: .init(texture: destination, mipLevel: level)` to a draw to
resolve the color samples while storing the multisampled attachment for later
reads. The destination must be a single-sample render attachment with matching
format and extent at the selected mip. A single-level view uses mip zero.

`rgba8Sint` and `rgba8Uint` attachments take `vec4<i32>` and `vec4<u32>`
fragment outputs respectively. Use `textureLoad` with the matching scalar type
in `texture_2d` or `texture_multisampled_2d` to read the integer channels.

Compute and fragment shaders can write 2D storage textures with `textureStore`
and query their dimensions. Create the texture with
`.storageBinding` and bind a single mip in the dispatch or draw `textures`
dictionary. Out-of-bounds stores leave the texture intact. Sampled bindings,
storage writes and color attachments can share a texture when the writable mip
ranges are disjoint from the other bindings.

The core storage formats cover the R32, RG32, RGBA8, RGBA16 and RGBA32
families. Devices with readable storage texture support can use `read`, and
the three R32 formats also support `read_write` on capable devices. Loads
from R32 and RG32 fill absent color channels with zero and alpha with one.
An out-of-bounds load returns zero in every lane. Copies of 8- and 16-byte
texels require buffer offsets aligned to their texel size. In uniform compute
control flow, `textureBarrier()` makes preceding storage texture writes
visible to later accesses in the same workgroup. Render pipelines can bind
read-only storage textures in either graphics stage and read-write storage
textures in fragment shaders. A shared read-only binding uses one logical slot.
The same formats and access modes are available with `texture_storage_2d_array`.
Bind a `.d2Array` view of one mip and pass a view-relative layer to `textureLoad`
or `textureStore`. `textureNumLayers` returns the view's layer count. Writable
views can share a texture when their subresources are disjoint from other bindings.

`atomicLoad`, `atomicStore`, `atomicAdd`, `atomicSub`, `atomicMin`, `atomicMax`,
`atomicAnd`, `atomicOr`, `atomicXor`, `atomicExchange` and
`atomicCompareExchangeWeak` support `atomic<i32>` and `atomic<u32>` workgroup
and read-write storage scalars and arrays. Atomic array accesses use the same
bounds checks as ordinary storage arrays. Compare-exchange returns both the old
value and an exchange flag. An out-of-bounds compare-exchange returns zero and
`false`.
Flat storage structures can mix supported ordinary and atomic fields, including
fixed arrays and an optional runtime-sized tail. HLSL bindings that contain
atomic fields use raw UAVs and report `hlslUsesByteAddressBuffer` in reflection.
The minimum binding size of a runtime-tailed structure includes its tail
padding with one array element. `arrayLength` uses the actual bound range.
Runtime integer division and remainder handle zero divisors and signed
overflow. Runtime shifts use the WGSL modulo-32 count. The same rules apply
per lane to concrete `vec2`, `vec3` and `vec4` of `i32` or `u32`. The
compiler also supports integer vector comparisons, signedness conversions,
swizzles and scalar broadcasts for arithmetic.

`compileRenderStage` selects a vertex or fragment entry point from a WGSL module
and emits stage-specific MSL, HLSL, GLSL or WGSL. The graphics profile
supports procedural vertices with `vertex_index` and `instance_index`, vertex buffer inputs by
`@location`, `vec4<f32>` position and location-zero color outputs. Interstage
varyings support perspective, linear and flat interpolation for `f32` scalars
and vectors, plus flat interpolation for `i32` and `u32` scalars and vectors.
The Swift device validates varying locations, scalar types, interpolation
and sampling when creating a render pipeline. When running emitted GLSL with
`flat, first` varyings, set `glProvokingVertex(GL_FIRST_VERTEX_CONVENTION)`. Desktop GLSL adjusts the vertex clip-space Y and Z
using the pinned Naga default convention. The offscreen render test uses the
same compiled stages as the runtime queue.
Fragment shaders can read `@builtin(position)` and `@builtin(front_facing)`.
The position contains pixel coordinates, viewport depth and reciprocal clip
`w`. When linking the generated GLSL pair in OpenGL, use `GL_CW` for the
WebGPU default front face to account for the generated vertex Y flip.
Fragment shaders can also use `dpdx`, `dpdy` and `fwidth` with the default,
fine or coarse variants. Validation checks uniform control flow through helper
calls. The Metal writer maps fine and coarse to its native derivative functions.
Module `diagnostic` directives and function `@diagnostic` attributes control
`derivative_uniformity`. `parseAndValidate` returns nonfatal messages in
`ValidatedModule.diagnostics`. Stage restrictions and workgroup barrier
validation still apply.
HLSL vertex shaders that use draw offsets report `vs_6_8` in `hlslProfile`.
GLSL vertex shaders using `instance_index` target desktop GLSL 4.60 for
`gl_BaseInstance`. The generated shader adds the base offset on these targets.
Vertex output structures can also be assembled in a helper using mutable
members. Supported vector locals and structure locals preserve snapshots when
their members are later reassigned.

Pass the stages to `device.createRenderPipeline`, create an RGBA8 or BGRA8
texture with `device.createTexture`, and submit `BufferCommand.render`,
`copyTextureToBuffer` and `read` through `device.submitCommands`. The current
render command opens one pass and issues one draw. Pass `clear:` to clear the
color attachment or `colorLoad: .load` to retain its contents from earlier
commands. Set the pipeline's `colorState` to configure independent RGB and
alpha blending or a channel write mask. Use the draw's `blendConstant` with
constant blend factors. Return a fragment output structure to write several
color locations. Configure locations 1 through 7 with `additionalColorTargets`
and bind the same locations in the draw's `additionalColors`. Each target has
its own format and color state. Each attachment has its own mip, load action
and optional resolve. Location 0 uses the draw's primary target. The portable
color attachment budget is 32 bytes per sample.
Use the draw's `raster` state to set viewport coordinates and depth range
or restrict the draw to a scissor rectangle. The scissor is measured against
the selected attachment mip. Metal shaders that write `frag_depth` use the
viewport depth range bound at `RenderStageArtifact.metalViewportDepthBufferIndex`.
The pipeline's `primitive.topology` selects points, line lists or strips, and triangle lists
or strips. The default is a triangle list. Indexed strips require a matching
`stripIndexFormat`. The largest index value restarts the strip. The Metal
runtime compiles point vertex stages with `SAGA_POINT_TOPOLOGY` to set the
point size to one pixel.
Use `primitive: .init(frontFace: .clockwise, cullMode: .back)` to configure
winding and face culling. The default is counterclockwise with culling disabled.
Use `renderIndexed` with an `IndexBufferBinding` and `IndexedDraw` for uint16
or uint32 indices, optional byte slices, `firstIndex` and signed `baseVertex`.
Swift validates the index buffer range before submission. Metal checks the
index values on the GPU and skips the draw if a vertex fetch would be out of bounds.
Both draw commands accept `DrawInstances` with an instance count and first
instance. Non-indexed draws also accept `firstVertex`. Set a layout's step mode
to `.instance` to read one record per instance. Swift validates the vertex and
instance ranges independently.
Vertex layouts accept float32, uint32 and sint32 attributes with one
to four components. Use ordered buffer slots, byte offsets and strides, including
zero stride for constant attributes. Swift validates shader input types and fetch
ranges before encoding. Texture copies cover the full 2D image and take a 256-byte-aligned
row stride. The last row requires space only for its pixels.
Use `copyTextureToTexture` to copy a color region directly between matching
textures or different mips of the same texture. Depth and multisampled copies
require matching full mip extents and sample counts. A combined depth/stencil
copy transfers both aspects.
Set `arrayLayerCount` to allocate a sampled 2D texture array, with up to 256
layers in the portable profile. Use `arrayLayer` in texture copies and uploads
to select a slice. A view can select mips and layers with `baseArrayLayer`
and `arrayLayerCount`. Use `dimension: .d2` for a single-layer storage view or
`.d2Array` to bind an array view, even when it contains one layer. A single-layer
view can serve as a render attachment while the same pass samples disjoint
layers. WGSL array textures support `textureLoad`, `textureDimensions`,
`textureNumLevels` and `textureNumLayers`. Color and depth arrays support
sampling and gathering, including comparison sampling for depth arrays. Sampling
clamps the layer index to the range of the bound view.
Use `.cube` or `.cubeArray` views of a square 2D texture array. Each cube uses
six consecutive layers in `+X, -X, +Y, -Y, +Z, -Z` order. Cube sampling and
gathering take a 3D direction, and `textureNumLayers` counts cubes in an array
view. Explicit base-level depth comparison on a GLSL cube array requires
`GL_EXT_texture_shadow_lod`.
Create a sampled volume with `TextureDescriptor` using `dimension: .d3` and
`depth`. All three axes shrink with the mip level. The portable profile
supports up to 2048 texels per axis for volume textures. A `.d3` view selects
mips and includes all Z slices in each selected mip. Pass `z` and `depth` to
copy several consecutive planes. Buffer copies pack planes consecutively at
the specified row stride, with no extra padding between planes.
`texture_storage_3d` binds one mip of a volume and addresses it with XYZ texel
coordinates. Compute and fragment shaders can write these views. Readable
views use the same format and access rules as 2D storage textures.
For rendering, allocate the volume with `.renderAttachment` and supply
`targetDepthSlice` on the draw or `depthSlice` on an additional color attachment.
The slice index is relative to the selected mip. A pass can render to different
slices of that mip as separate color attachments. Shader bindings can use other
mips of the volume in the same pass.
Use `dimension: .d1` for 1D sampled and storage textures. The portable runtime
uses one mip and requires `height` and `depth` to be one. WGSL uses scalar
coordinates for loads, stores and sampling, and returns the width from
`textureDimensions`.
Use `device.present(texture)` to present a copy-source color texture to the
attached surface in Swift queue order. The browser resizes its canvas to the
source dimensions. On macOS, attach a `MetalPresentationSurface` to a
`CAMetalLayer` and pass it to `MetalBackend.makeDevice(surface:)`. Metal
encodes on the main actor so the layer and drawable stay in the same isolation
domain. GPU completion remains asynchronous.

Run the AppKit window example with
`sh scripts/build-and-run-metal-window.sh`. Add `--smoke` to present two frames
at different sizes, validate the offscreen source pixels and wait for both
native presentation callbacks before exiting. General render pass encoding
remains separate work.

On Metal, pass the result of `compileCompute(.wgsl(source), to: .msl)` to
`device.createComputePipeline`, then submit `BufferCommand.dispatchCompute`
through `device.submitCommands`. Dispatch validates the pipeline owner, storage
binding slots, buffer sizes, usage and writable aliases before submission.
Run the WGSL queue and GPU readback test with
`swift test --filter wgslPipelineRunsThroughDeviceQueueAndMetalReadback`.

Play builds a headless macOS app with the same CLI binary; its report is appended to
`.build-play/saga-metal-diagnostic.log`. See [current verification](docs/metal-verification.md)
and [backend ADR](docs/adr/select-initial-backends.md) for the exact checked recipe and limits.
`BrowserComputeBackend` accepts a compiled WGSL shader and requires an injected
`BrowserGPUComputeHost`. Rendering requires a `BrowserGPURenderHost`. The
browser diagnostic supplies both through a JavaScript WebGPU host and uses a
Swift/Wasm reactor to run the same compute, texture and render commands.
The sample attaches an HTML canvas and presents the rendered frame. Applications
can run the diagnostic without a canvas, or release an attached canvas with
`closeDiagnosticCanvas`.

Build with the Swift 6.3.2 toolchain and its matching Wasm SDK, then serve the diagnostic locally.

```sh
sh scripts/build-web-diagnostic.sh
python3 -m http.server 8799 --bind 127.0.0.1 --directory .build-web-diagnostic/site
```

Open [the diagnostic](http://127.0.0.1:8799/) in a WebGPU browser. The result reports
the GPU readback, verified integer edge cases, checked runtime array elements, vertex stream and indexed draw
counts, instances per draw, verified render pixels, a color from the perspective gradient,
Swift-side validation, and the number of submissions
received by the JavaScript host. A hardware adapter is required.

With local Playwright Core and Chromium, the integration test also checks that the
diagnostic fails when the Wasm module is unavailable. Set `SAGA_PLAYWRIGHT` to the
local module path and `SAGA_CHROMIUM` to the browser executable if they are outside
the standard Playwright installation.

```sh
node scripts/test-web-diagnostic.mjs .build-web-diagnostic/site/SAGAWebDiagnostic.wasm
```

See `docs/foundation.md`, `docs/public-api-review.md` and
`docs/adr/define-resource-and-concurrency-model.md` for ownership, module boundaries,
verification recipes, and remaining work. The accepted SAGA-2 baseline in `docs/` remains
historical documentation evidence, not compiler or runtime conformance.
See the [delivery slices](docs/saga-delivery-slices.md) for the path to Metal, Web
and the full Naga compiler scope.
