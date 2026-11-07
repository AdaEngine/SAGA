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
profile. It supports `global_invocation_id`, signed and unsigned 32-bit scalar
arithmetic, `bool`, `i32` and `u32` locals, `if/else`, and structured loops with `continue`, `break` and
runtime `break if`. Storage supports reads and writes at group 0, slots 0–7. Storage types include
`u32` and fixed arrays of `u32`. Array accesses with supported indices return
zero on out-of-bounds reads and discard out-of-bounds writes. The generated
entry point is `saga_main` in MSL, HLSL and WGSL and `main` in GLSL.
`compileCompute` also returns the entry point, workgroup dimensions and minimum
buffer sizes needed to dispatch the shader. The compiler reports unsupported features explicitly.
Reachable helper functions can take and return supported scalar and vector
values, and can access the supported storage buffers. MSL passes captured buffers
as explicit arguments. Compute and render preserve call order and store each
call result before later writes.
Runtime integer division and remainder handle zero divisors and signed
overflow. Runtime shifts use the WGSL modulo-32 count. The same rules apply
per lane to concrete `vec2`, `vec3` and `vec4` of `i32` or `u32`. The
compiler also supports integer vector comparisons, signedness conversions,
swizzles and scalar broadcasts for arithmetic.

`compileRenderStage` selects a vertex or fragment entry point from a WGSL module
and emits stage-specific MSL, HLSL, GLSL or WGSL. The graphics profile
supports procedural vertices with `vertex_index`, vertex buffer inputs by
`@location`, `vec4<f32>` position and location-zero color outputs, and perspective-interpolated `f32` scalars and
vectors between stages. The Swift device validates varying locations, types
and sampling when creating a render pipeline. Desktop GLSL adjusts the vertex clip-space Y and Z
using the pinned Naga default convention. The offscreen render test uses the
same compiled stages as the runtime queue.
Vertex output structures can also be assembled in a helper using mutable
members. Supported vector locals and structure locals preserve snapshots when
their members are later reassigned.

Pass the stages to `device.createRenderPipeline`, create an RGBA8 or BGRA8
texture with `device.createTexture`, and submit `BufferCommand.render`,
`copyTextureToBuffer` and `read` through `device.submitCommands`. The current
render command clears one color attachment and issues one non-indexed triangle
list draw. Vertex layouts accept float32, uint32 and sint32 attributes with one
to four components. Use ordered buffer slots, byte offsets and strides, including
zero stride for constant attributes. Swift validates shader input types and fetch
ranges before encoding. Texture copies cover the full 2D image and take a 256-byte-aligned
row stride. The last row requires space only for its pixels. Rendering to a
window or canvas and general render pass encoding remain separate work.

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

Build with the Swift 6.3.2 toolchain and its matching Wasm SDK, then serve the diagnostic locally.

```sh
sh scripts/build-web-diagnostic.sh
python3 -m http.server 8799 --bind 127.0.0.1 --directory .build-web-diagnostic/site
```

Open [the diagnostic](http://127.0.0.1:8799/) in a WebGPU browser. The result reports
the GPU readback, verified integer edge cases, vertex stream count, verified render pixels, a color from the perspective gradient,
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
