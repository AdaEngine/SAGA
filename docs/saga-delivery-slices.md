# Swift wgpu delivery slices

The accepted [compiler scope](adr/port-full-naga-on-swift.md), [platforms](adr/use-metal-and-browser-webgpu.md) and [delivery gates](roadmap.md) define completion. The pinned Naga feature inventory supplies the compiler denominator. `main` is the current local branch and has no baseline commit yet.

Each slice starts with a failing test through the relevant public Swift API. New capabilities expand the supported profile additively and unsupported inputs continue to fail explicitly. Aim for at most 600 reviewable lines per slice. Split each remaining workstream by feature family as it reaches implementation.

| Slice | Workstream | Acceptance | Depends on | Budget |
|---|---|---|---|---:|
| Scalar compute writers | SAGA-6 | Validated WGSL storage write emits MSL, HLSL and GLSL. Metal executes the result and reads back the word. | SAGA-4, SAGA-5 compute subset | 450 |
| Fixed storage arrays | SAGA-6 | Scalar and fixed array loads preserve evaluation order. Indexed accesses use explicit robustness. Verify on Metal with in-bounds and out-of-bounds invocations. | Scalar compute writers | 550 |
| Compute reflection | SAGA-6 | Expose the emitted entry point, workgroup dimensions and exact minimum storage binding sizes. Metal readback tests consume the returned layout. | Fixed storage arrays | 250 |
| Typed compute dispatch on Metal | SAGA-7/8 | Device-owned pipeline and buffer bindings route WGSL through the compiler and queue to GPU readback. Validate binding size, aliases, ownership and device limits before encoding. | Compute reflection | 600 per pipeline or dispatch slice |
| Browser compute boundary | SAGA-6/10 | Emit WGSL and route the same Swift pipeline and command API to an explicit browser compute host. Reject mismatched artifacts and absent command hosts. | Typed compute dispatch | 400 |
| Browser host and Swift routing | SAGA-10 | Build Wasm, instantiate it in a browser and prove that a positive GPU operation and a Swift validation rejection pass through the facade. | Browser compute boundary | 600 per host or routing slice |
| Scalar control flow | SAGA-5/6 | Preserve mutable local initialization, scoped load snapshots, branches and loop continuation across MSL, HLSL, GLSL and WGSL. Run representative shaders on Metal and WebGPU. | Scalar compute writers, browser host | 600 |
| Vertex and fragment writers | SAGA-5/6 | Select one stage from a WGSL module, map vertex ID and render outputs, and render the emitted triangle with pixel readback. | Scalar control flow | 600 |
| Offscreen render submission on Metal | SAGA-7/8 | Route a compiled vertex and fragment pair through device-owned render pipelines and textures to pixel readback. Verify texture pinning and zero initialization. | Vertex and fragment writers | 600 |
| Offscreen render submission on Web | SAGA-10 | Route the same Swift render and copy commands through the browser host. Verify pixels, validation routing and resource cleanup. | Offscreen render submission on Metal, browser host | 600 |
| Interstage varyings | SAGA-5/6/8 | Emit perspective varyings from structured vertex results and direct or structured fragment inputs. Validate pipeline interfaces in Swift and compare perspective-correct pixel values on Metal and WebGPU. | Vertex and fragment writers, offscreen rendering | 600 |
| Shader function calls | SAGA-5/6 | Emit reachable helpers, preserve nested calls and local snapshots, and pass transitive storage captures in MSL. Verify compute side effects and mutable vertex result structures on Metal and WebGPU. | Scalar control flow, interstage varyings | 600 |
| Scalar integer lowering | SAGA-5/6 | Preserve i32 wrapping arithmetic, zero-divisor and overflow behavior, and modulo-32 runtime shifts. Reject tested static faults and verify edge-case GPU readback on Metal and WebGPU. | Shader function calls | 600 |
| Vector integer lowering | SAGA-5/6 | Preserve integer scalar rules lane by lane for concrete integer vectors, including scalar broadcasts, comparisons, conversions and swizzles. Verify Metal and WebGPU readback and GLSL/WGSL round-trips. | Scalar integer lowering | 600 |
| Vertex buffers | SAGA-6/7/8/10 | Emit vertex inputs by location, validate layouts and fetch bounds in Swift, and verify float and integer input streams on Metal and WebGPU. | Interstage varyings, offscreen rendering | 600 per shader or backend slice |
| Compute and graphics parity | SAGA-5/6/7/8 | Expand frontend, validation, reflection, robustness and writers by feature family. Validate actual target artifacts and add Metal and Web rendering and resource operations with readback tests. | Compute dispatch and browser host | 600 per feature slice |
| Remaining compiler formats and tooling | SAGA-5/6 | Validate full input, output, IR interchange, transform and CLI scope against the pinned Naga inventory and applicable external toolchains. | Shared IR, frontend and writer slices | 600 per format or tool slice |
| Presentation and conformance | SAGA-9/10/11 | Verify native presentation, browser canvas, resize, loss and async behavior, then run the applicable pinned CTS matrix with routing evidence. | Metal and Web graphics and resource operations | 600 per presentation or CTS slice |

The compiler slices use the existing validated IR boundary. The Metal pipeline tests exercise the
normal Swift submission path with GPU readback. The browser diagnostic covers compute and offscreen
rendering from Swift/Wasm through the JavaScript WebGPU host to hardware readback. It verifies
Swift-side validation before host submission and fails when Wasm or WebGPU is unavailable. The
reactor ABI uses the Swift 6.3.2 Wasm SDK and stays inside the browser executable. General render
passes, graphics resources, presentation, CTS and full Naga parity are separate acceptance gates.
