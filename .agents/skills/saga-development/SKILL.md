---
name: saga-development
description: Work on SAGA's Swift shader compiler, GPU runtime, and platform backends. Use for shader lowering, device and queue contracts, backend changes, or verification of compiler and GPU behavior.
---

# SAGA development

## Find the owning layer

| Area | Start here |
| --- | --- |
| Runtime API and descriptors | `SAGA` exposes the facade and aliases. `SAGATypes` owns backend-neutral descriptors, resource IDs, features, and limits. |
| Resource and queue semantics | `SAGACore/Device.swift` owns resource lifetimes, validation, error scopes, and queue acceptance. `SAGABackend` defines the typed command and completion contracts. |
| Shader pipeline | `SAGACompiler/Compiler.swift` orchestrates `SAGAFrontends` → `SAGAShaderIR` → `SAGAValidation` → `SAGABackends`. |
| Execution | `SAGAMetal` encodes native commands. `SAGAWebGPU` forwards to injected host protocols. The diagnostic reactor in `SAGAWebDiagnostic` and the JavaScript files in `web/` exercise that boundary. |

## Preserve the contracts

- Validate raw IR with the shared semantic validator before issuing a witness. Revalidate after transforms or decoding. Preserve diagnostic spans and shader budgets, including when handling malformed input. [Validation boundary](../../../docs/adr/validate-raw-shader-ir-before-witness.md)
- Keep ownership, usage, capability, and range checks on the Swift device side before encoding. Accepted work pins resources through backend retirement. Cancelling a ticket waiter leaves accepted GPU work running. Use explicit `Device.close()` at the owner boundary. [Ownership and concurrency](../../../docs/adr/define-resource-and-concurrency-model.md)
- Keep platform objects inside backends and honor the Metal main-actor boundary. The browser uses an injected WebGPU host and reports missing capability or host support explicitly.
- Treat wgpu/Naga at `a55352911bb26a1b33bc8d5fc03bdebb0a967f5a` as a read-only implementation reference. Use the pinned GPUWeb specs for normative rules and record provenance for fixture adaptations. [Provenance](../../../docs/provenance.md)
- Use the [accepted compiler scope](../../../docs/adr/port-full-naga-on-swift.md) and [delivery slices](../../../docs/saga-delivery-slices.md) for direction. Earlier baseline reports capture decisions and results as of their own dates.

## Verify the layer you changed

Run from the repository root. Filter Swift tests to the affected contract while iterating.

```sh
swift build -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
swift test -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
python3 -B docs/tools/verify_docs.py
```

Use the Swift Testing summary to count executed tests. The XCTest compatibility runner can report zero. Keep Swift tests, external shader validation, Metal readback, browser readback, and routed CTS results separate when reporting coverage.

- Native backend smoke on a Metal-capable Mac. `swift run SAGAMetalDiagnostic --count 4099` dispatches native MSL and checks every output word. For WGSL through the compiler and queue, run `swift test --filter wgslPipelineRunsThroughDeviceQueueAndMetalReadback`. [Metal verification](../../../docs/metal-verification.md)
- Browser diagnostic. Select Swift 6.3.2 and its matching Wasm SDK, then run `sh scripts/build-web-diagnostic.sh`. Use `node scripts/test-web-diagnostic.mjs .build-web-diagnostic/site/SAGAWebDiagnostic.wasm` with local Chromium, Playwright Core, and a hardware WebGPU adapter.
- Look up binding layouts, shader profiles, textures, draws, and presentation in the [shader and runtime notes](references/shader-and-runtime-notes.md). Use [frontend verification](../../../docs/frontend-verification.md) when reporting corpus results.
