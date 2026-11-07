# Use Metal on Apple and browser WebGPU on the web

Date: 2026-11-06. Owner: SAGA project. Status: **accepted by explicit user choice**, recorded during SAGA-2 baseline acceptance. This records an existing choice; it does not select additional shader formats.

## Context

The user selected **Apple/Metal + Swift web build/browser WebGPU**. Earlier task briefs and the original independence ADR predate that choice. SAGA must own its Swift API, validation and shader compiler instead of wrapping Dawn, wgpu or Rust Naga. The selected web backend is the browser API, not a second independently implemented browser GPU driver.

## Decision

- Apple native backend: Metal. Exact OS versions, devices, capability profiles and presentation integration still require a tested matrix.
- Web backend: Swift web build with browser WebGPU. Swift/Wasm toolchain, JS interop library, browser/version matrix and scheduling/lifetime contracts remain follow-on decisions; no implementation or dependency is selected here.
- No Dawn/wgpu/Rust Naga dependency in the SAGA project, directly or transitively, including bundled or linked WebGPU engines. The browser's internal implementation is browser-owned, not a SAGA dependency. This browser boundary is an explicit qualification of the original ADR's broad JS-host wording, not an exception allowing a SAGA FFI wrapper.
- Build an independent Swift WGSL frontend, semantic validation, IR and codegen. The human-input gate **WGSL→MSL versus all Naga formats remains unanswered**. Do not infer all Naga input/output formats, SPIR-V, GLSL or HLSL support from the platform choice; the baseline WGSL inventory does not approve additional formats. Native-extension parity also remains undecided.

## Alternatives and consequences

A native-only Metal product would omit the requested web path. A portable native multi-backend-first implementation would expand the approved scope. Neither is selected. Metal plus browser WebGPU follows the user's choice and avoids owning a browser GPU driver, but creates two different evidence boundaries.

On Apple, future GPU tests must exercise SAGA validation/compiler/backend and real Metal execution. On the web, tests must prove calls reach SAGA Swift API/validation before the intended browser backend. Stock browser CTS bypassing SAGA is not SAGA evidence. A SAGA-routed web CTS pass still uses the browser's WGSL compiler and GPU implementation: it is not proof of SAGA native codegen, independent WGSL compiler execution or Apple backend correctness. Report these profiles separately and never pool them into a native conformance claim.

## Verification and follow-up

SAGA-2 checks documents, inventories, hashes and validator tests only. No Swift build, shader compilation, GPU or CTS result follows from this ADR. Later work must audit the SAGA build/link graph, identify browser-owned implementation boundaries, record per-profile routing evidence and exercise a disabled-SAGA-facade negative control (no silent direct `navigator.gpu` fallback). Disabling the intended browser backend itself is not the negative control for the web profile.

Keep compiler-format and native-parity human-input gates. Resource/concurrency, browser interop/host and concrete matrix ADRs belong to their existing workstreams.

[Independence ADR](implement-independent-webgpu.md) · [Conformance methodology](../conformance.md) · [Roadmap](../roadmap.md).
