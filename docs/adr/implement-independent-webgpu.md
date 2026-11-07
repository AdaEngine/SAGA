# Implement independent WebGPU in Swift

Date: 2026-11-06. Owner: SAGA project; recorded by SAGA-2.
Status: **accepted direction by explicit user requirement**. This accepts independence only, not a backend, platform matrix, JS host or native-extension parity.

> Current applicability: the original record below is preserved. The dated addendum and [platform ADR](use-metal-and-browser-webgpu.md) record the subsequent Apple/Metal and browser WebGPU choice and qualify the browser-host boundary.

## Context

SAGA-1 requests a complete independently implemented Swift WebGPU, not an FFI wrapper around another WebGPU engine. The new checkout initially contains IDEA.md only. The Rust reference is fixed at `a55352911bb26a1b33bc8d5fc03bdebb0a967f5a`, read-only. Swift 6.3.3 / Xcode 26.6 are supplied environment facts; Swift 6.2 language compatibility is the target, to be verified by SAGA-3.

Normative API/language behavior, native wgpu additions and independent CTS are distinct scopes. A compute demo or passing stock-browser CTS cannot establish SAGA conformance. See [conformance methodology](../conformance.md) and [source/license audit](../provenance.md).

## Decision

Implement the public API, state/validation, resource lifecycle, command model and WGSL compiler pipeline in Swift. No Dawn, wgpu or Naga runtime dependency, including indirect/link-time dependencies or JS hosts that route GPU operations into those engines. Native OS GPU APIs and their shader compilers may be backend dependencies after platform approval; they are not the WebGPU/WGSL implementation and cannot replace validation.

Own WGSL lexer/parser, semantic validation, typed IR, reflection, robustness lowering and codegen. Consult pinned specifications and independent CTS for behavior; use Rust source only as explicitly attributed reference. Copy/adapt fixtures only after item-level license and provenance review. This is an independent implementation, **not a certified clean-room process**; reference inspection does not establish clean-room separation.

Require dependency/link denylist and proof-of-routing gates, frozen coverage denominators, negative validation, untrusted-input hardening and real-GPU execution before claiming corresponding support. Public claims are limited to tested revision/profile/matrix, never blanket parity inferred from one slice.

## Alternatives considered

- FFI to Dawn/wgpu: shorter delivery path, but violates explicit independence requirement; rejected.
- Naga as shader runtime/compiler: mature semantics/codegen, but violates explicit requirement; rejected.
- Independent Swift API/validation/compiler with approved native backends: largest effort and security burden, but meets ownership/control goal; accepted.

## Consequences

Positive: transparent ownership of API contracts, diagnostics, security boundary and concurrency; no hidden WebGPU engine dependency.

Costs/risks: substantial WGSL and backend correctness work, cross-platform differences, numerical precision rules, GPU memory safety, async lifetime races, JS-host impedance and independent CTS bridge maintenance. CTS pass is necessary evidence, not proof of all safety properties. License approval is not implied by this ADR.

## Follow-on decisions (proposed, not accepted here)

Platform/backend matrix and native wgpu parity already await the user. Do not repeat clarification requests. Define separate resource/concurrency, WGSL frontend/IR, backend selection, presentation/loss and CTS-host ADRs in their existing roadmap workstreams. No backend/platform was selected by SAGA-2.

## Fitness functions

SAGA-3: build graph/dependency audit must reject Dawn/wgpu/Naga runtime libraries; record Swift language compatibility and ownership contracts. SAGA-10: JS bridge handshake/sentinel and disabled-bridge negative control prove Swift routing. SAGA-11: link audit, pinned CTS results on approved real GPU matrix, security/concurrency stress and truthful release claims. None of these execution checks has run during this documentation task.

Accepted record is immutable by default; later changes require dated addendum or explicit superseding ADR. [Roadmap](../roadmap.md).

## Addendum — 2026-11-06 platform choice and browser boundary

The user has selected Apple/Metal + Swift web build/browser WebGPU. The [platform ADR](use-metal-and-browser-webgpu.md) records this existing decision. It supersedes the earlier pending-platform wording and qualifies the original prohibition on JS hosts routing to other engines: SAGA cannot depend on or bundle Dawn/wgpu/Rust Naga; the selected browser backend may use browser-owned internals. SAGA still owns its Swift API/validation and independent compiler. Browser execution alone is not proof of that compiler or native backend. Additional shader-format scope (WGSL→MSL versus all Naga formats), native parity and the detailed test matrix remain open. No runtime implementation is authorized by this addendum.
