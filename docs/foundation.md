# SAGA-3 foundation guide

Date: 2026-11-06. Scope: library boundaries and safety contracts, not an implemented WebGPU runtime.
The [resource/concurrency ADR](adr/define-resource-and-concurrency-model.md) is authoritative for
foundation semantics. [Full compiler scope](adr/port-full-naga-on-swift.md) remains unchanged.

## Module DAG

| Module | Direct dependencies | Responsibility |
| --- | --- | --- |
| SAGATypes | none | immutable descriptors, errors, feature/limit subset, resource/submission IDs |
| SAGABackend | SAGATypes | trusted Sendable backend device SPI, no platform handles |
| SAGACore | SAGATypes, SAGABackend | actor device, validation, lifetime, ordering, cancellation, scopes |
| SAGA | SAGATypes, SAGACore, SAGABackend | public facade with GPUDevice/GPUBufferDescriptor/GPUBackendDevice aliases |
| SAGAShaderIR | none | input/output families, diagnostics/spans, raw IR/transform/interchange contracts |
| SAGAValidation | SAGAShaderIR | validator and non-publicly-constructible validated witness |
| SAGAFrontends | SAGAShaderIR | WGSL/GLSL/SPIR-V frontend interface |
| SAGABackends | SAGAShaderIR, SAGAValidation | WGSL/GLSL/SPIR-V/MSL/HLSL/DOT generator interface |
| SAGACompiler | four compiler modules above | typed parse/validate/generate coordinator, unsupported default compiler |
| SAGAMetal | SAGATypes, SAGABackend | platform-isolated native entry point, deliberately unsupported |
| SAGAWebGPU | SAGATypes, SAGABackend | explicit BrowserGPUHost injection, no JS implementation/fallback |

All 11 targets are libraries. There are four library products plus one Swift Testing test target.
API/Core never import Metal. Compiler modules never depend on runtime or platform modules.
The manifest and sources are checked by [runtime policy](runtime-policy.json); external dependencies
are empty, not merely a substring denylist. See [policy notes](policy-notes.md) for checks and CI.

## Ownership and use

A trusted backend implementation advertises immutable `DeviceCapabilities`. Create the actor with
`try await Device.make(backend:requiredFeatures:)`; no global backend is silently selected.
Pass `BufferDescriptor` values to `createBuffer`. Returned IDs are Sendable names; copying an ID
neither destroys nor extends ownership. The device owns registry state and backend lifetime.
Keep the device alive while using IDs. Explicitly destroy buffers and await `device.close()` at
the owning component's shutdown boundary. `close` must not be called recursively from a backend
operation that it is waiting to retire. Backend implementors own platform object release even if
an API owner forgets explicit close; no hidden async deinit is provided here.

`submitResourceUse` exercises validation/pinning/order with a real completion contract and a test
backend. It does **not** encode commands, dispatch shaders or return fabricated GPU results.
Ordering is actor acceptance order, not the order in which unrelated client Tasks were created.
A cancelled ticket waiter does not cancel already accepted work. Popped scopes await operations
that started while they were active and are not affected by scopes pushed later.

## Compiler boundaries and downstream scope

The pipeline is generic over trusted Swift frontend, validator and generator implementations.
Raw IR cannot be passed directly to codegen: validation produces a witness. Test-only `TestIR`
checks control flow; it is not SAGA's final IR or a shader translator. Every default compiler
input/output request throws unsupported, including all 3 input x 6 output test pairs. This
negative Cartesian test does not assert that upstream supports arbitrary all-to-all translation.

[Compiler feature inventory](compiler-feature-inventory.json) pins source declarations/options/
capability constants/CLI arguments and fixture hashes to the Rust revision. It supplements the
16-component [scope manifest](naga-scope.json); neither is a semantic conformance denominator.
All compiler feature implementations remain not-started/not-run. Upcoming stages must refine
capability combinations, enum variants, semantic validations, fixture licensing and expanded
case denominators. No Rust source or fixture was copied into production Swift code.

- SAGA-4/7: concrete resources, mapping, command representation, GPU queue/backend ownership.
- SAGA-5: WGSL/GLSL/SPIR-V parsers, complete typed IR and semantic validation/diagnostics.
- SAGA-6: every accepted output family, reflection/layout/constant evaluation, robustness,
  compaction/overrides/transforms, full options, CLI, serialization and binary interchange parity.
- Web workstream: JS bridge, main-executor/browser host policy and SAGA-routed browser execution.
- SAGA-11: full format/options/capability/fixture parity, native+web runtime/release gates.

The root remains open. Full native wgpu extensions and Vulkan/D3D/OpenGL runtime backends are not
implicitly added by full Naga scope. Accepted SAGA-2 inventories and original evidence are
preserved; current checks use a separate runtime policy and current evidence files.

## Toolchains and build recipes

Manifest tools-version: 6.2. Language mode: 6 with strict concurrency, not "Swift language 6.2".
Local compilers discovered: Apple Swift 6.3.3/Xcode 26.6 and upstream Swift 6.3.2. Swift 6.2.x
compiler is not locally installed; CI includes a 6.2.4 job, but an unexecuted CI job is not proof.
Installed 6.2.4 Wasm SDK without matching compiler is not a 6.2 compiler verification.

Native checks (actual toolchain selected explicitly to avoid PATH/login-shell differences):

```
/usr/bin/swift test --scratch-path .build-apple -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
/usr/bin/swift build --scratch-path .build-apple -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
```

Matched web compilation (verified local path, SDK and all library targets):

```
/Users/vlad-prusakov/Library/Developer/Toolchains/swift-6.3.2-RELEASE.xctoolchain/usr/bin/swift build --scratch-path .build-wasm --swift-sdk swift-6.3.2-RELEASE_wasm -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
```

Output objects are WebAssembly targeting wasm32-unknown-wasip1. This is compilation portability
for the future browser host, not a linked browser application, JS integration, browser tests or
GPU execution. There is no runnable product and no Play recipe. A future executable gets its own
verified launch configuration, not a library-shaped fake app.

Current docs/policy/reference checks:

```
SAGA_REFERENCE=/Users/vlad-prusakov/Developer/wgpu python3 -B docs/tools/verify_docs.py
python3 -B docs/tools/runtime_policy.py
```

See [verification record](foundation-verification.md) for exact checks and limitations.

## SAGA-4 addendum — 2026-11-06

The SAGA-3 inventory/toolchain results above are historical. There are now 11 library targets,
one diagnostic executable and one test target. Metal is implemented for the deliberately narrow
command slice; the independent compiler is still unsupported. CommandBackendDevice is an optional
refinement; existing browser hosts remain valid and reject unsupported commands without fallback.
The new owner teardown hook is forwarded by BrowserBackend. See
[backend ADR](adr/select-initial-backends.md) and [Metal verification](metal-verification.md)
for timeout quarantine, command ABI, current tests and a genuinely verified Play recipe.
Web compilation now selects all four library roots explicitly, not the native CLI executable.
