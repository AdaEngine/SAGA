# Select the initial Metal command backend and preserve the browser boundary

- Status: accepted within SAGA-4 implementation authority; diagnostic ABI only
- Date: 2026-11-06
- Supersedes no platform choice: [Apple/Metal and browser WebGPU](use-metal-and-browser-webgpu.md)
  was already accepted; [full Naga](port-full-naga-on-swift.md) is unchanged.

## Context

SAGA-3 provided a trusted BackendDevice, actor-owned resources, validation and submission
lifetime harness, but no actual GPU execution. SAGA-4 requires a real adapter/device/queue,
compute dispatch and checked readback, without Dawn/wgpu/Rust Naga, CPU fallback or a claim
that hand-authored MSL validates the independent compiler. Rust remains read-only.

## Decision

Keep API/Core/Backend/Types platform-neutral. MetalAdapter explicitly discovers the OS
Metal device. MetalDevice owns one real MTLDevice/MTLCommandQueue and native registries
inside an actor. Capabilities advertise native maxBufferLength and no unverified features.
Missing adapter throws GPUError.adapterUnavailable; non-Metal targets cannot silently succeed.

Add optional CommandBackendDevice SPI alongside the existing BackendDevice. BufferCommand
carries only immutable IDs and values: staged UInt32 writes, diagnostic two-storage-buffer
compute dispatch with a UInt32 count, GPU copies, and ordered read snapshots. This limited
ABI is explicitly **not** the full WebGPU bind-group/pipeline/encoder API. The Metal-only
makeDiagnosticPipeline factory compiles the fixed MSL kernel `output[i] = input[i]*3u+7u`.
There is no API accepting arbitrary MSL through Core, no fake WGSL pipeline or CPU evaluator.
The CPU only generates input, checks every returned element and copies completed readback.
Pipeline IDs are backend-owned, device-namespaced names; Metal checks their existence before
encoding. The fixed native pipeline is cached once; owner handles are removed on close.

Device validates the entire command batch (liveness, namespace, flags, sizes, alignment,
overflow-safe ranges and forbidden aliases) before pinning and before backend entry. It
uses the same generic enqueue path as the foundation resource-use harness, preserving
acceptance order, error scopes, cancellation and deferred destruction. Readbacks are immutable
snapshots copied by GPU at each read command, not host reads of a buffer that later commands
may mutate. Metal revalidates the trusted SPI's descriptors and resource/pipeline membership.

All buffers in this initial native slice use shared storage; no pointer escapes the Metal
module. New buffers are zeroed before publication. CPU writes target distinct upload staging
buffers, then blit into GPU resources in queue order. Native memory is read only after real
command completion. No unsafe Sendable/isolation escape or new package/link dependency is added.

### Timeout, retirement and teardown refinement

This deliberately refines the SAGA-3 eventual-completion contract in
[ownership ADR](define-resource-and-concurrency-model.md): a successful submission completes
after actual GPU retirement. A **terminal timeout** ends the caller's bounded wait but must
not be interpreted as physical GPU retirement. Core marks the device lost and logically
releases its IDs. Metal retains command buffer, native resources and pipelines in an in-flight
quarantine until the real completion callback. No timed-out snapshot is read. Late callbacks
are idempotent through a single-consumer completion latch. A callback that never arrives may
retain this quarantine indefinitely: conservative memory retention is preferable to unsafe
release, and Metal has no safe universal force-cancel of committed command buffers here.

BackendDevice.close(owner:) has a default no-op for source compatibility; Core invokes it
after all logical operations retire. Metal removes owner buffers/pipeline handles, but never
quarantined batches. It can retain one cached fixed pipeline until the backend is released.
BrowserBackend forwards owner teardown to its injected host. A future browser command backend
can implement the optional command SPI/JS mapping; it is not implemented or claimed here.
No MainActor or Metal object requirement was introduced into shared contracts.

The completion deadline covers submitted GPU work, not arbitrary OS driver calls such as
synchronous MSL compilation. External verification commands additionally had process deadlines.
There is no claim of driver-reset/recovery support or a production WebGPU device-loss matrix.

## Alternatives and consequences

- Put all compute work into the CLI: smaller patch, but bypasses shared validation/lifetime;
  rejected. The CLI uses Core and the real optional command SPI.
- Add MSL handles/source or Metal-specific fields to Core: fast but obstructs the chosen web
  path; rejected. Only the explicitly diagnostic factory is native-specific.
- Force release/cancel GPU resources at timeout: unsafe; rejected in favor of quarantine.
- Build the full WebGPU encoder/bind-group/compiler surface in this slice: outside SAGA-4;
  deferred to the existing resource/compiler/render workstreams, not dropped from root scope.

Throughput remains conservative (Core serializes completion order), copies are deliberately
explicit, and the command ABI is narrow. Mapping/unmapping, textures, arbitrary shaders,
render/presentation, browser execution, full Naga and CTS are still separate requirements.

## Verification

[Current verification](../metal-verification.md) records GPU hardware, exact native/Wasm
commands, negative tests, callback-gated timeout/quarantine test, snapshots, compile-time
contracts, policy and Play recipe. A bounded real GPU run is not a WebGPU conformance claim.
