# Foundation public API review

Date: 2026-11-06. Scope: SAGA-3 Package.swift and all Sources/Tests. This is an engineering
review of foundation boundaries, not a stable API guarantee or external certification.
The accepted [full-Naga scope](adr/port-full-naga-on-swift.md) remains unimplemented.

## Outcome

Approve foundation for subsequent implementation after the findings below are fixed and final
verification passes. Risk remains medium: actor reentrancy and trusted backend completions need
continued regression coverage as real commands/resources are introduced.

An isolated read-only reviewer inspected source contracts and reproduced a lifecycle race with
an independent temporary Swift harness; the main implementation incorporated the finding. A
separate delegate authored/audited policy, historical baseline and CI files. Classic background
workers failed to start (`Worker runner unavailable`); they supplied no review or changes.

## Findings resolved

1. Async error scope pop originally discarded a scope while an allocation was outstanding.
   The failing regression was reproduced before the fix. Operations now retain original scope
   tokens, scope pending counts keep detached scopes alive, and pop waits for their completion.
   Tests cover first matching/full scopes, LIFO, async pop and errors arriving after a new scope.
2. Backend deviceLost errors originally did not transition the core device to terminal loss.
   Allocation/submission catches now invoke the common terminal-loss path; subsequent allocation
   and submissions reject. Tests cover both backend failure entry points and allocation cleanup.
3. Independent review reproduced overlapping loss/close returning while the second backend
   destruction was still blocked. Counting individual destroys was insufficient: pending count
   could reach zero between them. The entire loss teardown batch is now counted before its first
   await, in addition to per-operation counts. A gated two-destroy regression failed in BOTH
   loss+close and close+close cases with the pre-fix code, then was restored to the fixed version
   for final verification. It asserts the actual number of completed destroys observed when
   close returns, not a timing-based performance threshold.

## Contracts accepted

- API/Core/backend descriptors have no raw native/JS handle. No unsafe Sendable/actor escape.
- Buffer validation precedes backend entry; unknown/foreign/retired/pending resources are rejected
  for use. IDs are namespaces, not unforgeable security tokens. Serial numbers are never recycled.
- Resource-use acceptance validates the whole batch before pinning; duplicate IDs count once.
  Destroy invalidates immediately but defers physical cleanup until accepted work retires.
- Queue task dependencies serialize backend completion and retire pins on both success/failure.
  Ordinary submission failure does not poison subsequent queue work. Device loss is terminal.
- Owners explicitly await close. No hidden deinit task, premature cancellation release or silent
  default backend. Accepted work is not cancelled when a ticket waiter is cancelled.
- Compiler raw IR, validation witness and codegen are separate. Witness initializer is private;
  decoded/transformed IR cannot retain a proof without revalidation. Implementors are trusted
  and must keep IR immutable. There is no claimed protection against malicious injected validators.
- Text/SPIR-V container mismatch is rejected; semantic shader validation is not implemented here.
  Default Metal factory/compiler correctly throw unsupported. Test IR/artifacts are clearly synthetic.
- Import/module/dependency policy guards use a real dumped package and scanned source tree.
  CI requires a nonzero Swift Testing summary rather than accepting XCTest's compatibility zero.

## Compile-time checks

`scripts/verify-swift-contracts.py` typechecks one positive value-semantic IR example, and requires
compiler rejection for forged ValidatedModule construction, raw IR at a generator, non-Sendable IR,
mutable Sendable backend and synchronous cross-actor state access. It uses already built modules
and Swift 6 complete concurrency; expected compiler failures are test success, not ignored errors.

## Deliberate limits and follow-up

The APIs are foundation contracts, not the final WebGPU surface. Only buffer descriptor/lifetime
and a small feature/limit model exist. There is no mapping, texture/pipeline/command execution,
full typed IR, production frontend/codegen/validator, CLI, binary compatibility or GPU/browser test.
Browser executor confinement/JS interop, rich compiler options and final IR schemas require later
implementation and ADRs. Backend calls must eventually return on loss; recursive close from a
backend callback is forbidden because it would await its own retirement. Ticket wait cancellation
is cooperative, not prompt. CI YAML has not been executed on hosted runners. Swift 6.2.x is not
locally installed; tools-version6.2/language6 do not prove a 6.2 compiler run.

Exact results and evidence: [foundation verification](foundation-verification.md).

## SAGA-4 addendum — 2026-11-06

The foundation review above is historical, not a statement that Metal is still a stub.
A focused independent read-only review of the new command/lifetime/Metal code found no
confirmed memory-safety or ordering blocker for the diagnostic path. It identified potential
pipeline registry growth on retained backends and missing executable coverage of timeout
quarantine. The implementation now caches one native diagnostic pipeline, reuses an owner
handle and removes owner handles on Core.close; a repeated create/close test covers this.
A callback-gated real Metal test now forces a terminal timeout, verifies logical close removes
owner registries while two native buffers remain in an in-flight batch, then releases the
callback and verifies actual retirement. This is controlled callback delay, not an actual
hung-driver/device-removal experiment. New public command and teardown contracts are recorded
in [backend ADR](adr/select-initial-backends.md); [current verification](metal-verification.md).
