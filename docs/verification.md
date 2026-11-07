# Documentation verification

SAGA-2 baseline acceptance, 2026-11-06. Run from the saga checkout:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 docs/tools/verify_docs.py
make -C docs test
```

[Machine report](verification-results.json) contains the actual documentation checks. [Source verification](source-verification.json) retains the earlier HTTP 200 observations for two GitHub commit endpoints and five immutable raw inputs. [Source hashes](source-hashes.json) retain snapshot fingerprints. **This acceptance reran local checks, not those network requests.** This is documentation validation, not compilation or GPU conformance.

The validator checks frozen inventory counts/IDs, honest not-started/not-run status, source anchors/tokens, planned CTS directories against the complete pinned tree, native tokens, deterministic extraction, retained source hashes, reference revision/paths, the frozen task DAG, local Markdown links and inventory anchors. Task dependency equality is against the captured baseline, not a live task API query. Upstream snapshot README links are not SAGA-authored local links. Living URLs are navigation aids; not all external anchors are HTTP-tested.

## Actual acceptance results

- `python3 docs/tools/verify_docs.py`: **2,499 checks passed**, 73 local links, 809 unique IDs, 10 DAG nodes / 16 edges.
- `make -C docs test` and direct Python unittest discovery: **8 tests passed**. Two added regression tests cover the scope guard (including negative runtime/dependency paths) and accepted-platform/open-compiler-gate metadata with all execution flags false.
- Initial acceptance: validator passed 2,490 checks; existing tests were **5/6**, failing because pre-existing `.sloppy/plans/wgpu-rust/` was treated as this task's unauthorized change. The guard now distinguishes project file types from change ownership. It permits retained planning artifacts, still rejects runtime/dependency files outside docs/plans, and the before/after fingerprint audit separately checks that this turn changes only `docs/`. No unrelated file was removed to make the test pass.
- Red step after adding the regressions: missing guard helper and missing decision metadata failed as expected, alongside the existing scope failure; the final checks above pass without removing source/hash/ID/link checks.
- Local wgpu HEAD is `a55352911bb26a1b33bc8d5fc03bdebb0a967f5a`, with clean porcelain. Reference paths, native feature tokens and retained source fingerprints were checked without Rust writes.
- No Swift build/tests, Rust tests, native shader compilation, GPU dispatch/readback, render/presentation/device-loss scenario or CTS execution was performed.

## SAGA-2 Definition of Done review

| Criterion | Evidence and qualification |
|---|---|
| Every major API/validation/shader subsystem has status and test source | `inventory.json` / `inventory.md` contain 750 review units; `conformance.md` maps major subsystems to planned CTS/Swift/GPU tests. All implementation/verification rows remain not-started/not-run. This is not an exhaustive atomic-assertion manifest. |
| Initial measurable denominator and open decisions | 139 IDL declarations + 150 WebGPU sections + 461 WGSL sections = 750 coarse review units; 59 native flags separate. Assertion and expanded CTS case denominators remain unknown. Compiler-format scope, native parity and detailed matrix stay open. |
| Sources, revisions, licenses and realistic stages | Immutable pins/snapshots/hashes and retained notices in `provenance.md` / `sources/`; SAGA-2..11 DAG and exit gates in `roadmap.md` / `roadmap.json`. Fixture-level licensing and project outbound license remain release follow-ups, not silently approved. |
| Independence and no unauthorized backend choice | Independence ADR preserved with dated addendum; separate `adr/use-metal-and-browser-webgpu.md` records the user's existing Apple/Metal + browser WebGPU choice. SAGA project Dawn/wgpu/Rust Naga dependencies remain forbidden; browser-owned internals are distinguished. Additional formats are not selected. |
| Documentation verification, not fabricated CTS results | Passing validator/docs tests and explicit false execution flags. Source existence/hashes are not spec/CTS alignment, Swift compatibility or GPU conformance. |

The documentation baseline meets SAGA-2's initial-inventory DoD, not SAGA-1's implementation/release DoD. Browser-backed web evidence must remain separate from SAGA native compiler/Metal evidence.

## Scope and remaining gates

Only `docs/` was edited; pre-existing `IDEA.md`, `.sloppy/plans/`, source snapshots and inventories are preserved. No Package.swift/runtime work, Rust checkout writes, commits or publication. There is no runnable SAGA target, so no Swift build or Play recipe is invented. `make -C docs test` is the documentation check target, not an application build. Swift 6.3.3/Xcode 26.6 remain task-supplied facts; Swift 6.2 compatibility is a SAGA-3 gate.

The accepted Apple/Metal + Swift web/browser WebGPU direction must not be asked again. The workflow human-input gate **WGSL→MSL versus all Naga formats remains unanswered**. Native-extension parity, exact OS/GPU/browser matrix, interop/CTS host, assertion refinement, expanded CTS enumeration, spec/CTS alignment, fixture licenses and outbound project license remain as documented in [conformance](conformance.md), [provenance](provenance.md) and [roadmap](roadmap.md). SAGA-1 stays open.
