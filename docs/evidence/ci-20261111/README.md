# CI repair — 2026-11-11

Hosted failure: https://github.com/AdaEngine/SAGA/actions/runs/43061979046
All jobs rejected Swift6.2.4/6.3.2 in setup-swift v2's static release list, before compilation.

## Local fixes
- Commit-pinned Swiftly-based setup-swift v3; exact Swift and Wasm SDK versions retained.
- Explicit Python3.12 (tomllib requires >=3.11).
- Runtime policy updated for 18 existing targets, narrow platform links/imports and hash-pinned native ABI exceptions, with rejection tests; frozen baseline retained.
- Validator expression rules split into six helpers, preserving case bodies. Debug outer frame reduced from338416 to7552 bytes; SIGBUS reproducers pass.
- WASI-only exhaustive switch fixed by explicitly rejecting unsupported renderMultiIndirect.

## Checks
Strict native builds succeeded on Swift6.3.2 and Apple6.3.3. Six compile-time contract checks pass. Focused validator/reproducer41 tests pass (worker report).
Docs/reference80 tests pass, zero skips: [log](docs-reference.log).
Four configured Wasm targets compile; matching6.3.2 reactor builds and links. No live browser claim.
actionlint1.7.12 passes with shellcheck disabled/unavailable; diff whitespace check passes.

Full strict6.3.2 suite completes but FAILS:2037 tests,3 issues. [Log](native-tests.log).
- Access-path GPU test both variants: expected[0,10,20,99,40], actual[0,10,20,30,99].
- Structured pre-test loop GPU test: expected22, actual0.

Original tests and expectations preserved. Standalone emitted MSL reproduces access-path error; volatile device pointer changes output to expected, but is not a validated general fix. [MSL](saga-path-repro.metal), [probe](saga-path-repro.swift). Loop [MSL](saga-loop-original.metal), [probe](saga-loop-probe.swift). No SPIR-V frontend patch retained.

No commits/pushes/hosted reruns. Swift6.2.4 not locally verified. CI is NOT declared green; remaining GPU regressions need a validated Metal-side fix. Existing D3D12 global-debug concurrency risk is documented separately, not fixed.
