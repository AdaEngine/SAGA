# Matrix arithmetic and configured frontend corpus

Full SAGA-5 remains incomplete. This is an evidence handoff, not acceptance.

## Reproduce

From SAGA root on the current macOS build host (Python 3.11+ with tomllib):

```sh
python3 docs/tools/verify_configured_frontend_corpus.py --reference /Users/vlad-prusakov/Developer/wgpu --output docs/evidence/saga-5/matrix-and-configured-corpus
/usr/bin/swift test --scratch-path .build-validation -Xswiftc -warnings-as-errors -Xswiftc -strict-concurrency=complete
```

The wrapper checks immutable pinned bytes, builds strict native compiler objects, uses SwiftPM output-file-map membership (not stale wildcard objects), executes configured rows, and validates source/config/options provenance. Runner infrastructure errors are nonzero exits. Its successful execution does not mean the shader expectations passed: inspect results.json.

270 inputs: 32 pass, 200 fail, 36 unsupported SPIR-V assembly, 2 rejection-only. This tests frontend snapshot acceptance and configured pre-compaction validation only, not full output/diagnostic/CTS parity. No shader bytes copied. See pinned oracle source hashes in plan.json.

Native 229 tests, strict build, matched Swift6.3.2 Wasm compile-only, docs66 checks pass. Shared IR/private witness/frozen baseline hashes preserved. Historical raw-boundary evidence is not overwritten.

The corpus exposed actual loop lowering errors. Authored regression tests first failed for shared/orphan blocks, then passed after preserving original child-block ownership and distinct continuing blocks. Validator graph rules were not weakened. Local constructor shadowing first returned a witness for invalid WGSL; it now rejects.

Remaining failures include directives/resources/f16/full builtin and aggregate support, GLSL profiles/preprocessing/types, SPIRV advanced families, break-if scope and integer fragment-output interface rules. Exact per-input diagnostics remain in results.json. No external tool/permission blocker is claimed for unfinished compiler implementation.
