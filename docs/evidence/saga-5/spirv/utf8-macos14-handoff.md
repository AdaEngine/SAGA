# Narrow UTF-8 macOS14 compatibility verification

Current production SPIRVFrontend.swift:100-101 already contains `String(decoding: bytes, as: UTF8.self)` plus exact `Array(s.utf8) == bytes` rejection and zero-padding check. The unavailable `String(validating:as:)` initializer is absent. No production edits were needed in this follow-up. Deployment target unchanged; no Foundation added; conflicting lowerers left untouched.

Extracted the exact production `string` method into utf8-macos14-verification.swift with only diagnostic stubs. Compiled with `/usr/bin/swiftc -target arm64-apple-macosx14.0 -swift-version 6 -warnings-as-errors -strict-concurrency=complete`, then executed on current host OS. 14 checks PASS: ASCII, empty, multibyte Unicode, legitimate U+FFFD,4096-byte name; invalid overlong/surrogate/out-of-range/truncated/stray continuation UTF8, nonzero padding and missing terminator rejected. This tests the exact method but not full decoder integration or actual macOS14 runtime.

Budget: decoder charges actual input words and instruction words before invoking bounded-linear append/decode/roundtrip; never ID-bound allocation. Method is internal, not a separate externally budgeted API. Exact extraction/source fingerprints in utf8-macos14-verification.json.

Initial isolated harness compile failed Swift type-check complexity in its word-packing expression; split into explicit UInt32 temporaries, then strict compile+14 checks passed. No checks weakened.

Final integrated strict SAGACompiler build attempted and failed existing duplicate SPVBlockRecord/SPVValue/SPVFunctionLowerer declarations in FunctionLowering.swift versus SPIRVFunction.swift. No UTF8 availability error shown; no full build success claimed. Full output archived utf8-followup-integration-build.log. Reconcile sole ownership before further source edits. SAGA-15/full frontend DoD remains BLOCKED. No tasks/workflow/commits/publication/dependencies changed.
