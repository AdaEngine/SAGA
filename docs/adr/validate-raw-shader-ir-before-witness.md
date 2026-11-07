# Validate raw shader IR before issuing a witness

- Status: accepted for incremental implementation; NOT full SAGA-5 or Naga acceptance
- Date: 2026-11-06
- Extends [shared contract](design-wgsl-frontend.md) and [full Naga scope](port-full-naga-on-swift.md); neither is superseded.

## Context

Three independently implemented parsers can produce malformed raw IR. Successful syntax/name resolution does not prove type correctness, address-space access, stage compatibility, scoped evaluation, capability availability or uniform control flow. A frontend-supplied bool or required-capability mask must not grant validation. The independent Swift implementation is incomplete; unsupported semantics must remain errors.

## Decision

`IndependentShaderCompiler.parseAndValidate` dispatches WGSL/GLSL/SPIR-V and invokes the concrete `ShaderSemanticValidator` before `ValidatedModule.checking` returns its immutable witness. The generic foundation validator SPI remains trusted extensibility, not an untrusted input hook or sandbox. The witness constructor remains private, transforms/decoding still produce raw IR requiring revalidation.

The concrete checker recomputes supported expression result types, checks prior handles and function order, core type layouts and storage access, resource/interface bindings, explicit emit/effect-result producers, statement ownership, return paths and transitive stage/uniformity restrictions. It never uses arbitrary frontend metadata as proof. Validation state and budgets belong to each invocation, no mutable Sendable class/shared counter. Recursion, names, nodes and work are bounded; raw malformed handles/cycles must throw, not trap.

Unavailable rule families reject with `unsupportedConstruct`. Current precise families and gaps: [semantic profile](../semantic-validation-coverage.json). In particular aggregate uniform/IO rules, extended stages/resources/atomics/subgroups, full constant/override evaluation, diagnostic filters and precise all-path uniformity remain incomplete. Conservative rejections are not parity: loops are treated as potentially divergent and only a subset of type equivalence/overloads is accepted. No new code generation path consumes unchecked modules.

## Alternatives

- Trust parsed IR or frontend capability declarations: rejected; raw IR is public and can be constructed/decoded without a parser.
- Claim a permissive unsupported pass until all rules exist: rejected; would issue an unjustified witness.
- Disable generic foundation protocols: unnecessary breakage; keep trusted SPI and use concrete built-in validator for public dispatch.

## Consequences and verification

Useful core programs can be checked now without shrinking the accepted full-Naga goal. Unsupported valid programs remain rejected and must be implemented/tested before full SAGA-5 completion. Parser tests, raw-IR semantic tests, integration tests, real configured upstream fixture expectations, native/Wasm compile and GPU/CTS evidence are distinct gates.

Current [verification report](../frontend-verification.md) and [machine results](../frontend-results.json) preserve actual tests and measured unsupported corpus cases. No output codegen or runnable application target was added; no commits/publication/Rust writes or forbidden dependencies.
