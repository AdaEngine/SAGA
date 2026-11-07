import SAGACompiler
import SAGAFrontends
import SAGAShaderIR
import Testing

@Suite struct FrontendIntegrationTests {
  // Independent literal fixtures, not upstream copies. These test end-to-end boundaries, not parity.
  @Test func threeActualFrontendModulesAcquireWitnessOnlyAfterBuiltInValidation() throws {
    let inputs: [ShaderInput] = [
      .wgsl("@compute @workgroup_size(1) fn main() {}"),
      .glsl("#version 450\nvoid main(){}", stage: .compute, defines: [:]),
      .spirv([
        0x0723_0203, 0x0001_0300, 0, 100, 0, 0x0002_0011, 1, 0x0003_000e, 0, 1, 0x0005_000f, 5, 10,
        0x6e69_616d, 0, 0x0006_0010, 10, 17, 1, 1, 1, 0x0002_0013, 1, 0x0003_0021, 2, 1,
        0x0005_0036, 1, 10, 0, 2, 0x0002_00f8, 11, 0x0001_00fd, 0x0001_0038,
      ]),
    ]
    for input in inputs {
      let checked = try IndependentShaderCompiler().parseAndValidate(input)
      #expect(checked.module.entryPoints.count == 1)
      #expect(checked.module.entryPoints[0].stage == .compute)
    }
  }
  @Test func parserSuccessDoesNotBypassSemanticValidation() throws {
    // Workgroup size exceeds pinned validator limit; syntax is valid.
    let source = "@compute @workgroup_size(20000) fn main() {}"
    let raw = try WGSLFrontend().parse(.wgsl(source))
    #expect(raw.entryPoints.count == 1)
    #expect(throws: ShaderFailure.self) {
      try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
    }
  }
  @Test func invalidInputAndBudgetNeverReturnValidatedWitness() throws {
    let inputs: [ShaderInput] = [
      .wgsl("fn main( {"), .glsl("#version 450\nvoid main(}", stage: .compute, defines: [:]),
      .spirv([0x0723_0203]),
    ]
    for input in inputs {
      #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(input)
      }
    }
    #expect(throws: ShaderFailure.self) {
      try IndependentShaderCompiler().parseAndValidate(
        .wgsl("fn f() {}"), limits: ShaderLimits(work: 1))
    }
  }
}

extension FrontendIntegrationTests {
  @Test func uniformAggregateLayoutIsCheckedAfterWGSLParsing() throws {
    let valid =
      "struct U { value: vec4<f32>, } @group(0) @binding(0) var<uniform> u: U; @compute @workgroup_size(1) fn main() {}"
    let checked = try IndependentShaderCompiler().parseAndValidate(.wgsl(valid))
    #expect(checked.module.globalVariables.count == 1)
    let invalid =
      "struct Inner { a: f32, } struct Outer { a: Inner, b: f32, } @group(0) @binding(0) var<uniform> u: Outer; @compute @workgroup_size(1) fn main() {}"
    let raw = try WGSLFrontend().parse(.wgsl(invalid))
    #expect(raw.globalVariables.count == 1)
    do {
      _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(invalid))
      Issue.record("Invalid nested uniform layout acquired a witness")
    } catch ShaderFailure.diagnostics(let ds) { #expect(ds.first?.code == .invalidType) }
  }
}

extension FrontendIntegrationTests {
  @Test func wgslAtomicAndRuntimeArrayLegalityReachesConcreteValidator() throws {
    let good = [
      "struct S { counter: atomic<u32>, } var<workgroup> s: S;",
      "struct S { counter: atomic<u32>, } @group(0) @binding(0) var<storage, read_write> s: S;",
      "@group(0) @binding(0) var<storage> s: array<u32>;",
      "struct S { bits: array<bool, 2>, } var<private> s: S;",
    ]
    for source in good { _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
    let bad = [
      "struct S { counter: atomic<u32>, } var<private> s: S;",
      "struct S { counter: atomic<u32>, } @group(0) @binding(0) var<storage> s: S;",
      "struct S { bits: array<bool, 2>, } @group(0) @binding(0) var<storage> s: S;",
    ]
    // Nested unsized types now fail earlier in the frontend; retain exact rejection and raw-IR validator tests.
    let nested = "struct R { items: array<u32>, } struct S { r: R, } @group(0) @binding(0) var<storage> s: S;"
    do { _ = try WGSLFrontend().parse(.wgsl(nested)); Issue.record("Nested unsized type accepted") }
    catch ShaderFailure.diagnostics(let ds) {
      #expect(ds.first?.code == .invalidType)
      #expect(ds.first?.span == (try SourceSpan(startUTF8: 43, endUTF8: 44)))
    }
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(nested)) }
    for source in bad {
      let parsed = try WGSLFrontend().parse(.wgsl(source))
      #expect(parsed.globalVariables.count == 1)
      #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
      }
    }
  }
}

extension FrontendIntegrationTests {
  @Test func wgslStructEntryPointInputsUseCheckedMemberBindings() throws {
    let source =
      "struct In { @builtin(local_invocation_index) index: u32, } @compute @workgroup_size(1) fn main(input: In) {}"
    let checked = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
    #expect(checked.module.entryPoints.count == 1)
    let collision =
      "struct In { @location(0) a: f32, @location(0) b: f32, } @fragment fn main(input: In) {}"
    _ = try WGSLFrontend().parse(.wgsl(collision))
    do {
      _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(collision))
      Issue.record("IO collision acquired witness")
    } catch ShaderFailure.diagnostics(let ds) {
      #expect(ds.first?.code == .invalidBinding)
      #expect(ds.first?.span != nil)
    }
  }
}

extension FrontendIntegrationTests {
    @Test func runtimeArrayStorageStructGetsWitnessWithMinimumSpan() throws {
        for fields in ["items: array<u32>,", "prefix: u32, items: array<vec3<f32>>,"] {
            let source = "struct Buffer { \(fields) } @group(0) @binding(0) var<storage> data: Buffer;"
            let checked = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
            #expect(checked.module.globalVariables.count == 1)
        }
    }
}

extension FrontendIntegrationTests {
    @Test func boolWGSLLayoutWorksOnlyInNonHostAddressSpaces() throws {
        let declaration = "struct S { first: bool, flags: array<vec3<bool>, 2>, last: bool, } "
        for suffix in ["var<private> value: S;", "var<workgroup> value: S;", "fn f() { var value: S; }"] {
            let checked = try IndependentShaderCompiler().parseAndValidate(.wgsl(declaration + suffix))
            let type = try #require(checked.module.types.values.first { $0.name == "S" })
            guard case .structure(let fields, let span) = type.kind else { continue }
            #expect(fields.map(\.offset) == [0, 16, 48]); #expect(span == 64)
        }
        for space in ["uniform", "storage, read", "storage, read_write"] {
            let source = declaration + "@group(0) @binding(0) var<\(space)> value: S;"
            _ = try WGSLFrontend().parse(.wgsl(source))
            do { _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source)); Issue.record("Host bool accepted") }
            catch ShaderFailure.diagnostics(let ds) { #expect(ds.first?.code == .invalidAddressSpace) }
        }
    }
}

extension FrontendIntegrationTests {
    @Test func explicitUniformMemberLayoutAcquiresWitness() throws {
        let source = "struct Inner { x: f32, } struct S { @size(16) a: Inner, b: f32, } @group(0) @binding(0) var<uniform> value: S;"
        let checked = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        #expect(checked.module.globalVariables.count == 1)
    }
}

extension FrontendIntegrationTests {
    @Test func constantAttributesReachValidatedUniformLayout() throws {
        let source = "const A = 8u; const B: u32 = A * 2u; struct S { @align(B) @size(i32(B) + 16i) x: f32, } @group(1 - 1) @binding(u32(2)) var<uniform> value: S;"
        let checked = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        #expect(checked.module.globalVariables.values[0].binding == .init(group: 0, binding: 2))
    }
}

extension FrontendIntegrationTests {
    @Test func abstractModuleValuesMaterializePerContextNotGloballyToI32() throws {
        for source in [
            "const A = 16; struct S { @align(A) x: f32, } @group(0) @binding(0) var<uniform> s: S;",
            "fn f() -> u32 { return A; } const A = 4294967295;",
            "const A = 16; fn f() -> i32 { return A; } fn g() -> u32 { return A; } fn h() -> f32 { return A; }",
            "const A = 16; fn f() -> u32 { return A + 1; }",
            "const A = 16; fn f() { var x = A; var y: u32 = A; }"
        ] {
            let m = try IndependentShaderCompiler().parseAndValidate(.wgsl(source)).module
            #expect(!m.types.values.contains { if case .scalar(let s) = $0.kind {s.kind == .abstractInt || s.kind == .abstractFloat} else {false} })
            #expect(!m.constants.values.contains { $0.name == "A" })
        }
        // Creation-time arithmetic may exceed i32 without needing any runtime materialization.
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl("const Big = 4294967296; const A = Big / 268435456; struct S { @align(A) x: f32, }"))
        for source in [
            "const A = 4294967295; fn f() -> i32 { return A; }",
            "const A = -1; fn f() -> u32 { return A; }",
            "const A = 4294967296; fn f() { var x = A; }",
            "const A = 1u; fn f() -> i32 { return A; }"
        ] { #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) } }
    }
}

extension FrontendIntegrationTests {
    @Test func forwardFunctionsGlobalsAndConstantDiamondPreserveEffects() throws {
        let source = "fn entry() { sink(); value = helper(A); } const A = Left + Right; const Left = Base; const Right = Base; const Base = 2; var<private> value: u32; fn helper(x: u32) -> u32 { return x + A; } fn sink() {}"
        let checked = try IndependentShaderCompiler().parseAndValidate(.wgsl(source)).module
        let entry = try #require(checked.functions.values.first { $0.name == "entry" })
        let calls = entry.statements.values.compactMap { s -> String? in
            if case .call(let f,_,_) = s.kind { return checked.functions.values[f.index].name }; return nil
        }
        #expect(calls == ["sink", "helper"])
        let names = checked.functions.values.compactMap(\.name)
        #expect(names == ["sink", "helper", "entry"])
        for f in checked.functions.values {
            if let name = f.name {
                let offset = try #require(source.range(of: "fn " + name))
                let start = source[..<offset.lowerBound].utf8.count + 3
                #expect(f.location == .source(try SourceSpan(startUTF8: start, endUTF8: start + name.utf8.count)))
            }
        }
        for source in ["fn a() { b(); } fn b() { a(); }", "fn a() { a(); }", "fn a() -> u32 { return b(); } fn b() -> u32 { return a(); }"] {
            #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
        }
    }
    @Test func abstractCompactionPreservesConcreteTypesAndSourceOrder() throws {
        let source = "const A = 16; const C: u32 = 3u; struct S { x: u32, } var<private> value: S; fn f(x: u32) -> u32 { var a: array<u32, 2>; a[0] = A; if x > C { value.x = a[0]; } return A; }"
        let m = try IndependentShaderCompiler().parseAndValidate(.wgsl(source)).module
        #expect(m.constants.values.map(\.name) == ["C"])
        #expect(m.globalVariables.values.map(\.name) == ["value"])
        let f = try #require(m.functions.values.first)
        #expect(f.statements.values.contains { if case .ifCondition = $0.kind { true } else { false } })
        #expect(f.expressions.values.allSatisfy { e in e.type.index < m.types.count })
    }
}

extension FrontendIntegrationTests {
    @Test func abstractIntegerOperatorsAndLocalShadowingUseContext() throws {
        let valid = [
            "const A = 16; fn f() -> u32 { return ~(~A); }",
            "const A = 16; fn f() -> i32 { return -A; }",
            "const A = 16; fn f() -> u32 { let A = 32u; return A; }"
        ]
        for source in valid { _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
        for source in ["const A = 4294967295; fn f() -> i32 { return A; }", "const A = 16; fn f() -> u32 { return A + 1i; }"] {
            do { _ = try WGSLFrontend().parse(.wgsl(source)); Issue.record("Unrepresentable/mixed context accepted") }
            catch ShaderFailure.diagnostics(let ds) { #expect(ds.first?.code == .typeMismatch); #expect(ds.first?.span != nil) }
        }
    }
}

extension FrontendIntegrationTests {
    @Test func abstractConstantsUseConcreteBuiltinConsensus() throws {
        for source in [
            "const A = 16; fn f() -> u32 { return min(A, 2u); }",
            "const A = 4294967295; fn f() -> u32 { return max(A, 2u); }",
            "const A = 16; fn f() -> f32 { return max(A, 2.0f); }",
            "const A = 16; fn f() -> i32 { return clamp(A, 1, 32); }"
        ] { _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
        for source in [
            "const A = 4294967296; fn f() -> u32 { return min(A, 1u); }",
            "const A = 1; fn f() -> u32 { return min(1i, 2u); }"
        ] { #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) } }
    }
}

extension FrontendIntegrationTests {
    @Test func abstractFloatDeclarationUsesSameLoweringBoundary() throws {
        for source in [
            "fn f() -> f32 { return A; } const A = 1.5;",
            "const A = 1.5; const B = A; fn f() -> f32 { return B; }",
            "const A = -1.5; fn f() -> f32 { return A; }"
        ] {
            let m = try IndependentShaderCompiler().parseAndValidate(.wgsl(source)).module
            #expect(!m.types.values.contains { if case .scalar(let s) = $0.kind {s.kind == .abstractFloat} else {false} })
            #expect(m.constants.count == 0)
        }
        for source in ["const A = 1.5; fn f() -> i32 { return A; }", "const A = 1e100; fn f() -> f32 { return A; }", "const A=1.5; struct S { @align(A) x:f32, }"] {
            #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
        }
    }
}

extension FrontendIntegrationTests {
  @Test func moduleAndFunctionAssertionsAreCompileTimeOnly() throws {
    let source="const A = 1 < 2; const_assert A; fn f() -> bool { const z = 2; const_assert z > 0; { const z = false; const_assert !z; } const_assert z == 2; return A; }"
    let m=try IndependentShaderCompiler().parseAndValidate(.wgsl(source)).module
    #expect(m.functions.count == 1)
    #expect(m.types.values.allSatisfy { if case .scalar(let s) = $0.kind {s.kind != .abstractInt} else {true} })
    for s in ["fn f() { let x = 1; const_assert x == 1; }", "fn f(x: bool) { const_assert true || x; }", "fn f() { if false { const_assert false; } }", "fn f() { const x = true; { let x = false; const_assert x; } }"] {
      #expect(throws:ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(s)) }
    }
  }
}

extension FrontendIntegrationTests {
  @Test func localConstScopesInLoopsAndForwardModulesStayIsolated() throws {
    for source in [
      "const N = 2; fn f() { for(var i=0; i<1; i++) { const z=1; const_assert z<N; } }",
      "fn f() { const N = 0; const_assert A; } const A = N == 2; const N = 2;",
      "fn f(flag: bool) { loop { const z=true; const_assert z; if flag { break; } continuing { const_assert true; } } }"
    ] { _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
    for source in ["fn f() { const z=true; const z=false; }", "fn f() { const_assert z; const z=true; }", "fn f() { { const z=true; } const_assert z; }"] {
      #expect(throws:ShaderFailure.self) {try IndependentShaderCompiler().parseAndValidate(.wgsl(source))}
    }
    let m=try IndependentShaderCompiler().parseAndValidate(.wgsl("fn f() { const a=1; const_assert a==1; }")).module
    #expect(m.functions.values[0].statements.count == 1) // implicit return only
  }
}

extension FrontendIntegrationTests {
  @Test func assertionsAfterRuntimeTerminatorsAreNotSilentlySkipped() throws {
    for source in ["fn f() { return; const_assert true; }", "fn f() { loop { break; continuing { const_assert true; } } }"] {
      _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
    }
    for source in ["fn f() { return; const_assert false; }", "fn f() { loop { break; continuing { const_assert false; } } }"] {
      do {_ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source));Issue.record("Unevaluated false assertion")}
      catch ShaderFailure.diagnostics(let ds) {#expect(ds.first?.message == "const_assert evaluated to false")}
    }
  }
}

extension FrontendIntegrationTests {
  @Test func booleanConstValuesCanFeedIntegerAttributesViaExplicitConversion() throws {
    let source="const Flag = 1 < 2; struct S { @align(u32(Flag) * 16u) x: f32, } @group(0) @binding(0) var<uniform> s: S;"
    _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
    for e in ["true", "Flag && true"] {
      #expect(throws:ShaderFailure.self) {try IndependentShaderCompiler().parseAndValidate(.wgsl("const Flag=true; struct S { @align(\(e)) x: f32, }"))}
    }
    _ = try IndependentShaderCompiler().parseAndValidate(.wgsl("fn f() { const Count=2u; var data: array<u32, Count>; const_assert Count==2u; }"))
  }
}

extension FrontendIntegrationTests {
  @Test func moduleAndLocalFloatConstPreserveForwardScopesAndConcreteIRBoundary() throws {
    for source in [
      "const_assert A < 2.0; const A = 1.0 + 0.5; fn f()->f32{return A;}",
      "fn f(){const x=1.0; const_assert x<2.0; {const x:f32=3.0; const_assert x>2.0f;} const_assert x==1.0;}",
      "const X:f32 = 16777216.0f + 1.0f; const_assert X==16777216.0f;"
    ] {
      let m=try IndependentShaderCompiler().parseAndValidate(.wgsl(source)).module
      #expect(!m.types.values.contains {if case .scalar(let s)=$0.kind {s.kind == .abstractFloat} else {false}})
    }
    for source in ["fn f(x:f32){const_assert true || (x==0.0f);}", "fn f(){let x=1.0; const_assert x<2.0;}", "const A=1.0; struct S{@align(A)x:f32,}", "const A=1.0; @compute @workgroup_size(A) fn f(){}"] {
      #expect(throws:ShaderFailure.self) {try IndependentShaderCompiler().parseAndValidate(.wgsl(source))}
    }
  }
}

extension FrontendIntegrationTests {
  @Test func scalarAliasFloatConstructorsValidateThroughUnchangedWitness() throws {
    let source="const_assert B(1); alias B=bool; alias F=Real; alias Real=f32; const Value:F=F(1.5)+F(0.5); fn f()->F { const local:F=Value*2.0; const_assert local==4.0f; return local; }"
    let module=try IndependentShaderCompiler().parseAndValidate(.wgsl(source)).module
    let literalValues=module.functions.values.flatMap { $0.expressions.values }.compactMap { e -> Float? in if case .literal(.f32(let v))=e.kind {return v};return nil }
    #expect(literalValues.contains(4.0))
    for source in ["const Value=1e100; fn f()->f32 { return Value; }", "const Value:f32=1.5; fn f()->i32 { return Value; }", "alias B=bool; fn f(B:bool){const_assert B(1);}"] {
      #expect(throws:ShaderFailure.self) {try IndependentShaderCompiler().parseAndValidate(.wgsl(source))}
    }
  }
}

extension FrontendIntegrationTests {
 @Test func vectorConstantValuesLowerThroughExistingValidatedContracts() throws {
  for source in [
   "const V=vec2<bool>(true,true); const_assert all(V); fn f()->vec2<bool>{return V;}",
   "alias V=vec3<f32>; fn f()->V {const x:V=V(1.5); const_assert all(x==V(1.5)); return x;}",
   "const V=vec2(4294967295,2); fn f()->vec2<u32>{return V;}"
  ] {_ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))}
 }
}

extension FrontendIntegrationTests {
 @Test func vectorContextChecksPerComponentAndFloatBuiltinsEndToEnd() throws {
  for source in [
   "const V=vec2(1.5,2.5); fn f()->vec2<f32>{return V;}",
   "const V=vec4(1,2,3,4); fn f(){const x:vec4<u32> = V; const_assert all(x==vec4<u32>(1u,2u,3u,4u));}",
   "alias V=vec2<f32>; const Value=V(sqrt(0x1p2f),inverseSqrt(4.0f)); fn f()->V{return Value;}"
  ] {_ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))}
  for source in [
   "const V=vec2(4294967295,2); fn f()->vec2<i32>{return V;}",
   "const V=vec2(-1,2); fn f()->vec2<u32>{return V;}",
   "const V=vec2(3.4028235e38); fn f()->vec2<f32>{return V;}",
   "const V:vec2<i32> = vec2<u32>(1u,2u);",
   "const V=vec2<bool>(true); @compute @workgroup_size(V) fn f(){}"
  ] {#expect(throws:ShaderFailure.self) {try IndependentShaderCompiler().parseAndValidate(.wgsl(source))}}
 }
}

extension FrontendIntegrationTests {
 @Test func vectorConstOperationsMaterializeThroughExistingWitness() throws {
  for source in [
   "const V=vec2(1,2)+3; fn f()->vec2<u32>{return V;}",
   "const V=vec4(1,2,3,4); fn f()->u32{return V.wzyx[0];}",
   "fn f()->vec2<f32>{const V=select(vec2(1.5,2.5),vec2<f32>(3.5f),vec2<bool>(true,false)); const_assert V[1]==2.5f; return V;}"
  ] {_ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))}
 }
}

extension FrontendIntegrationTests {
 @Test func constantVectorRuntimeAccessCannotAcquireWitnessOutOfBounds() throws {
  for source in ["const V=vec2<bool>(true); fn f()->bool{return V[2];}", "const V=vec2(1,2); fn f()->i32{return V[-1];}", "const V=vec2<u32>(1u); fn f()->u32{return V[4294967295u];}"] {
   #expect(throws:ShaderFailure.self) {try IndependentShaderCompiler().parseAndValidate(.wgsl(source))}
  }
  _ = try IndependentShaderCompiler().parseAndValidate(.wgsl("const V=vec2(4294967295,2); fn f()->u32{return V.xy[0];}"))
 }
}

extension FrontendIntegrationTests {
 @Test func matrixCompileTimeValuesUseCheckedRuntimeBoundary() throws {
  let source="const_assert all(M[0]==vec2<f32>(1.0,2.0)); alias T=mat2x2<f32>; const M=T(1.0,2.0,3.0,4.0); fn f(){const_assert M[1].y==4.0f;}"
  _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
  for source in ["const M=mat2x2<f32>(); fn f()->mat2x2<f32>{return M;}", "fn f(){const M=mat2x2<f32>();}"] {
   _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
  }
  #expect(throws:ShaderFailure.self) {try IndependentShaderCompiler().parseAndValidate(.wgsl("const M=mat2x2<f32>(); const_assert true || all(vec2(M).x==vec2(0.0));"))}
 }
}

extension FrontendIntegrationTests {
 @Test func matrixValuesNowLowerToCheckedRuntimeColumnCompose() throws {
  for source in [
   "const M=mat2x2<f32>(1.0,2.0,3.0,4.0); fn f()->mat2x2<f32>{return M;}",
   "const M=mat2x2(1,2,3,4); fn f()->mat2x2<f32>{return M;}",
   "alias M=mat3x2<f32>; fn f()->M{const m:M=M(vec2<f32>(1.0),vec2<f32>(2.0),vec2<f32>(3.0)); const_assert m[2].y==3.0f; return m;}",
   "const M=mat2x2<f32>(); fn f(){var m:mat2x2<f32> = M;}"
  ] {
   let m=try IndependentShaderCompiler().parseAndValidate(.wgsl(source)).module
   #expect(m.functions.values.contains { $0.expressions.values.contains {if case .compose = $0.kind {true} else {false}} })
  }
  for source in ["const M=mat2x2(1e100,0.0,0.0,1.0); fn f()->mat2x2<f32>{return M;}", "const M=mat2x2(1,2,3,4); fn f()->mat3x2<f32>{return M;}"] {
   #expect(throws:ShaderFailure.self) {try IndependentShaderCompiler().parseAndValidate(.wgsl(source))}
  }
 }
}

extension FrontendIntegrationTests {
 @Test func runtimeMatrixConstructorsUseColumnMajorShapeAndContext() throws {
  for source in [
    "fn f(x:f32)->mat2x2<f32>{return mat2x2<f32>(x,2.0,3.0,4.0);}",
    "fn f(x:vec3<f32>)->mat2x3<f32>{return mat2x3<f32>(x,vec3<f32>(1.0));}",
    "alias M=mat2x2<f32>; fn f(x:M)->M{return M(x);}",
    "fn f(x:vec2<f32>)->vec4<f32>{return vec4<f32>(x,1.0,2.0);}"
  ] { _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
  for source in [
    "fn f(x:vec3<f32>)->mat2x2<f32>{return mat2x2<f32>(x,x);}",
    "fn f()->mat2x2<f32>{return mat2x2<f32>(1u,2u,3u,4u);}",
    "fn f(x:mat3x2<f32>)->mat2x2<f32>{return mat2x2<f32>(x);}",
    "fn f(x:vec3<f32>)->vec4<f32>{return vec4<f32>(x,x);}",
    "fn f()->vec2<bool>{return vec2<bool>(1,2);}"
  ] { #expect(throws:ShaderFailure.self) {try IndependentShaderCompiler().parseAndValidate(.wgsl(source))} }
 }
}

extension FrontendIntegrationTests {
 @Test func runtimeMatrixColumnIndexMaterializesAbstractScalarWithoutForcingWholeMatrix() throws {
  for expression in ["M[0][0]", "M[1].x", "M[0].yx[1]"] {
   let m=try IndependentShaderCompiler().parseAndValidate(.wgsl("const M=mat2x2(1,2,3,4); fn f()->f32{return \(expression);}")).module
   #expect(m.types.values.allSatisfy {t in
     switch t.kind {case .scalar(let s),.vector(_,let s),.matrix(_,_,let s): return s.kind != .abstractFloat && s.kind != .abstractInt;default:return true}
   })
  }
 }
}
