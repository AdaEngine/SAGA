import Testing
import SAGAShaderIR
import SAGAFrontends

// Independently authored SAGA fixtures; no upstream shader bytes copied.
@Suite struct WGSLFrontendTests {
    @Test func computeLowersStorageEffectsAndTypedExpressions() throws {
        let m = try WGSLFrontend().parse(.wgsl("""
        @group(0) @binding(0) var<storage, read_write> values: array<u32>;
        @compute @workgroup_size(8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            values[i] = values[i] + 1u;
        }
        """))
        #expect(m.entryPoints.count == 1)
        #expect(m.entryPoints[0].workgroupSize.x == 8)
        #expect(m.globalVariables.count == 1)
        let fn = m.functions.values[0]
        #expect(fn.statements.values.contains { if case .store = $0.kind { true } else { false } })
        #expect(fn.statements.values.contains { if case .emit = $0.kind { true } else { false } })
    }
    @Test func functionCallsBranchesScopesAndReturns() throws {
        let m = try WGSLFrontend().parse(.wgsl("""
        fn twice(x: u32) -> u32 { return x * 2u; }
        @compute @workgroup_size(1) fn main() {
            var x: u32 = twice(3u);
            if x > 2u { x = twice(x); } else { x = 1u; }
        }
        """))
        #expect(m.functions.count == 2)
        #expect(m.functions.values[1].statements.values.contains { if case .call = $0.kind { true } else { false } })
    }
    @Test func rejectsNamesTypesAndMalformedSyntax() throws {
        for source in ["fn f() { missing = 1u; }", "fn f() { let a: bool = 1u; }", "fn f() { let a = ; }", "fn f() { let x = 1u; x = 2u; }", "fn f() { break; }", "fn f() { let x=1u; let x=2u; }"] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl(source)) }
        }
    }
    @Test func nestedCommentsUnicodeAndExactUTF8Diagnostics() throws {
        _ = try WGSLFrontend().parse(.wgsl("/* outer /* inner */ done */ fn café() { let α = 2u; }"))
        let source = "fn café() { let α = missing; }"
        do { _ = try WGSLFrontend().parse(.wgsl(source)); Issue.record("expected error") }
        catch ShaderFailure.diagnostics(let ds) {
            #expect(ds[0].code == .unresolvedName)
            let bytes = Array(source.utf8)
            let span = try #require(ds[0].span)
            #expect(String(decoding: bytes[span.startUTF8..<span.endUTF8], as: UTF8.self) == "missing")
        }
    }
    @Test func budgetsAndUnsupportedAreNotSuccess() throws {
        for limits in [try ShaderLimits(tokens: 1), try ShaderLimits(nodes: 1), try ShaderLimits(depth: 1), try ShaderLimits(work: 1)] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend(limits: limits).parse(.wgsl("fn main() { let a = ((1u+2u)); }")) }
        }
        #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl("enable made_up; fn f() {}")) }
        #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl("fn f() { textureStore(); }")) }
    }
    @Test func deterministicBoundedFuzz() throws {
        var seed: UInt64 = 0x53414741
        let alphabet = Array("fn abc0123u{}()[];:+-*/=@<>!\n".utf8)
        let frontend = WGSLFrontend(limits: try ShaderLimits(sourceBytes: 512, tokens: 256, nodes: 512, depth: 24, work: 10000))
        for _ in 0..<256 {
            var bytes: [UInt8] = []
            for _ in 0..<64 { seed = seed &* 6364136223846793005 &+ 1; bytes.append(alphabet[Int(seed >> 32) % alphabet.count]) }
            let input = ShaderInput.wgsl(String(decoding: bytes, as: UTF8.self))
            let first = Result { try frontend.parse(input) }
            let second = Result { try frontend.parse(input) }
            switch (first, second) {
            case (.success(let a), .success(let b)): #expect(a == b)
            case (.failure(let a as ShaderFailure), .failure(let b as ShaderFailure)): #expect(a == b)
            default: Issue.record("nondeterministic result")
            }
        }
    }
    @Test func cancellationBeforeScan() async throws {
        let task = Task {
            while !Task.isCancelled { await Task.yield() }
            return try WGSLFrontend().parse(.wgsl("fn f() {}"))
        }
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
    }
    @Test func shortCircuitCallIsOnlyInConditionalBranch() throws {
        let m = try WGSLFrontend().parse(.wgsl("fn flag() -> bool { return true; } fn f(a: bool) -> bool { return a && flag(); }"))
        let f = m.functions.values[1]
        let root = f.blocks.values[f.body.index]
        #expect(!root.statements.contains { if case .call = f.statements.values[$0.index].kind { true } else { false } })
        let conditional = try #require(root.statements.first { if case .ifCondition = f.statements.values[$0.index].kind { true } else { false } })
        if case .ifCondition(_,let accept,let reject) = f.statements.values[conditional.index].kind {
            #expect(f.blocks.values[accept.index].statements.contains { if case .call = f.statements.values[$0.index].kind { true } else { false } })
            #expect(f.blocks.values[reject.index].statements.isEmpty)
        }
    }
    @Test func loopsSwitchAndContinuingProduceStructuredIR() throws {
        let module = try WGSLFrontend().parse(.wgsl("""
        fn f() {
            var total = 0u;
            for (var i = 0u; i < 4u; i++) { total += i; }
            while total < 10u { total++; }
            loop { total++; continuing { break if total > 12u; } }
            switch total { case 1u, 2u: { total = 3u; } default: { total = 4u; } }
        }
        """))
        let function = module.functions.values[0]
        #expect(function.statements.values.filter { if case .loop = $0.kind { true } else { false } }.count == 3)
        #expect(function.statements.values.contains { if case .switchValue = $0.kind { true } else { false } })
        var budget = ShaderBudget()
        try function.walkBlocks(budget:&budget) { _ in }
    }
    @Test func aliasesStructuresConstructorsAndNaturalLayout() throws {
        let m = try WGSLFrontend().parse(.wgsl("""
        alias Index = u32;
        struct Data { x: Index, position: vec3<f32>, tail: array<vec2<f32>, 2>, }
        @group(1) @binding(2) var<storage, read_write> data: Data;
        fn f() { let d = Data(1u, vec3<f32>(2.0f), array<vec2<f32>, 2>(vec2<f32>(0.0f),vec2<f32>(1.0f))); data.x = d.x; }
        """))
        let d = try #require(m.types.values.first { $0.name == "Data" })
        if case .structure(let members,let span) = d.kind {
            #expect(members.map(\.offset) == [0,16,32]); #expect(span == 48)
        } else { Issue.record("missing structured type") }
    }
    @Test func scalarMaterializationAndConstantArithmetic() throws {
        let m = try WGSLFrontend().parse(.wgsl("const N = 7; fn f() { let x: i32 = -2; let y = 2+3*4; let z: u32 = N; let b = bool(true); }"))
        #expect(m.functions.values[0].expressions.values.contains { $0.kind == .literal(.i32(14)) })
        #expect(m.functions.values[0].expressions.values.contains { $0.kind == .literal(.i32(-2)) })
        #expect(m.functions.values[0].expressions.values.contains { $0.kind == .literal(.u32(7)) })
        for bad in ["fn f() { let n = 9223372036854775807+1; }", "fn f() { let n = 1/0; }", "fn f() { let n: u32 = -1; }"] {
            #expect(throws:ShaderFailure.self) { try WGSLFrontend().parse(.wgsl(bad)) }
        }
    }
    @Test func unicodeScalarIdentityDoesNotNormalizeIdentifiers() throws {
        let composed = String(decoding:[0xc3,0xa9],as:UTF8.self)
        let decomposed = String(decoding:[0x65,0xcc,0x81],as:UTF8.self)
        #expect(!composed.utf8.elementsEqual(decomposed.utf8))
        let source = "fn f() { let " + composed + " = 1u; let " + decomposed + " = 2u; let x = " + composed + " + " + decomposed + "; }"
        let m = try WGSLFrontend().parse(.wgsl(source))
        #expect(m.functions.count == 1)
    }
    @Test func failClosedLexicalAndSemanticBoundaries() throws {
        for s in [
            "/* unterminated", "fn f() { let x = 00; }", "fn f() { let x = 1e+; }", "fn f() { let x = 4294967296u; }",
            "fn f() { let x = 1.0 & 2.0; }", "fn f() { let x = 1u + true; }", "fn f() -> u32 { if true { return 1u; } }",
            "@group(0) var<storage> a: array<u32>;", "fn f() { let<function> x = 1u; }", "var<private, read> a: u32;",
            "fn f() { { let x = 1u; } let y=x; }", "fn f() { loop { continuing { continue; } } }",
            "fn f() { switch 1u { case 1u: {} case 1u: {} default: {} } }"
        ] { #expect(throws:ShaderFailure.self) { try WGSLFrontend().parse(.wgsl(s)) } }
    }
    @Test func formatCapabilitiesAndDocOptionsAreNotIgnored() throws {
        #expect(throws:ShaderFailure.self) { try WGSLFrontend().parse(.spirv([])) }
        let without = WGSLFrontend(options:WGSLFrontendOptions(capabilities:[]))
        #expect(throws:ShaderFailure.self) { try without.parse(.wgsl("@fragment fn f(@builtin(sample_index) i: u32) {}")) }
        #expect(throws:ShaderFailure.self) { try WGSLFrontend(options:WGSLFrontendOptions(parseDocComments:true)).parse(.wgsl("/// doc\nfn f() {}")) }
        _ = try WGSLFrontend(options:WGSLFrontendOptions(parseDocComments:true)).parse(.wgsl("// regular\nfn f() {}"))
    }
    @Test func boundedAdversarialStructuredInputs() throws {
        let f = WGSLFrontend(limits:try ShaderLimits(sourceBytes:20000,tokens:3000,nodes:5000,depth:24,work:100000))
        for source in [String(repeating:"/*",count:200)+String(repeating:"*/",count:200),
                       "fn f(){let x="+String(repeating:"(",count:200)+"1u"+String(repeating:")",count:200)+";}",
                       "fn f(){"+String(repeating:"if true {",count:100)+String(repeating:"}",count:100)+"}",
                       "alias T = "+String(repeating:"array<",count:100)+"u32"+String(repeating:",1>",count:100)+";"] {
            #expect(throws:ShaderFailure.self) { try f.parse(.wgsl(source)) }
        }
    }

}

extension WGSLFrontendTests {
    @Test func runtimeArrayStructHasMinimumFootprintAndAlignedOffset() throws {
        for (members, offset, expectedSpan) in [("items: array<u32>,", UInt32(0), UInt32(4)), ("prefix: u32, items: array<vec3<f32>>,", 16, 32)] {
            let m = try WGSLFrontend().parse(.wgsl("struct Buffer { \(members) } @group(0) @binding(0) var<storage> data: Buffer;"))
            let t = try #require(m.types.values.first { $0.name == "Buffer" })
            guard case .structure(let fields, let span) = t.kind else { Issue.record("Missing struct"); continue }
            let last = try #require(fields.last)
            guard case .array(_, .dynamic, let stride) = m.types.values[last.type.index].kind else { Issue.record("Missing runtime array"); continue }
            #expect(last.offset == offset)
            #expect(span >= last.offset + stride)
            #expect(span == expectedSpan)
        }
    }
    @Test func unsizedTypesCannotBeNestedOrFollowedByMembers() throws {
        for source in [
            "struct S { items: array<u32>, tail: u32, }",
            "struct R { items: array<u32>, } struct S { nested: R, }",
            "alias R = array<u32>; alias S = array<R, 2>;",
            "alias R = array<u32>; alias S = array<R>;",
            "struct R { items: array<u32>, } alias S = array<R, 2>;"
        ] {
            do { _ = try WGSLFrontend().parse(.wgsl(source)); Issue.record("Unsized aggregate accepted") }
            catch ShaderFailure.diagnostics(let ds) { #expect(ds.first?.code == .invalidType); #expect(ds.first?.span != nil) }
        }
    }
    @Test func runtimeLayoutOverflowAndBudgetsFailWithoutTraps() async throws {
        for source in [
            "alias Huge = array<vec4<f32>, 4294967295>; struct S { huge: Huge, items: array<u32>, }",
            "alias Huge = array<u32, 1073741823>; struct S { huge: Huge, items: array<vec4<f32>>, }"
        ] { #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl(source)) } }
        let source = "struct S { prefix: u32, items: array<vec3<f32>>, }"
        for limits in [try ShaderLimits(work: 20), try ShaderLimits(depth: 1), try ShaderLimits(nodes: 1)] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend(limits: limits).parse(.wgsl(source)) }
        }
        let task = Task { while !Task.isCancelled { await Task.yield() }; return try WGSLFrontend().parse(.wgsl(source)) }
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
    }
}

extension WGSLFrontendTests {
    @Test func boolNaturalLayoutUsesWGSLFootprintWithoutChangingIRWidth() throws {
        for (name, stride) in [("bool", UInt32(4)), ("vec2<bool>", 8), ("vec3<bool>", 16), ("vec4<bool>", 16)] {
            let raw = try WGSLFrontend().parse(.wgsl("alias A = array<\(name), 3>; struct S { first: bool, second: \(name), tail: bool, }"))
            let array = try #require(raw.types.values.first { if case .array = $0.kind { return true }; return false })
            guard case .array(_, _, let actual) = array.kind else { continue }
            #expect(actual == stride)
            let t = try #require(raw.types.values.first { $0.name == "S" })
            guard case .structure(let members, let span) = t.kind else { continue }
            #expect(members[0].offset == 0)
            #expect(members[1].offset == stride)
            #expect(members[2].offset == (name == "vec3<bool>" ? 28 : 2 * stride))
            #expect(span == (name == "bool" ? 12 : name == "vec2<bool>" ? 24 : name == "vec3<bool>" ? 32 : 48))
            for type in raw.types.values {
                switch type.kind {
                case .scalar(let s), .vector(_, let s): if s.kind == .bool { #expect(s.width == 1) }
                default: break
                }
            }
        }
    }
    @Test func boolNestedLayoutAndOverflowAreChecked() throws {
        let m = try WGSLFrontend().parse(.wgsl("struct B { b: bool, } struct S { a: bool, bs: array<B, 2>, v: vec3<bool>, } alias Rows = array<S, 2>;"))
        let t = try #require(m.types.values.first { $0.name == "S" })
        guard case .structure(let members, let span) = t.kind else { Issue.record("Missing S"); return }
        #expect(members.map(\.offset) == [0, 4, 16]); #expect(span == 32)
        #expect(m.types.values.contains { if case .array(_, .constant(2), 32) = $0.kind { return true }; return false })
        #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl("struct S { huge: array<bool, 536870912>, }")) }
        for limits in [try ShaderLimits(work: 25), try ShaderLimits(depth: 1), try ShaderLimits(nodes: 1)] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend(limits: limits).parse(.wgsl("struct S { a: array<vec3<bool>, 3>, }")) }
        }
    }
}

extension WGSLFrontendTests {
    @Test func explicitMemberLayoutPropagatesIntoNestedArrayStride() throws {
        let m = try WGSLFrontend().parse(.wgsl("struct B { @align(32) @size(48) b: bool, tail: u32, } alias A = array<B, 2>; struct Outer { prefix: u32, bs: A, }"))
        let b = try #require(m.types.values.first { $0.name == "B" })
        guard case .structure(let fields, let span) = b.kind else { return }
        #expect(fields.map(\.offset) == [0, 48]); #expect(span == 64)
        let out = try #require(m.types.values.first { $0.name == "Outer" })
        guard case .structure(let outerFields, let outerSpan) = out.kind else { return }
        #expect(outerFields.map(\.offset) == [0, 32]); #expect(outerSpan == 160)
        #expect(m.types.values.contains { if case .array(_, .constant(2), 64) = $0.kind { true } else { false } })
    }
    @Test func invalidMemberLayoutAttributesAreTypedErrors() throws {
        for source in ["struct S { @align(0) a: u32, }", "struct S { @align(3) a: u32, }", "struct S { @align(2) a: bool, }", "struct S { @size(3) a: bool, }", "struct S { @size(16) a: array<u32>, }", "struct S { @align(2147483648) a: u32, }", "struct S { @size(2147483647) a: u32, b: u32, }", "struct S { @align(8) @align(16) a: u32, }"] {
            do { _ = try WGSLFrontend().parse(.wgsl(source)); Issue.record("Invalid layout accepted") }
            catch ShaderFailure.diagnostics(let ds) { #expect(ds.first?.span != nil); #expect([ShaderDiagnosticCode.invalidType, .duplicateName].contains(ds[0].code)) }
        }
    }
}

extension WGSLFrontendTests {
    @Test func layoutAttributeDiagnosticsAndCancellationStayBounded() async throws {
        let source = "/* α */ struct S { @align(2) b: bool, }"
        let text = Array(source.utf8)
        do { _ = try WGSLFrontend().parse(.wgsl(source)); Issue.record("Invalid bool alignment accepted") }
        catch ShaderFailure.diagnostics(let ds) {
            #expect(ds.first?.code == .invalidType)
            let span = try #require(ds.first?.span)
            #expect(String(decoding: text[span.startUTF8..<span.endUTF8], as: UTF8.self) == "2")
        }
        let valid = "struct S { @align(16) @size(32) b: bool, } alias A = array<S, 2>;"
        for limits in [try ShaderLimits(work: 30), try ShaderLimits(depth: 1), try ShaderLimits(nodes: 1)] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend(limits: limits).parse(.wgsl(valid)) }
        }
        let task = Task { while !Task.isCancelled { await Task.yield() }; return try WGSLFrontend().parse(.wgsl(valid)) }
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
    }
}

extension WGSLFrontendTests {
    @Test func namedAttributeConstantsAndIntegerPrecedence() throws {
        for (declarations, expression) in [
            ("const A = 16u;", "A"),
            ("const A = 2; const B = A * (3 + 1);", "B + 8"),
            ("const A: i32 = 16;", "u32(A)"),
            ("const A = 0x10u;", "i32(A)"),
            ("const A = 1u;", "(A << 4u)"),
            ("const A = 32;", "A / 2 + 4 * 0"),
            ("const A = -8;", "-A * 2")
        ] {
            let source = declarations + "struct S { @align(\(expression)) @size(2 * (8 + 8)) x: f32, y: f32, } alias Rows = array<S, 2>;"
            let m = try WGSLFrontend().parse(.wgsl(source))
            let s = try #require(m.types.values.first { $0.name == "S" })
            guard case .structure(let fields, let span) = s.kind else { continue }
            #expect(fields.map(\.offset) == [0, 32]); #expect(span == 48)
        }
    }
    @Test func invalidAttributeExpressionsFailWithTypedRanges() throws {
        for expression in ["-1", "1u / 0u", "1i % 0i", "2147483647i + 1i", "4294967295u + 1u", "9223372036854775807 + 1", "1u + 1i", "1.0", "true", "1u << 32u", "u32(-1)"] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl("struct S { @align(\(expression)) x: f32, }")) }
        }
        let source = "/* α */ struct S { @align(missing + 4) x: f32, }"
        do { _ = try WGSLFrontend().parse(.wgsl(source)); Issue.record("Unresolved name accepted") }
        catch ShaderFailure.diagnostics(let ds) {
            #expect(ds.first?.code == .unresolvedName)
            let span = try #require(ds.first?.span)
            #expect(String(decoding: Array(source.utf8)[span.startUTF8..<span.endUTF8], as: UTF8.self) == "missing")
        }
        for source in ["var<private> A: u32; struct S { @align(A) x: f32, }", "override A: u32 = 16u; struct S { @align(A) x: f32, }", "const A = true; struct S { @align(A) x: f32, }"] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl(source)) }
        }
    }
    @Test func attributeExpressionBudgetsAndCancellation() async throws {
        let source = "const A = 8u; struct S { @align((A + A) * (1u + 0u)) x: f32, }"
        for limits in [try ShaderLimits(work: 30), try ShaderLimits(depth: 2), try ShaderLimits(nodes: 2)] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend(limits: limits).parse(.wgsl(source)) }
        }
        let task = Task { while !Task.isCancelled { await Task.yield() }; return try WGSLFrontend().parse(.wgsl(source)) }
        task.cancel(); await #expect(throws: CancellationError.self) { try await task.value }
    }
}

extension WGSLFrontendTests {
    @Test func constIntegerWrappingAndGrammarMatchPinnedRules() throws {
        for expression in ["4294967295u + 5u", "u32(-1i) + 5u", "u32(i32(4294967295u)) + 5u", "(2147483647i + 1i) - 2147483644i", "(1u | 3u) + 1u", "(16u >> 2u)", "(-i32(-2147483648)) - 2147483644i"] {
            let m = try WGSLFrontend().parse(.wgsl("struct S { @size(\(expression)) x: f32, }"))
            #expect(m.types.values.contains { if case .structure(_, 4) = $0.kind { true } else { false } })
        }
        for expression in ["1u | 2u & 3u", "1 + 1 << 3u", "1u << 1u << 2u", "1 < < 4u", "+16", "1u +"] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl("struct S { @align(\(expression)) x: f32, }")) }
        }
        // No suffix-float confusion for hex integer digits f, and existing scalar constant materialization survives.
        _ = try WGSLFrontend().parse(.wgsl("const A = 0xff; const X: f32 = 1; struct S { @size(A + 1) x: f32, }"))
    }
    @Test func workgroupConstantExpressionsHaveCommonTypeAndPositiveValues() throws {
        let m = try WGSLFrontend().parse(.wgsl("const X = 4u; @compute @workgroup_size(X * 2u, u32(2), 1) fn main() {}"))
        #expect(m.entryPoints[0].workgroupSize == .init(x: 8, y: 2, z: 1))
        for parameters in ["1i, 1u", "0", "-1", "1.0", "missing"] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl("@compute @workgroup_size(\(parameters)) fn main() {}")) }
        }
        for source in ["@align(16) fn f() {}", "@size(16) var<private> a: f32;", "struct S { @builtin(u32(1)) a: u32, }", "@compute(1) @workgroup_size(1) fn f() {}"] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl(source)) }
        }
    }
}

extension WGSLFrontendTests {
    @Test func creationFixedArrayCountsUseSameBoundedConstantRules() throws {
        let m = try WGSLFrontend().parse(.wgsl("const coeff = 2u; const N = coeff; alias A = array<vec3<bool>, N * (3u + 1u)>; struct S { a: A, }"))
        #expect(m.types.values.contains { if case .array(_, .constant(8), 16) = $0.kind { true } else { false } })
        for count in ["0", "-1", "true", "1u / 0u", "unknown", "4294967295u"] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl("alias A = array<vec4<f32>, \(count)>; struct S { a: A, }")) }
        }
    }
    @Test func attributedIntegerExpressionsRejectMalformedTreesDeterministically() throws {
        for expression in ["()", "u32()", "u32(1,2)", "(16", "16)", "((16),)", "1 ** 2", "1 + +2", "1u | 2u ^ 4u", "(1u << 32u)"] {
            let source="struct S { @align(\(expression)) x: f32, }"
            let first=Result {try WGSLFrontend().parse(.wgsl(source))}
            let second=Result {try WGSLFrontend().parse(.wgsl(source))}
            switch (first,second) {
            case (.failure(let a as ShaderFailure),.failure(let b as ShaderFailure)):#expect(a == b)
            default:Issue.record("Malformed tree did not deterministically fail")
            }
        }
    }
}

extension WGSLFrontendTests {
    @Test func boundedIntegerExpressionPropertiesAndDeepInputs() throws {
        var seed: UInt32 = 0x53414741
        for _ in 0..<128 {
            seed = seed &* 1664525 &+ 1013904223
            let a = seed % 50 + 4, b = (seed >> 8) % 50
            let source = "const A = \(a)u; const B = \(b)u; struct S { @size((A + B) * 4u) x: f32, }"
            let m = try WGSLFrontend().parse(.wgsl(source))
            let s = try #require(m.types.values.first { $0.name == "S" })
            guard case .structure(_, let span) = s.kind else { Issue.record("Missing S"); continue }
            #expect(span == (a + b) * 4)
        }
        let unary = String(repeating: "- ", count: 256) + "16"
        let brackets = String(repeating: "(", count: 256) + "16" + String(repeating: ")", count: 256)
        for expression in [unary, brackets] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl("struct S { @align(\(expression)) x: f32, }")) }
        }
        let source = "const L = 2u; struct In { @location(L - 2u) x: f32, }"
        _ = try WGSLFrontend().parse(.wgsl(source))
    }
}

extension WGSLFrontendTests {
    @Test func integerConstBuiltinsEvaluateBeforeMemberLayout() throws {
        let source = "const A = min(8u + 8u, (1u << 5u)); const B = max(4, 8); const C = clamp(B, 4, 16); struct S { @align(A) @size(u32(C) * 4u) x: f32, }"
        let m = try WGSLFrontend().parse(.wgsl(source))
        let s = try #require(m.types.values.first { $0.name == "S" })
        guard case .structure(_, let span) = s.kind else { return }
        #expect(span == 32)
        for call in ["min(1i, 1u)", "min(1)", "max(1, 2, 3)", "clamp(4, 8, 2)", "min(true, 16)"] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl("struct S { @size(\(call)) x: f32, }")) }
        }
    }
}

extension WGSLFrontendTests {
    @Test func moduleDependencyIndexResolvesForwardTypesConstantsAndDiamonds() throws {
        for source in [
            "struct S { @align(A) x: T, } alias T = f32; const A = 16u;",
            "alias Rows = array<S, N>; struct S { a: T, } const N = L + R; const R = Base; const L = Base; const Base = 2u; alias T = u32;",
            "const A: T = B; alias T = u32; const B = 16u; struct S { @align(A) a: f32, }",
            "alias A = B; alias B = C; alias C = u32; struct S { a: A, }"
        ] {
            let m = try WGSLFrontend().parse(.wgsl(source))
            #expect(!m.types.values.isEmpty)
        }
        let source = "struct S { @align(A) x: f32, } const A = 16u;"
        let m = try WGSLFrontend().parse(.wgsl(source))
        let s = try #require(m.types.values.first { $0.name == "S" })
        #expect(s.location == .source(try SourceSpan(startUTF8: 7, endUTF8: 8)))
    }
    @Test func moduleDependencyCyclesDuplicatesAndUnknownsAreLocated() throws {
        for source in ["const A = A;", "const A = B; const B = A;", "alias A = A;", "alias A = B; alias B = A;", "struct A { b: B, } struct B { a: A, }", "const A = 1u; alias A = u32;", "struct A { x: u32, } const A = 2u;"] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl(source)) }
        }
        let source = "/* α */ alias T = Missing;"
        do { _ = try WGSLFrontend().parse(.wgsl(source)); Issue.record("Unknown type accepted") }
        catch ShaderFailure.diagnostics(let ds) {
            #expect(ds.first?.code == .unresolvedName)
            let span = try #require(ds.first?.span)
            #expect(String(decoding: Array(source.utf8)[span.startUTF8..<span.endUTF8], as: UTF8.self) == "Missing")
        }
    }
    @Test func moduleDependencyBudgetsAndCancellationAreBounded() async throws {
        let deep = (0..<200).map { "alias A\($0) = A\($0+1);" }.joined() + "alias A200 = u32;"
        #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl(deep)) }
        let source = "struct S { x: A, } alias A = B; alias B = u32;"
        for limits in [try ShaderLimits(work: 20),try ShaderLimits(nodes: 2),try ShaderLimits(depth: 2)] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend(limits: limits).parse(.wgsl(source)) }
        }
        let task = Task { while !Task.isCancelled { await Task.yield() }; return try WGSLFrontend().parse(.wgsl(source)) }
        task.cancel(); await #expect(throws: CancellationError.self) { try await task.value }
    }
}

extension WGSLFrontendTests {
    @Test func moduleDependencyDiagnosticsAndWideDAGAreStable() throws {
        for (source,word,code) in [
            ("/* α */ const A = B; const B = A;", "A;", ShaderDiagnosticCode.invalidReference),
            ("/* α */ const A = 1u; alias A = u32;", "A = u32", .duplicateName),
            ("/* α */ struct S { x: Missing, }", "Missing", .unresolvedName)
        ] {
            do { _ = try WGSLFrontend().parse(.wgsl(source)); Issue.record("Bad graph accepted") }
            catch ShaderFailure.diagnostics(let ds) {
                #expect(ds.first?.code == code)
                let span = try #require(ds.first?.span)
                let token = String(decoding:Array(source.utf8)[span.startUTF8..<span.endUTF8],as:UTF8.self)
                #expect(token == String(word.prefix { $0.isLetter }))
            }
        }
        var parts = ["struct S { @size(A0) x: f32, }"]
        for n in 0..<10 {
            parts += ["const A\(n) = L\(n) + R\(n);", "const L\(n) = A\(n+1);", "const R\(n) = A\(n+1);"]
        }
        parts.append("const A10 = 4u;")
        // A layered diamond is linear cached work, not exponential source substitution.
        let m = try WGSLFrontend(limits: ShaderLimits(nodes: 2000, work: 25000)).parse(.wgsl(parts.joined()))
        #expect(m.types.values.contains { if case .structure(_,4096) = $0.kind {true} else {false} })
        let forms = ["@", "@a(", "alias", "const A=", "struct A {", "fn f( {", "var<storage", "struct A { x:u32, } alias", "fn f() {} @compute"]
        for source in forms {
            #expect(throws: ShaderFailure.self) {try WGSLFrontend().parse(.wgsl(source))}
        }
    }
}

extension WGSLFrontendTests {
    @Test func indexedNamesShadowPredeclaredFunctionsWithoutFalseEvaluation() throws {
        for source in [
            "const min = 4; struct S { @size(min(8, 16)) x: f32, }",
            "const u32 = 4; struct S { @align(u32(16)) x: f32, }",
            "const T = 1; struct S { x: T, }",
            "struct S { @size(max(8, 16)) x: f32, } const max = 4u;",
            "fn f() { let sink = 1u; sink(); } fn sink() {}",
            "fn f() -> u32 { let helper = 1u; return helper(); } fn helper() -> u32 { return 1u; }"
        ] {
            #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl(source)) }
        }
    }
}

extension WGSLFrontendTests {
  @Test func booleanConstComparisonsLogicalAndBitwiseRules() throws {
    for e in ["1 < 2", "2u >= 2u", "1 != 2", "true == !false", "(1 + 2 * 3 == 7) && (false || true)", "(true & false) == false", "(true | false) == true"] {
      _ = try WGSLFrontend().parse(.wgsl("const A = \(e); const_assert A;"))
    }
    for e in ["1 < 2 < 3", "true && false || true", "true & false | true", "true < false", "1u == 1i", "1 && true", "!1"] {
      #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl("const A = \(e);")) }
    }
  }
  @Test func constShortCircuitStillValidatesSkippedSyntaxNamesAndTypes() throws {
    for e in ["true || (i32(5000000000) == 0)", "!(false && (9223372036854775807 + 1 == 0))"] {
      _ = try WGSLFrontend().parse(.wgsl("const_assert \(e);"))
    }
    for e in ["true || missing", "false && 1", "true || (1u == 1i)", "true || (1 +)", "true || (1 / 0 == 0)", "false & (i32(5000000000) == 0)", "false || (i32(5000000000) == 0)"] {
      #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl("const A = \(e);")) }
    }
  }
  @Test func assertionsResolveForwardDependenciesAndRejectFalsePrecisely() throws {
    _ = try WGSLFrontend().parse(.wgsl("const_assert A; const A = B < 2; const B = 1; const_assert(A == true);"))
    for s in ["const A = B; const B = A;", "const_assert 1;", "var<private> x: bool; const_assert true || x;", "override x: bool = true; const_assert x;"] {
      #expect(throws: ShaderFailure.self) { try WGSLFrontend().parse(.wgsl(s)) }
    }
    let source="/* α */ const_assert 2 < 1;"
    do { _ = try WGSLFrontend().parse(.wgsl(source)); Issue.record("False assertion accepted") }
    catch ShaderFailure.diagnostics(let ds) {
      #expect(ds.first?.code == .invalidType)
      let span = try #require(ds.first?.span)
      #expect(String(decoding:Array(source.utf8)[span.startUTF8..<span.endUTF8],as:UTF8.self) == "2 < 1")
    }
  }
  @Test func booleanAndAssertionResourceLimitsRemainBounded() async throws {
    let source="const A = 1 < 2; const_assert A && (true == true);"
    for l in [try ShaderLimits(work:10),try ShaderLimits(nodes:2),try ShaderLimits(depth:1)] {
      #expect(throws: ShaderFailure.self) { try WGSLFrontend(limits:l).parse(.wgsl(source)) }
    }
    let deep=String(repeating:"!",count:256)+"true"
    #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const_assert \(deep);"))}
    let task=Task {while !Task.isCancelled {await Task.yield()};return try WGSLFrontend().parse(.wgsl(source))}
    task.cancel();await #expect(throws:CancellationError.self) {try await task.value}
  }
}

extension WGSLFrontendTests {
  @Test func skippedBranchTypeConversionsAndMandatoryDivisionAreStillChecked() throws {
    for e in ["true || (4294967296 == 1u)", "true || (min(4294967296, 1u) == 1u)", "true || (1 / (1 - 1) == 0)", "true || (i32(-2147483648) / -1i == 0)", "true || unknown"] {
      #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const A = \(e);"))}
    }
    // Integer division remains well-defined: no accidental skipped-operand placeholders causing div-zero.
    _ = try WGSLFrontend().parse(.wgsl("const_assert true || (1 / (1 + 1) == 0);"))
    for e in ["true || (1u << 32u == 0u)", "true || (clamp(1, 3, 2) == 0)"] {
      #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const_assert \(e);"))}
    }
    for e in ["true || ((i32(5000000000) / 2i) == 0i)", "true || ((i32(5000000000) >> 1u) == 0i)", "true || (clamp(i32(5000000000), 0i, 1i) == 0i)"] {
      _ = try WGSLFrontend().parse(.wgsl("const_assert \(e);"))
    }
  }
  @Test func assertionFailureAndInvalidNamesHaveExactUnicodeLocations() throws {
    for (source, expected, code) in [
      ("/* 🦎 */ fn f() { const_assert (false); }", "(false)", ShaderDiagnosticCode.invalidType),
      ("/* 🦎 */ const A = true || absent;", "absent", .unresolvedName),
      ("/* 🦎 */ const A: u32 = 1 < 2;", "1", .typeMismatch)
    ] {
      do {_ = try WGSLFrontend().parse(.wgsl(source));Issue.record("Invalid assertion/expression accepted")}
      catch ShaderFailure.diagnostics(let ds) {
        #expect(ds.first?.code == code)
        let span=try #require(ds.first?.span)
        #expect(String(decoding:Array(source.utf8)[span.startUTF8..<span.endUTF8],as:UTF8.self) == expected)
      }
    }
    for source in ["const A = false && B; const B = A;", "const_assert A; const A = A;", "const A = true; const A = false;", "@size(1) const_assert true;", "const_assert true"] {
      #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl(source))}
    }
  }
}

extension WGSLFrontendTests {
  @Test func boolAndIntegerConstConstructorsPreserveScalarTypes() throws {
    for e in ["bool(1)", "!bool(0u)", "bool(true)", "i32(true) == 1i", "u32(false) == 0u", "bool(i32(true) + 1i)"] {
      _ = try WGSLFrontend().parse(.wgsl("const_assert \(e);"))
    }
    for e in ["bool(1, 2)", "true ^ false", "1 == true"] {
      #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const A=\(e);"))}
    }
  }
}

extension WGSLFrontendTests {
  @Test func booleanConstTreePropertiesAndMalformedInputsAreDeterministic() throws {
    var seed:UInt32=0x53414741
    for _ in 0..<128 {
      seed=seed &* 1664525 &+ 1013904223
      let a=seed % 1000,b=(seed >> 10) % 1000
      let expected=a < b
      let source="const Result = (\(a)u < \(b)u); const_assert Result == \(expected ? "true":"false");"
      let m=try WGSLFrontend().parse(.wgsl(source))
      #expect(m.globalExpressions.values.contains { $0.kind == .literal(.bool(expected)) })
    }
    let alphabet=["true","false","!","(",")","&&","||","==","<",",","/","0","missing"]
    for _ in 0..<256 {
      var pieces:[String]=[]
      for _ in 0..<8 {seed=seed &* 1664525 &+ 1013904223;pieces.append(alphabet[Int(seed % UInt32(alphabet.count))])}
      let source="const_assert "+pieces.joined(separator:" ")+";"
      let first=Result {try WGSLFrontend(limits:ShaderLimits(nodes:500,depth:24,work:5000)).parse(.wgsl(source))}
      let second=Result {try WGSLFrontend(limits:ShaderLimits(nodes:500,depth:24,work:5000)).parse(.wgsl(source))}
      switch (first,second) {
      case (.success(let a),.success(let b)):#expect(a == b)
      case (.failure(let a as ShaderFailure),.failure(let b as ShaderFailure)):#expect(a == b)
      default:Issue.record("Non-deterministic or untyped failure")
      }
    }
    let chain=Array(repeating:"true",count:256).joined(separator:" && ")
    #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const_assert \(chain);"))}
  }
}

extension WGSLFrontendTests {
  @Test func scalarBooleanConstBuiltinsAndSelectUseEagerArguments() throws {
    for e in ["all(true)", "any(true)", "!any(false)", "select(false, true, true)", "select(1, 2u, true) == 2u", "select(1u, 2, false) == 1u"] {
      _ = try WGSLFrontend().parse(.wgsl("const_assert \(e);"))
    }
    for e in ["all(1)", "any(true, false)", "select(true, 1, true)", "select(1u, 1i, true)", "select(1, 2, 1)", "select(i32(5000000000), 1i, true) == 1i"] {
      #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const A=\(e);"))}
    }
  }
}

extension WGSLFrontendTests {
  @Test func floatingConstantsUseTypedPrecisionAndComparisons() throws {
    for e in ["1.0 < 2.0", "-1.5 + 2.5 == 1.0", "(1.0 + 2.0 * 3.0) == 7.0", "(16777216.0f + 1.0f) - 16777216.0f == 0.0f", "(16777216.0 + 1.0) - 16777216.0 == 1.0", "1 + 0.5 == 1.5", "1 + 0.5f == 1.5f", "f32(1.5) == 1.5f", "bool(0.5)", "i32(3.9f)==3i", "u32(-1.0f)==0u", "u32(1e20f)==4294967040u"] {
      _ = try WGSLFrontend().parse(.wgsl("const_assert \(e);"))
    }
    for e in ["1u + 0.5", "~1.0", "1.0 & 2.0", "1.0 << 2u", "1.0 == true", "f32(1e100)", "1e308 * 2.0", "3.4e38f * 2.0f", "1.0/0.0", "0.0/0.0", "1.0%0.0"] {
      #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const A=\(e);"))}
    }
  }
  @Test func lazyFloatExpressionsValidateTypesWithoutEvaluatingOrdinarySkippedWork() throws {
    for e in ["true || (f32(1e100)==0.0f)", "true || (1.0/0.0 == 0.0)", "true || (1e308*2.0 == 0.0)"] {
      _ = try WGSLFrontend().parse(.wgsl("const_assert \(e);"))
    }
    for e in ["true || (1u == 1.0f)", "true || missing", "true || (1e100 == 0.0f)", "true || (1.0 +)", "false | (f32(1e100)==0.0f)"] {
      #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const A=\(e);"))}
    }
  }
}

extension WGSLFrontendTests {
  @Test func scalarAliasConstructorsResolveTypesAndRespectLexicalShadowing() throws {
    for source in ["alias B=bool; const_assert B(1);", "const_assert B(1); alias B=Flag; alias Flag=bool;", "alias F=f32; const x=F(1.25); const_assert x<2.0f;", "alias I=i32; const_assert I(3.9f)==3i;", "alias F=f32; fn f(){const x=F(1.5); const_assert x==1.5f;}"] {
      _ = try WGSLFrontend().parse(.wgsl(source))
    }
    for source in ["alias B=bool; fn f(){const B=1; const_assert B(1);}", "const B=1; const_assert B(1);", "alias B=C; alias C=B; const_assert B(1);", "alias B=bool; const_assert B(1,2);", "alias F=f32; const_assert true || (F(missing)==1.0f);"] {
      #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl(source))}
    }
  }
  @Test func floatingRoundingConversionsAndDomainsAreNotIntegerWrapping() throws {
    for e in ["f32(true)==1.0f", "i32(1e20f)==2147483520i", "i32(-1e20f)==i32(-2147483648)", "u32(4294967295.0)==4294967295u", "(5.5%2.0)==1.5", "min(1.25,2.0)==1.25", "max(1,2.0f)==2.0f", "clamp(2.5,1.0,2.0)==2.0", "select(1.0,2.0f,true)==2.0f"] {
      _ = try WGSLFrontend().parse(.wgsl("const_assert \(e);"))
    }
    for e in ["true || (clamp(1.0,3.0,2.0)==1.0)", "true || (1e100 + 1.0f == 0.0f)", "select(f32(1e100),1.0f,true)==1.0f", "min(1.0,2u)", "f32(3.4028235e38)", "1e309", "-1e309"] {
      #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const A=\(e);"))}
    }
    _ = try WGSLFrontend().parse(.wgsl("const_assert true || (clamp(f32(1e100),0.0f,1.0f)==0.0f);"))
  }
}

extension WGSLFrontendTests {
  @Test func floatConstPrecisionPropertiesAndResourceLimitsRemainBounded() async throws {
    // Values sit on binary32 binade boundaries: each intermediate must round in the operand type.
    for power in 24...30 {
      let base=UInt64(1)<<power
      let step=UInt64(1)<<(power-23)
      let source="const x=f32(\(base)); const_assert (x+\(step).0f)-x==\(step).0f; const_assert (x+0.25f)-x==0.0f;"
      _ = try WGSLFrontend().parse(.wgsl(source))
    }
    for e in ["f32()==0.0f", "!bool()", "i32()==0i", "u32()==0u", "(1.0f/2.0f)==0.5f", "(1/2.0)==0.5", "1.0f%0.5f==0.0f", "f32(1e-46)==0.0f", "true || (select(1e100,0.0f,true)==0.0f)"] {
      if e.contains("select") {
        #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const_assert \(e);"))}
      } else {_ = try WGSLFrontend().parse(.wgsl("const_assert \(e);"))}
    }
    let source="alias F=f32; const A=1.25; fn f(){const x:F=A*2.0; const_assert x==2.5f;}"
    for limits in [try ShaderLimits(work:20),try ShaderLimits(nodes:2),try ShaderLimits(depth:2)] {
      #expect(throws:ShaderFailure.self) {try WGSLFrontend(limits:limits).parse(.wgsl(source))}
    }
    let deep=String(repeating:"- ",count:256)+"1.0"
    #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const A=\(deep);"))}
    let task=Task {while !Task.isCancelled {await Task.yield()};return try WGSLFrontend().parse(.wgsl(source))}
    task.cancel();await #expect(throws:CancellationError.self) {try await task.value}
  }
  @Test func floatConstDiagnosticsAndCreationChecksPreserveUTF8AndLazyTyping() throws {
    for (expression, token, code) in [("1.0 < 0.0","1.0 < 0.0",ShaderDiagnosticCode.invalidType), ("true || (1.0 + missing == 0.0)","missing",.unresolvedName), ("1.0u < 2.0","1.0u",.invalidToken)] {
      let source="/* 🦎 */ const_assert \(expression);"
      do {_ = try WGSLFrontend().parse(.wgsl(source));Issue.record("Invalid floating expression accepted")}
      catch ShaderFailure.diagnostics(let ds) {
        #expect(ds.first?.code == code)
        let span=try #require(ds.first?.span)
        #expect(String(decoding:Array(source.utf8)[span.startUTF8..<span.endUTF8],as:UTF8.self)==token)
      }
    }
    for source in ["override x:f32=1.0; const_assert x<2.0;", "const A=B+1.0; const B=A;", "fn f(){const f32=1; const_assert true || (f32(1)==1.0f);}", "const x=1.0; struct S{@size(x) a:f32,}", "const x=1.0; alias A=array<u32,x>;"] {
      #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl(source))}
    }
  }
}

extension WGSLFrontendTests {
 @Test func vectorConstConstructorsAliasesAndBooleanReductions() throws {
  for source in [
   "const_assert all(vec2<bool>(true,true));",
   "alias V=vec2<bool>; const_assert all(V(true,true));",
   "const_assert all(V(true)); alias V=W; alias W=vec4<bool>;",
   "const_assert !any(vec3<bool>());",
   "const V=vec4(true,vec2(true,true),true); const_assert all(V);",
   "const_assert all(vec2<bool>(vec2<i32>(1,2)));",
   "const_assert all(vec3<f32>(vec3(1,2,3)) == vec3<f32>(1.0,2.0,3.0));",
   "const_assert all(vec2(1,2) == vec2<u32>(1u,2u));",
   "const_assert all(vec2() == vec2<i32>(0));"
  ] {_ = try WGSLFrontend().parse(.wgsl(source))}
 }
 @Test func invalidVectorConstructorsValidateWholeSkippedTree() throws {
  for e in ["all(vec2<bool>(true,false,true))","all(vec2<bool>(1,1))","all(vec3<bool>(vec2<bool>(true)))","all(vec2<i32>(1u))","all(vec2(1i,2u))","all(vec4(vec2(true),true))","all(vec5<bool>(true))","all(vec2<missing>(true))","true || all(vec2<bool>(true,missing))","true || all(vec2<f32>(1e100,0.0))","all(vec2<vec2<bool>>(true))"] {
   #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const A=\(e);"))}
  }
  for s in ["alias V=vec2<bool>; fn f(){const V=true; const_assert all(V(true));}","alias V=W; alias W=V; const_assert all(V(true));"] {
   #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl(s))}
  }
 }
}

extension WGSLFrontendTests {
 @Test func vectorComponentsPrecisionLazyValidationAndBudgets() async throws {
  for e in [
   "all(vec4<bool>(vec2<bool>(true),vec2<bool>(true)))",
   "all(vec2<f32>(vec2(16777217,16777216)) == vec2<f32>(16777216.0f))",
   "all(vec2<u32>(vec2(-1i,2i)) == vec2<u32>(4294967295u,2u))",
   "any(vec3<bool>(false,true,false))",
   "true || all(vec2<f32>(vec2(1e100)) == vec2<f32>(0.0f))"
  ] {_ = try WGSLFrontend().parse(.wgsl("const_assert \(e);"))}
  for e in ["true || all(vec2<f32>(1e100))","all(vec2<u32>(-1))","true || all(vec2<f32>(vec2(missing)))","all(vec2<bool>(true) == vec3<bool>(true))","vec2<bool>(true) && true"] {
   #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const A=\(e);"))}
  }
  let source="alias V=vec4<bool>; const_assert all(V(vec2<bool>(true),vec2<bool>(true)));"
  for l in [try ShaderLimits(nodes:3),try ShaderLimits(work:20),try ShaderLimits(depth:2)] {
   #expect(throws:ShaderFailure.self) {try WGSLFrontend(limits:l).parse(.wgsl(source))}
  }
  let task=Task {while !Task.isCancelled {await Task.yield()};return try WGSLFrontend().parse(.wgsl(source))};task.cancel()
  await #expect(throws:CancellationError.self) {try await task.value}
 }
 @Test func vectorConstantMalformationsAreBoundedAndLocated() throws {
  let source="/* 🦎 */ const_assert all(vec2<bool>(true,missing));"
  do {_ = try WGSLFrontend().parse(.wgsl(source));Issue.record("Unknown accepted")}
  catch ShaderFailure.diagnostics(let ds) {
   #expect(ds[0].code == .unresolvedName);let span=try #require(ds[0].span)
   #expect(String(decoding:Array(source.utf8)[span.startUTF8..<span.endUTF8],as:UTF8.self)=="missing")
  }
  for count in 0...12 {
   let expr="vec2<bool>("+Array(repeating:"true",count:count).joined(separator:",")+")"
   if count == 0 || count == 1 || count == 2 {_ = try WGSLFrontend().parse(.wgsl("const V=\(expr);"))}
   else {#expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const V=\(expr);"))}}
  }
  let nested=String(repeating:"vec2<bool>(",count:80)+"true"+String(repeating:")",count:80)
  #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const V=\(nested);"))}
 }
}

extension WGSLFrontendTests {
 @Test func floatingSqrtBuiltinsAndHexLiteralsHaveTypedPrecision() throws {
  for e in ["sqrt(4.0)==2.0","sqrt(4.0f)==2.0f","inverseSqrt(4.0f)==0.5f","sqrt(4)==2.0","all(sqrt(vec2<f32>(4.0,9.0))==vec2<f32>(2.0,3.0))","0x1p0 < 2.0","0X1.8P1f==3.0f","0x.8==0.5","0x1.000002p0f > 1.0f","0x1p-149f > 0.0f"] {
   _ = try WGSLFrontend().parse(.wgsl("const_assert \(e);"))
  }
  for e in ["sqrt(-1.0)","inverseSqrt(0.0f)","sqrt(4u)","sqrt(true)","sqrt(1.0,2.0)","0x1.00000001p0f","0x1p-150f","0x1p128f","0x1p1024"] {
   #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const A=\(e);"))}
  }
  _ = try WGSLFrontend().parse(.wgsl("const_assert true || sqrt(-1.0)==0.0;"))
 }
}

extension WGSLFrontendTests {
 @Test func vectorUnaryOperationsAndHexPrecisionEdgesAreChecked() throws {
  for e in ["all(!vec3<bool>(false))","all((-vec2<i32>(1,2))==vec2<i32>(-1,-2))","all((~vec2<u32>(0u))==vec2<u32>(4294967295u))","0x1.f==1.9375","0x0001.000000000000000p0f==1.0f","0x1.fffffep127f==3.4028234663852886e38f"] {
   _ = try WGSLFrontend().parse(.wgsl("const_assert \(e);"))
  }
  for e in ["!vec2<u32>(1u)","-vec2<u32>(1u)","~vec2<f32>(1.0f)","0x1.0000000000000001p0f","0x.p1f","0x1p+f"] {
   #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const A=\(e);"))}
  }
 }
}

extension WGSLFrontendTests {
 @Test func vectorPropertyCasesAndHexDiagnosticsAreDeterministic() throws {
  var seed:UInt32=0x53414741
  for _ in 0..<96 {
   seed=seed &* 1664525 &+ 1013904223
   let n=Int(seed%3)+2
   let xs=(0..<n).map { (seed >> UInt32($0*3)) & 1 == 1 }
   let args=xs.map {$0 ? "true":"false"}.joined(separator:",")
   let source="const V=vec\(n)<bool>(\(args)); const_assert all(V)==\(xs.allSatisfy {$0} ? "true":"false"); const_assert any(V)==\(xs.contains(true) ? "true":"false");"
   let a=try WGSLFrontend().parse(.wgsl(source)),b=try WGSLFrontend().parse(.wgsl(source));#expect(a == b)
  }
  let alphabet=["vec2","<","bool",">","(",")",",","true","false","missing","1u"]
  for _ in 0..<128 {
   var pieces:[String]=[]
   for _ in 0..<9 {seed=seed &* 1664525 &+ 1013904223;pieces.append(alphabet[Int(seed%UInt32(alphabet.count))])}
   let source="const V="+pieces.joined(separator:" ")+";"
   let a=Result {try WGSLFrontend(limits:ShaderLimits(nodes:500,depth:24,work:5000)).parse(.wgsl(source))}
   let b=Result {try WGSLFrontend(limits:ShaderLimits(nodes:500,depth:24,work:5000)).parse(.wgsl(source))}
   switch(a,b) {case(.success(let x),.success(let y)):#expect(x == y);case(.failure(let x as ShaderFailure),.failure(let y as ShaderFailure)):#expect(x == y);default:Issue.record("Untyped/non-deterministic vector failure")}
  }
  let source="/* 🦎 */ const X=0x1.0000000000000001p0f;"
  do {_ = try WGSLFrontend().parse(.wgsl(source));Issue.record("Inexact hex f32 accepted")}
  catch ShaderFailure.diagnostics(let ds) {
   #expect(ds[0].code == .invalidToken);let span=try #require(ds[0].span)
   #expect(String(decoding:Array(source.utf8)[span.startUTF8..<span.endUTF8],as:UTF8.self)=="0x1.0000000000000001p0f")
  }
  let long="0x1."+String(repeating:"0",count:1000)+"1p0f"
  #expect(throws:ShaderFailure.self) {try WGSLFrontend(limits:ShaderLimits(work:200)).parse(.wgsl("const X=\(long);"))}
 }
}

extension WGSLFrontendTests {
 @Test func vectorArithmeticAndSelectHaveNormativeShapesAndPrecision() throws {
  for e in [
   "all(vec2(1,2)+vec2(3,4)==vec2(4,6))",
   "all((2-vec2(1,3))==vec2(1,-1))",
   "all((vec2<u32>(4294967295u,2u)+1u)==vec2<u32>(0u,3u))",
   "all((vec2(6,8)/2)==vec2(3,4))", "all((12/vec2(3,4))==vec2(4,3))",
   "all((vec2(5,6)%2)==vec2(1,0))", "all((vec2(2,3)*2)==vec2(4,6))",
   "all((vec2<u32>(1u,3u)^vec2<u32>(2u,1u))==vec2<u32>(3u,2u))",
   "all((vec2<i32>(-4i,8i)>>vec2<u32>(1u,2u))==vec2<i32>(-2i,2i))",
   "all((vec2(1,2)<<vec2<u32>(2u,1u))==vec2(4,4))",
   "all((vec2<f32>(16777216.0f)+vec2<f32>(1.0f))-vec2<f32>(16777216.0f)==vec2<f32>(0.0f))",
   "all(select(vec2(1,2),vec2<u32>(3u,4u),vec2<bool>(true,false))==vec2<u32>(3u,2u))",
   "all(select(vec2<bool>(false),vec2<bool>(true),true))",
   "!any(select(vec2<bool>(false),vec2<bool>(true),false))"
  ] {_ = try WGSLFrontend().parse(.wgsl("const_assert \(e);"))}
  for e in ["vec2(1,2)&1", "vec2(1,2)<<1u", "vec2(1,2)+vec3(1,2,3)", "vec2<i32>(1i)+vec2<u32>(1u)", "vec2<bool>(true)+vec2<bool>(true)", "select(vec2(1,2),vec2(3,4),vec3<bool>(true))", "select(1,2,vec2<bool>(true))", "true || all((vec2(1,2)/vec2(1,0))==vec2(1))", "true || all((vec2<i32>(1)<<vec2<u32>(32u))==vec2<i32>(1))", "true || all((vec2(4294967296)+vec2<u32>(0u))==vec2<u32>(0u))"] {
   #expect(throws:ShaderFailure.self, Comment(rawValue:e)) {try WGSLFrontend().parse(.wgsl("const A=\(e);"))}
  }
 }
 @Test func constVectorPostfixAccessHasExactBoundsAndSwizzleRules() throws {
  for e in ["vec2(1,2)[0]==1", "vec2(1,2)[1u]==2", "vec2(1,2).y==2", "all(vec4(1,2,3,4).bgra==vec4(3,2,1,4))", "all(vec2(1,2).xyxx==vec4(1,2,1,1))", "(vec2(1,2)+1).yx[0]==3", "vec4(1,2,3,4)[1+1]==3"] {
   _ = try WGSLFrontend().parse(.wgsl("const_assert \(e);"))
  }
  for e in ["vec2(1,2)[-1]", "vec2(1,2)[2u]", "vec2(1,2)[4294967295u]", "vec2(1,2)[true]", "vec2(1,2)[1.0]", "vec2(1,2).xr", "vec2(1,2).z", "vec2(1,2).xxxxx", "vec2(1,2).α", "1.x", "true || (vec2(1,2)[2]==0)"] {
   #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const A=\(e);"))}
  }
 }
}

extension WGSLFrontendTests {
 @Test func vectorOperationDomainsAreCheckedBeforeLazyEvaluation() throws {
  for e in [
   "true || all((vec2<i32>(i32(-2147483648),1i)/vec2<i32>(-1i,1i))==vec2<i32>(0i))",
   "true || all((vec2(1,2)%vec2(0,1))==vec2(0))",
   "true || all((vec2<u32>(2147483648u)<<vec2<u32>(1u))==vec2<u32>(0u))",
   "true || all((vec2<i32>(1i)>>vec2<u32>(32u))==vec2<i32>(0i))",
   "true || all((vec2(1e100)+vec2<f32>(0.0f))==vec2<f32>(0.0f))",
   "all(select(vec2<f32>(f32(1e100)),vec2<f32>(0.0f),true)==vec2<f32>(0.0f))",
   "true || all(select(vec2(4294967296),vec2<u32>(0u),true)==vec2<u32>(0u))"
  ] {#expect(throws:ShaderFailure.self,Comment(rawValue:e)) {try WGSLFrontend().parse(.wgsl("const_assert \(e);"))}}
  for e in [
   "all((vec2(1)<<vec2<u32>(32u))==vec2(4294967296))",
   "true || all((vec2<i32>(i32(5000000000))/vec2<i32>(2i))==vec2<i32>(0i))",
   "true || all((vec2<f32>(f32(1e100))+vec2<f32>(1.0f))==vec2<f32>(0.0f))"
  ] {_ = try WGSLFrontend().parse(.wgsl("const_assert \(e);"))}
  for e in ["vec2(9223372036854775807)+1", "vec2(9223372036854775807)*2", "vec2<f32>(3.4e38f)*2.0f"] {
   #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const A=\(e);"))}
  }
 }
 @Test func vectorAccessDiagnosticsUseOriginalUnicodeSpans() throws {
  for (e,expected,code) in [
   ("vec2(1,2)[-1]","-1",ShaderDiagnosticCode.invalidType),
   ("vec2(1,2)[2u]","2u",.invalidType),
   ("vec2(1,2)[4294967295u]","4294967295u",.invalidType),
   ("vec2(1,2).xr","xr",.invalidType),
   ("vec2(1,2).xxxxx","xxxxx",.invalidType),
   ("vec2(1,2).α","α",.invalidType),
   ("vec2(1,2)[missing]","missing",.unresolvedName)
  ] {
   let source="/* 🦎 */ const A=\(e);"
   do {_ = try WGSLFrontend().parse(.wgsl(source));Issue.record("Invalid access accepted")}
   catch ShaderFailure.diagnostics(let ds) {
    #expect(ds.first?.code==code);let span=try #require(ds.first?.span)
    #expect(String(decoding:Array(source.utf8)[span.startUTF8..<span.endUTF8],as:UTF8.self)==expected)
   }
  }
 }
 @Test func vectorOperationPropertiesBudgetsAndCancellationStayBounded() async throws {
  var seed:UInt32=0x53414741
  for _ in 0..<96 {
   seed=seed &* 1664525 &+ 1013904223
   let a=seed%100,b=(seed>>8)%100,c=(seed>>16)%10+1
   let source="const V=vec2(\(a),\(b)); const S=(V+\(c)).yx; const_assert S[0]==\(b+c); const_assert S.x==\(b+c); const_assert all(select(V,V+1,vec2<bool>(false,true))==vec2(\(a),\(b+1)));"
   _ = try WGSLFrontend().parse(.wgsl(source))
  }
  let source="const V=vec2(1,2); const_assert all(select(V+1,V*2,vec2<bool>(true,false)).yx==vec2(3,2));"
  _ = try WGSLFrontend().parse(.wgsl(source))
  for limits in [try ShaderLimits(work:30),try ShaderLimits(nodes:4),try ShaderLimits(depth:2)] {
   #expect(throws:ShaderFailure.self) {try WGSLFrontend(limits:limits).parse(.wgsl(source))}
  }
  let deep="vec2(1,2)"+String(repeating:".yx",count:128)
  #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const V=\(deep);"))}
  let task=Task {while !Task.isCancelled {await Task.yield()};return try WGSLFrontend().parse(.wgsl(source))};task.cancel()
  await #expect(throws:CancellationError.self) {try await task.value}
 }
}

extension WGSLFrontendTests {
 @Test func matrixConstConstructorsAndColumnAccessUseAuditedOverloads() throws {
  for source in [
   "const M=mat2x2<f32>(1.0,2.0,3.0,4.0); const_assert all(M[0]==vec2<f32>(1.0,2.0)); const_assert M[1][0]==3.0f;",
   "alias M=mat3x2<f32>; const X=M(vec2<f32>(1.0),vec2<f32>(2.0),vec2<f32>(3.0)); const_assert X[2].y==3.0f;",
   "const M=mat2x3(); const_assert all(M[1]==vec3(0.0));",
   "const M=mat2x2(1,2,3,4); const N=mat2x2<f32>(M); const_assert N[1].y==4.0f;"
  ] {_ = try WGSLFrontend().parse(.wgsl(source))}
  for expression in ["mat2x2<f32>(1.0)","mat2x2<f32>(1u,2u,3u,4u)","mat2x2<f32>(vec3<f32>(1.0),vec3<f32>(2.0))","mat2x2<f32>(1.0,2.0,3.0)","mat2x2<f32>()[2]","mat2x2<f32>().xy","mat2x2<bool>()","mat2x2<f32>(mat3x2<f32>())"] {
   #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const X=\(expression);"))}
  }
 }
}

extension WGSLFrontendTests {
 @Test func matrixConstPrecisionShapesAndBudgetsAreExplicit() async throws {
  for c in 2...4 {for r in 2...4 {
   let components=(0..<(c*r)).map {String($0)+".0"}.joined(separator:",")
   let source="const M=mat\(c)x\(r)<f32>(\(components)); const_assert M[\(c-1)][\(r-1)]==\(c*r-1).0f;"
   _ = try WGSLFrontend().parse(.wgsl(source))
  }}
  _ = try WGSLFrontend().parse(.wgsl("const M=mat2x2<f32>(16777217,0,0,0); const_assert M[0][0]==16777216.0f;"))
  for expression in ["mat2x2<f32>(1e100,0.0,0.0,1.0)","true || mat2x2<f32>()[2][0]==0.0f","bool(mat2x2<f32>())","mat2x2<f32>()[true]"] {
   #expect(throws:ShaderFailure.self) {try WGSLFrontend().parse(.wgsl("const A=\(expression);"))}
  }
  let source="const M=mat4x4<f32>(); const_assert all(M[0]==vec4<f32>());"
  for l in [try ShaderLimits(nodes:3),try ShaderLimits(work:20),try ShaderLimits(depth:1)] {
   #expect(throws:ShaderFailure.self) {try WGSLFrontend(limits:l).parse(.wgsl(source))}
  }
  let t=Task {while !Task.isCancelled {await Task.yield()};return try WGSLFrontend().parse(.wgsl(source))};t.cancel();await #expect(throws:CancellationError.self) {try await t.value}
 }
}

extension WGSLFrontendTests {
 @Test func malformedVectorOperationTreesOnlyReturnTypedFailures() throws {
  var seed:UInt32=0x53414741
  let alphabet=["V","[","]",".","xy","xr","α","0","-1","2u","+","*","&&",",","(",")","select","true"]
  for _ in 0..<192 {
   var tokens:[String]=[]
   for _ in 0..<10 {seed=seed &* 1664525 &+ 1013904223;tokens.append(alphabet[Int(seed%UInt32(alphabet.count))])}
   let source="const V=vec2(1,2); const X="+tokens.joined(separator:" ")+";"
   let a=Result {try WGSLFrontend(limits:ShaderLimits(nodes:500,depth:24,work:5000)).parse(.wgsl(source))}
   let b=Result {try WGSLFrontend(limits:ShaderLimits(nodes:500,depth:24,work:5000)).parse(.wgsl(source))}
   switch (a,b) {case (.success(let x),.success(let y)):#expect(x == y);case(.failure(let x as ShaderFailure),.failure(let y as ShaderFailure)):#expect(x == y);default:Issue.record("Untyped or unstable operation failure")}
  }
 }
}
