import Testing
import SAGAShaderIR
import SAGAFrontends

// Independently authored test literals, not copied from Naga fixtures.
@Suite struct GLSLFrontendTests {
    func parse(_ source: String, stage: ShaderStage = .compute, defines: [String: String] = [:]) throws -> IRModule {
        try GLSLFrontend(options: .init(stage: IRStage(stage))).parse(.glsl(source, stage: stage, defines: defines))
    }
    @Test func computeHasRealTypedArithmeticAndControlFlow() throws {
        let m = try parse("#version 450\nlayout(local_size_x=4) in; void main(){uint x=1u; while(x<4u){x=x+1u;} if(x==4u){return;} }")
        #expect(m.entryPoints[0].workgroupSize.x == 4)
        #expect(m.functions.values.contains { $0.expressions.values.contains { if case .binary(op: .add, left: _, right: _) = $0.kind { return true }; return false } })
        #expect(m.functions.values.contains { $0.statements.values.contains { if case .loop = $0.kind { return true }; return false } })
    }
    @Test func objectAndFunctionMacrosAndConditionalDefines() throws {
        let m = try parse("#version 460\n#define PLUS(a,b) ((a)+(b))\n#if defined(ENABLED) && ENABLED == 2\nvoid main(){int x=PLUS(1,2);}\n#else\n#error wrong branch\n#endif", defines: ["ENABLED":"2"])
        #expect(m.functions.values[0].localVariables.count == 1)
    }
    @Test func vertexOutputWrapperAndHelperCall() throws {
        let m = try parse("#version 450\nfloat twice(float a){return a*2.0;} void main(){gl_Position=vec4(twice(1.0));}", stage: .vertex)
        #expect(m.entryPoints[0].stage == .vertex)
        let entry = try m.functions.value(for: m.entryPoints[0].function)
        #expect(entry.result != nil)
        #expect(m.functions.values.contains { $0.statements.values.contains { if case .call = $0.kind { return true }; return false } })
    }
    @Test func rejectsMalformedNamesTypesAndUnsupportedRatherThanSkipping() throws {
        for source in ["void main(){foo=1;}","void main(){int x=true;}","void main(){int x=;}","void main(){for(;;}","void main(){return 1;}","void main(){break;}","void main(){int x=1; int x=2;}"] {
            #expect(throws: ShaderFailure.self) { try parse("#version 450\n"+source) }
        }
    }
    @Test func versionStageAndOptionErrors() throws {
        for version in ["100", "330", "450 es", "450 compatibility"] {
            #expect(throws: ShaderFailure.self) { try parse("#version \(version)\nvoid main(){}") }
        }
        for version in ["440", "450 core", "460"] { _ = try parse("#version \(version)\nvoid main(){}") }
        #expect(throws: ShaderFailure.self) { try GLSLFrontend(options: .init(stage:.vertex)).parse(.glsl("#version 450\nvoid main(){}",stage:.compute,defines:[:])) }
        #expect(throws: ShaderFailure.self) { try parse("#version 450\nvoid main(){}", defines:["not a name":"1"]) }
    }
    @Test func originalUTF8DiagnosticAndMacroUseSite() throws {
        let s="#version 450\n// 🌏é\n#define BAD missing\nvoid main(){int x=BAD;}"
        do { _ = try parse(s); Issue.record("expected unresolved name") }
        catch ShaderFailure.diagnostics(let ds) {
            let span = try #require(ds.first?.span)
            #expect(ds.first?.code == .unresolvedName)
            let bytes=Array(s.utf8)
            #expect(String(decoding: bytes[span.startUTF8..<span.endUTF8],as:UTF8.self)=="BAD")
        }
    }
    @Test func boundedTokensWorkDepthAndCancellation() async throws {
        for limits in [try ShaderLimits(tokens:2), try ShaderLimits(nodes:0), try ShaderLimits(depth:2), try ShaderLimits(work:3)] {
            let f=GLSLFrontend(options:.init(stage:.compute),limits:limits)
            #expect(throws: ShaderFailure.self) { try f.parse(.glsl("#version 450\nvoid main(){int x=((((1))));}",stage:.compute,defines:[:])) }
        }
        let task=Task { while !Task.isCancelled { await Task.yield() }; return try parse("#version 450\nvoid main(){}") }
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
    }
    @Test func deterministicMalformedFuzzIsBoundedAndOnlyTypedFailures() throws {
        var seed: UInt64=14
        let alphabet=Array("{}()[];012ab#/+*<>=!\n ".utf8)
        let f=GLSLFrontend(options:.init(stage:.compute), limits:try ShaderLimits(sourceBytes:4096,tokens:512,nodes:512,depth:32,work:20000))
        for n in 0..<256 {
            var bytes=Array("#version 450\n".utf8)
            for _ in 0..<(n%80) { seed=seed &* 6364136223846793005 &+ 1; bytes.append(alphabet[Int(seed%UInt64(alphabet.count))]) }
            do { _ = try f.parse(.glsl(String(decoding:bytes,as:UTF8.self),stage:.compute,defines:[:])) }
            catch is ShaderFailure {} // No unrelated errors or crashes are masked.
        }
    }
    @Test func storageBlockAndForLoopHaveRealAccessAndContinuing() throws {
        let m=try parse("#version 450\nlayout(set=0,binding=0,std430) buffer Data { uint values[]; } data; layout(local_size_x=8) in; void main(){for(uint i=0u;i<8u;i++){data.values[i]=i+1u;}}")
        #expect(m.globalVariables.values.contains { $0.binding == IRResourceBinding(group:0,binding:0) })
        #expect(m.functions.values.contains { $0.statements.values.contains { if case .store = $0.kind { return true }; return false } })
        #expect(m.types.values.contains { if case .array(_, .dynamic, _) = $0.kind { return true }; return false })
    }
    @Test func preprocessorShortCircuitDoesNotDivideByZero() throws {
        _ = try parse("#version 450\n#if 0 && (1/0)\n#error dead\n#endif\nvoid main(){}")
        _ = try parse("#version 450\n#if 1 || (1/0)\nvoid main(){}\n#endif")
    }
    @Test func rejectsReadOnlyAndNonconstantConst() throws {
        #expect(throws: ShaderFailure.self) { try parse("#version 450\nlayout(set=0,binding=0,std430) readonly buffer Data { uint x; } data; void main(){data.x=1u;}") }
        #expect(throws: ShaderFailure.self) { try parse("#version 450\nint f(){return 2;} void main(){const int x=f();}") }
    }

}
