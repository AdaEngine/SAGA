import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let overrideArrayShader = """
    @id(4) override lanes = 2u;
    override count = lanes * 2u + 1u;
    override unused: u32;
    alias Scratch = array<u32, count>;
    var<workgroup> scratch: Scratch;
    var<workgroup> extra: array<u32, lanes + 1u>;
    var<workgroup> uncalled: array<u32, unused>;
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;
    fn neverCalled() { uncalled[7u] = 100u; }
    fn readNext(i: u32) -> u32 { return scratch[(i + 1u) % lanes]; }
    @compute @workgroup_size(lanes)
    fn run(@builtin(local_invocation_index) i: u32) {
        let initial = scratch[lanes + i] + extra[i];
        workgroupBarrier();
        scratch[i] = 10u + i;
        extra[i] = 20u + i;
        workgroupBarrier();
        output[i] = initial * 1000u + readNext(i) + extra[i];
    }
    """

@Test func preservesNamedAndComputedArrayCountsInTypedIR() throws {
    let module = try IndependentShaderCompiler().parseAndValidate(.wgsl(overrideArrayShader)).module
    let named = try #require(module.globalVariables.values.first { $0.name == "scratch" })
    let computed = try #require(module.globalVariables.values.first { $0.name == "extra" })
    guard case .array(_, .override(let namedCount), 4) = try module.types.value(for: named.type).kind,
          case .array(_, .override(let computedCount), 4) = try module.types.value(for: computed.type).kind else {
        Issue.record("Expected typed deferred array counts")
        return
    }
    #expect(try module.overrides.value(for: namedCount).name == "count")
    let synthetic = try module.overrides.value(for: computedCount)
    #expect(synthetic.name == nil && synthetic.id == nil && synthetic.initializer != nil)
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func specializesWorkgroupArrayFootprints(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(overrideArrayShader)
    let first = try compiler.compileCompute(source, to: format)
    let second = try compiler.compileCompute(source, to: format, constants: ["4": 4, "unused": 50000])
    #expect(first.workgroupSize == .init(x: 2))
    #expect(second.workgroupSize == .init(x: 4))
    #expect(first.workgroupStorageByteSize == 48)
    #expect(second.workgroupStorageByteSize == 80)
    guard case .text(_, let code) = second.artifact else { Issue.record("Expected shader source"); return }
    if format == .wgsl {
        #expect(code.contains("array<u32, 9>") && code.contains("array<u32, 5>"))
        _ = try compiler.parseAndValidate(.wgsl(code))
    } else {
        #expect(code.contains("[9]") && code.contains("[5]"))
    }
}

@Test(arguments: [
    ("override n = 2u; var<private> a: array<u32, n>;", ShaderDiagnosticCode.invalidAddressSpace),
    ("override n = 2u; @group(0) @binding(0) var<storage, read> a: array<u32, n>;", .invalidAddressSpace),
    ("override n = 2u; var<workgroup> a: array<array<u32, n>, 2>;", .invalidType),
    ("override n = 2u; struct S { a: array<u32, n> }", .invalidType),
    ("override n = true; var<workgroup> a: array<u32, n>;", .typeMismatch),
])
func rejectsInvalidOverrideArrayDeclarations(source: String, code: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid override-sized array accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

@Test func checksSpecializedWorkgroupArrayCountAndReflectsStorageSize() throws {
    let source = ShaderInput.wgsl("""
        override n: i32 = 2;
        var<workgroup> data: array<u32, n>;
        @compute @workgroup_size(1) fn run() { data[1] = 1u; }
        """)
    let compiler = IndependentShaderCompiler()
    _ = try compiler.parseAndValidate(source)
    for count in [-1.0, 0.0, 1.0] {
        #expect(throws: ShaderFailure.self) {
            try compiler.compileCompute(source, to: .msl, constants: ["n": count])
        }
    }
    let shader = try compiler.compileCompute(source, to: .msl, constants: ["n": 4097])
    #expect(shader.workgroupStorageByteSize == 16_400)
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(.wgsl(overrideArrayShader), to: .wgsl, constants: ["": 2])
    }
}

@Test func preservesTypeIdentityForOverrideArraySyntax() throws {
    let source = """
        override n = 2u;
        var<workgroup> a: array<vec3f, n>;
        var<workgroup> b: array<vec3f, n>;
        var<workgroup> c: array<vec3f, (n)>;
        var<workgroup> d: array<vec3f, n + 0u>;
        """
    let globals = try IndependentShaderCompiler().parseAndValidate(.wgsl(source)).module.globalVariables.values
    let types = Dictionary(uniqueKeysWithValues: globals.compactMap { value in value.name.map { ($0, value.type) } })
    #expect(types["a"] == types["b"])
    #expect(types["a"] != types["c"])
    #expect(types["c"] != types["d"])
}

@Test func renderSpecializationIgnoresComputeOnlyArrayCounts() throws {
    let source = ShaderInput.wgsl("""
        override count: u32;
        override color = 0.5f;
        var<workgroup> scratch: array<u32, count>;
        @compute @workgroup_size(1) fn computeMain() { scratch[7u] = 1u; }
        @vertex fn vertexMain() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
        @fragment fn fragmentMain() -> @location(0) vec4f { return vec4f(color); }
        """)
    _ = try IndependentShaderCompiler().compileRenderStage(source, entryPoint: "fragmentMain", to: .glsl)
}

#if canImport(Metal)
@Test func evaluatesSpecializedWorkgroupArraysOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(overrideArrayShader)
    let first = try compiler.compileCompute(source, to: .msl)
    let second = try compiler.compileCompute(source, to: .msl, constants: ["4": 4])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let a = try await device.createComputePipeline(first)
        let b = try await device.createComputePipeline(second)
        let outputA = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let outputB = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: a, bindings: [.init(binding: 0): outputA], workgroups: .init(x: 1)),
            .dispatchCompute(pipeline: b, bindings: [.init(binding: 0): outputB], workgroups: .init(x: 1)),
            .copy(source: outputA, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
            .copy(source: outputB, destination: read, byteCount: 16), .read(buffer: read, byteCount: 16),
        ])
        #expect(try await ticket.wait() == [[31, 31, 0, 0], [31, 33, 35, 33]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
