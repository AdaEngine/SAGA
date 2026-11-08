import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let overrideShader = """
    @id(7) override width = 2u;
    override offset = base;
    override base = 3u;
    override addLane: bool = true;
    override unused: u32;
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 4>;
    fn ignored() -> u32 { return unused; }
    fn bias() -> u32 { return offset; }
    @compute @workgroup_size(width)
    fn run(@builtin(local_invocation_index) i: u32) {
        let initial = bias();
        output[i] = select(initial, initial + i, addLane);
    }
    """

@Test func validatesTypedOverridesBeforeSpecialization() throws {
    let module = try IndependentShaderCompiler().parseAndValidate(.wgsl(overrideShader)).module
    #expect(Set(module.overrides.values.compactMap(\.name)) == ["width", "base", "offset", "addLane", "unused"])
    let width = try #require(module.overrides.values.first { $0.name == "width" })
    #expect(width.id == 7)
    #expect(try module.types.value(for: width.type).kind == .scalar(.u32))
    let offset = try #require(module.overrides.values.first { $0.name == "offset" })
    let initializer = try #require(offset.initializer)
    guard case .override(let base) = try module.globalExpressions.value(for: initializer).kind else {
        Issue.record("Expected typed dependency for offset")
        return
    }
    #expect(try module.overrides.value(for: base).name == "base")
    #expect(module.entryPoints.first?.workgroupOverrides?.x != nil)
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func specializesDefaultOverridesBeforeBackendLowering(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(overrideShader), to: format)
    #expect(shader.workgroupSize == .init(x: 2))
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(!source.contains("override "))
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(source)) }
}

@Test(arguments: [
    ("override x;", ShaderDiagnosticCode.typeMismatch),
    ("override x: vec2f = vec2f(1);", .typeMismatch),
    ("@id(3) override x = 1u; @id(3) override y = 2u;", .duplicateName),
    ("@id(65536) override x = 1u;", .invalidBinding),
    ("override a = b; override b = a;", .invalidReference),
    ("override x = 1u; const y = x;", .typeMismatch),
    ("override x = 1.0f; @compute @workgroup_size(x) fn run() {}", .typeMismatch),
    ("override x = 1i; @compute @workgroup_size(x, 1u) fn run() {}", .typeMismatch),
])
func diagnosesInvalidOverrideDeclarations(source: String, code: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid override declaration accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func specializesPipelineValuesByNameAndNumericID(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(overrideShader), to: format,
                                             constants: ["7": 4.75, "base": 8, "addLane": 0, "unused": 1])
    #expect(shader.workgroupSize == .init(x: 4))
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(!source.contains("override "))
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(source)) }
}

@Test(arguments: [
    (["width": 4.0], ShaderDiagnosticCode.unresolvedName),
    (["7": 0.0], .invalidBinding),
    (["base": -1.0], .invalidType),
    (["addLane": Double.infinity], .invalidType),
])
func rejectsInvalidPipelineConstants(constants: [String: Double], code: ShaderDiagnosticCode) throws {
    do {
        _ = try IndependentShaderCompiler().compileCompute(.wgsl(overrideShader), to: .msl, constants: constants)
        Issue.record("Invalid pipeline constants accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == code)
    }
}

@Test func requiresMissingOverrideOnlyInSelectedStage() throws {
    let source = ShaderInput.wgsl("""
        override width: f32;
        override color: f32;
        @vertex fn vertexMain() -> @builtin(position) vec4f { return vec4f(width, 0, 0, 1); }
        @fragment fn fragmentMain() -> @location(0) vec4f { return vec4f(color); }
        """)
    let compiler = IndependentShaderCompiler()
    _ = try compiler.compileRenderStage(source, entryPoint: "vertexMain", to: .msl, constants: ["width": 0.25])
    _ = try compiler.compileRenderStage(source, entryPoint: "fragmentMain", to: .wgsl, constants: ["color": 1])
    do {
        _ = try compiler.compileRenderStage(source, entryPoint: "fragmentMain", to: .wgsl)
        Issue.record("Missing required fragment override accepted")
    } catch ShaderFailure.diagnostics(let issues) {
        #expect(issues.first?.code == .invalidReference)
    }
}

#if canImport(Metal)
@Test func specializesTwoIndependentOverridePipelinesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(overrideShader)
    let first = try compiler.compileCompute(source, to: .msl)
    let second = try compiler.compileCompute(source, to: .msl, constants: ["7": 4, "base": 8, "addLane": 0])
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
        #expect(try await ticket.wait() == [[3, 4, 0, 0], [8, 8, 8, 8]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
