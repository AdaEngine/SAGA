import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let callStatementShader = """
    @group(0) @binding(0) var<storage, read_write> out: array<u32, 4>;
    var<workgroup> cell: atomic<u32>;

    @compute @workgroup_size(1) fn run() {
        let value = required(3u);
        _ = required(4u);
        update(9u);
        for (mark(); out[0] < 23u; _ = required(2u)) { update(5u); }
        for (set(0u); out[2] < 2u; tick()) {}
        atomicStore(&cell, 1u);
        atomicLoad(&cell);
        atomicAdd(&cell, 4u);
        out[3] = value + atomicLoad(&cell);
    }
    @must_use fn required(delta: u32) -> u32 {
        update(delta);
        return delta * 2u;
    }
    fn update(delta: u32) -> u32 {
        out[0] += delta;
        return out[0];
    }
    fn mark() { out[1] = out[0]; }
    fn set(value: u32) { out[2] = value; }
    fn tick() { out[2] += 1u; }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func compilesExplicitResultDiscardAndVoidLoopCalls(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(callStatementShader), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader source"); return }
    #expect(!source.isEmpty)
    #expect(shader.storageBindings.count == 1)
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(source)) }
}

@Test(arguments: [
    ("required();", "required"),
    ("for (required(); false;) {}", "required"),
    ("for (; false; required()) {}", "required"),
    ("abs(-1i);", "abs"),
    ("vec2<u32>(1u);", "vec2"),
    ("bitcast<u32>(1i);", "bitcast"),
    ("workgroupUniformLoad(&cell);", "workgroupUniformLoad"),
])
func diagnosesIgnoredRequiredResultsAtTheCallee(statement: String, name: String) throws {
    let source = """
        var<workgroup> cell: u32;
        @compute @workgroup_size(1) fn run() { \(statement) }
        @must_use fn required() -> u32 { return 42u; }
        """
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Accepted a discarded required return value")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        let diagnostic = try #require(diagnostics.first)
        #expect(diagnostic.code == .unusedResult)
        let span = try #require(diagnostic.span)
        let bytes = Array(source.utf8)
        #expect(String(decoding: bytes[span.startUTF8..<span.endUTF8], as: UTF8.self) == name)
    }
}

@Test(arguments: [
    "@must_use fn f() {}",
    "@must_use() fn f() -> u32 { return 1u; }",
    "@must_use(1) fn f() -> u32 { return 1u; }",
    "@must_use @must_use fn f() -> u32 { return 1u; }",
    "@must_use const x = 1u;",
    "fn f() -> @must_use u32 { return 1u; }",
    "fn value() -> u32 { return 42u; } @compute @workgroup_size(1) fn run() { (value()); }",
    "fn value() -> u32 { return 42u; } @compute @workgroup_size(1) fn run() { value() + (1u); }",
    "fn tick() {} @compute @workgroup_size(1) fn run() { const tick = 1u; for (; false; tick()) {} }",
])
func rejectsInvalidMustUseDeclarationsAndNonCallStatements(source: String) {
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
}

#if canImport(Metal)
@Test func executesExplicitAndImplicitDiscardedCallsInOrderOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(callStatementShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let pipeline = try await device.createComputePipeline(shader)
    let output = try await device.createBuffer(.init(size: 16, usage: [.storage, .copySource]))
    let read = try await device.createBuffer(.init(size: 16, usage: [.mapRead, .copyDestination]))
    let snapshots = try await device.submitCommands([
        .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
        .copy(source: output, destination: read, byteCount: 16),
        .read(buffer: read, byteCount: 16),
    ]).wait()
    #expect(snapshots == [[23, 16, 2, 11]])
    await device.close()
}
#endif
