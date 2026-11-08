import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let helperCompute = """
    @group(0) @binding(0) var<storage, read_write> data: array<u32, 4>;
    @compute @workgroup_size(2)
    fn run(@builtin(global_invocation_id) id: vec3u) {
        let once = fold(id.x);
        data[id.x] = once + once;
    }
    fn fold(i: u32) -> u32 {
        let pair = combine(bump(i), bump(i));
        put(i, 99u);
        return pair;
    }
    fn combine(a: u32, b: u32) -> u32 { return a * 10u + b; }
    fn bump(i: u32) -> u32 {
        let before = data[i];
        put(i, before + 1u);
        return before;
    }
    fn put(i: u32, value: u32) { data[i] = value; }
    fn unused(value: u32) -> u32 { return value / 3u; }
    """

private let helperRender = """
    struct Output {
        @builtin(position) position: vec4f,
        @location(0) color: vec4f,
    }
    @vertex fn v() -> Output { return make(vec4f(0.0, 0.0, 0.0, 1.0)); }
    fn make(position: vec4f) -> Output {
        var output: Output;
        output.position = position;
        output.color.r = 0.25;
        output.color.g = 0.5;
        output.color.b = 1.0;
        output.color.a = 1.0;
        return output;
    }
    @fragment fn f(@location(0) color: vec4f) -> @location(0) vec4f { return shade(color); }
    fn shade(color: vec4f) -> vec4f {
        var local = color;
        let before = local;
        local.r = 0.0;
        return vec4f(before.r, local.g, local.b, 1.0);
    }
    """

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsReachableHelpersWithOrderedCallsAndStorageCapture(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(helperCompute), to: format)
    guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader text"); return }
    #expect(source.contains("saga_call_"))
    #expect(!source.contains(" / "))
    #expect(shader.storageBindings.map(\.minimumByteSize) == [16])
    if format == .msl {
        #expect(source.contains("device uint* saga_buffer_0)"))
        #expect(source.contains(", saga_buffer_0);"))
        #expect(source.contains("(uint saga_param_0, uint saga_param_1) {"))
    }
    if format == .wgsl || format == .glsl {
        let input: ShaderInput = format == .wgsl ? .wgsl(source) : .glsl(source, stage: .compute, defines: [:])
        let checked = try compiler.parseAndValidate(input)
        #expect(checked.module.entryPoints.first?.stage == .compute)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsRenderHelpersWithVectorAndStageStructureResults(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    for (entry, stage) in [("v", ShaderStage.vertex), ("f", .fragment)] {
        let shader = try compiler.compileRenderStage(.wgsl(helperRender), entryPoint: entry, to: format)
        guard case .text(_, let source) = shader.artifact else { Issue.record("Expected shader text"); return }
        #expect(shader.stage == stage)
        #expect(source.contains("saga_call_"))
        if format == .wgsl || format == .glsl {
            let input: ShaderInput = format == .wgsl ? .wgsl(source) : .glsl(source, stage: stage, defines: [:])
            #expect(try compiler.parseAndValidate(input).module.entryPoints.first?.stage.rawValue == stage.rawValue)
        }
    }
}

@Test func rejectsRecursiveHelper() {
    let source = """
        fn foo(x: u32) -> u32 { return foo(x); }
        @compute @workgroup_size(1) fn run() { _ = foo(3u); }
        """
    #expect(throws: ShaderFailure.self) {
        try IndependentShaderCompiler().compileCompute(.wgsl(source), to: .msl)
    }
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsFixedArrayHelperResult(to format: ShaderOutputFormat) throws {
    let source = """
        fn pair(x: u32) -> array<u32, 2> { return array<u32, 2>(x, x + 1u); }
        @group(0) @binding(0) var<storage, read_write> result: array<u32, 2>;
        @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) index: u32) {
            result[index] = pair(3u)[index];
        }
        """
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(source), to: format)
    guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected source"); return }
    #expect(generated.contains("saga_call_"))
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(generated)) }
}

#if canImport(Metal)
@Test func executesStorageCapturingHelpersExactlyOnceOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(helperCompute), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let storage = try await device.createBuffer(.init(size: 24, usage: [.storage, .copyDestination, .copySource]))
        let read = try await device.createBuffer(.init(size: 24, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: storage, offset: 0, words: [0, 1, 2, .max, 555, 777]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): storage], workgroups: .init(x: 3)),
            .copy(source: storage, destination: read, byteCount: 24),
            .read(buffer: read, byteCount: 24),
        ])
        #expect(try await ticket.wait() == [[2, 24, 46, 4_294_967_276, 555, 777]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
