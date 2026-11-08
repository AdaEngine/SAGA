import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let privateVariableShader = """
    struct State { samples: array<vec2u, 2>, counter: u32, }
    const seed = array(vec2u(3u, 7u), vec2u(11u, 13u));
    override base = 17u;
    var<private> state = State(seed, base);
    var<private> baseline: vec2u = vec2u(1u, 2u);
    var<private> ticks: u32;
    @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 8>;

    fn bump(index: i32, id: u32) -> u32 {
        let old = ticks;
        ticks += 1u;
        state.samples[index].y = 41u + id;
        state.counter += id;
        return old;
    }
    fn forward(index: i32, id: u32) -> u32 { return bump(index, id); }

    @compute @workgroup_size(4)
    fn run(@builtin(global_invocation_id) gid: vec3u) {
        let id = gid.x;
        let index = i32(id) - 1i;
        let initial = state.counter + baseline.x;
        let first = forward(index, id);
        let second = forward(index, id);
        output[id] = vec4u(initial + first * 100u, state.counter + second * 100u,
                          state.samples[index].y, state.samples[1].x);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesPrivateShaderVariables(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(privateVariableShader), to: format, constants: ["base": 29])
    #expect(shader.storageBindings.map(\.binding.binding) == [0])
    #expect(shader.workgroupStorageByteSize == 0)
    guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected source"); return }
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(generated)) }
    if format == .msl { #expect(generated.contains("thread uint& saga_private_")) }
}

private let privateRenderShader = """
    override red = 0.25f;
    var<private> state = vec4f(red, 0.0f, 0.0f, 1.0f);
    fn take() -> vec4f {
        state.x += 0.25f;
        return state;
    }
    @vertex fn vertexMain(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
        let reach = 1.0f + 4.0f * take().x;
        if id == 0u { return vec4f(-1.0f, -1.0f, 0.0f, 1.0f); }
        if id == 1u { return vec4f(reach, -1.0f, 0.0f, 1.0f); }
        return vec4f(-1.0f, reach, 0.0f, 1.0f);
    }
    @fragment fn fragmentMain() -> @location(0) vec4f {
        let color = take();
        return vec4f(select(0.0f, 1.0f, color.x == 0.75f), color.yzw);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesPrivateVariablesInRenderStages(format: ShaderOutputFormat) throws {
    let source = ShaderInput.wgsl(privateRenderShader)
    let compiler = IndependentShaderCompiler()
    for entry in ["vertexMain", "fragmentMain"] {
        let shader = try compiler.compileRenderStage(source, entryPoint: entry, to: format, constants: ["red": 0.5])
        guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected source"); return }
        if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(generated)) }
    }
}

@Test func onlyReachablePrivateInitializersRequireOverrides() throws {
    let source = ShaderInput.wgsl("""
        override needed: u32;
        var<private> unused: u32 = needed;
        var<private> active = 2u;
        @fragment fn a() -> @location(0) vec4f { return vec4f(f32(active)); }
        @fragment fn b() -> @location(0) vec4f { return vec4f(f32(unused)); }
        """)
    let compiler = IndependentShaderCompiler()
    _ = try compiler.compileRenderStage(source, entryPoint: "a", to: .wgsl)
    #expect(throws: ShaderFailure.self) { try compiler.compileRenderStage(source, entryPoint: "b", to: .wgsl) }
    _ = try compiler.compileRenderStage(source, entryPoint: "b", to: .wgsl, constants: ["needed": 7])
}

@Test func rejectsOutOfBoundsPrivateInitializerAtPipelineCreation() throws {
    let source = ShaderInput.wgsl("""
        override lane = 0u;
        var<private> value = array(10u, 20u)[lane];
        @compute @workgroup_size(1) fn run() { _ = value; }
        """)
    let compiler = IndependentShaderCompiler()
    _ = try compiler.compileCompute(source, to: .wgsl)
    #expect(throws: ShaderFailure.self) {
        try compiler.compileCompute(source, to: .wgsl, constants: ["lane": 2])
    }
}

@Test(arguments: [
    "var<private> seed: u32; var<private> invalid = seed;",
    "fn seed() -> u32 { return 1u; } var<private> invalid = seed();",
    "var<workgroup> invalid: u32 = 3u;",
])
func rejectsInvalidPrivateInitializers(source: String) {
    #expect(throws: ShaderFailure.self) { try IndependentShaderCompiler().parseAndValidate(.wgsl(source)) }
}

#if canImport(Metal)
@Test func privateShaderVariablesAreInvocationLocalOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(privateVariableShader), to: .msl,
                                                                 constants: ["base": 29])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let output = try await device.createBuffer(.init(size: 128, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 128, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 2)),
            .copy(source: output, destination: read, byteCount: 128), .read(buffer: read, byteCount: 128),
        ])
        #expect(try await ticket.wait() == [[
            30, 129, 0, 11, 30, 131, 42, 11, 30, 133, 43, 11, 30, 135, 0, 11,
            30, 137, 0, 11, 30, 139, 0, 11, 30, 141, 0, 11, 30, 143, 0, 11,
        ]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
@Test func privateVariablesReachBothRenderStagesOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(privateRenderShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vertexMain", to: .msl, constants: ["red": 0.5])
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fragmentMain", to: .msl, constants: ["red": 0.5])
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                           usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .render(pipeline: pipeline, target: target, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 512),
        ])
        let snapshots = try await ticket.wait()
        let pixels = try #require(snapshots.first)
        #expect([pixels[0], pixels[1], pixels[64], pixels[65]] == [0xff0000ff, 0xff0000ff, 0xff0000ff, 0xff0000ff])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
