import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let nestedUniformDeclarations = """
    struct Pair { dims: vec2u, @size(8) tag: u32, }
    struct Cell { entries: array<Pair, 2>, transform: mat2x2f, }
    struct Settings { @size(16) flags: u32, cells: array<Cell, 2>, @size(16) tail: u32, }
    @group(0) @binding(0) var<uniform> settings: Settings;

    """

private let nestedUniformCompute = """
    @group(0) @binding(1) var<storage, read_write> output: array<vec4u, 4>;
    @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
        let outer = i32(lane) - 1i;
        let inner = select(1u, 2u, lane == 2u);
        let cell = settings.cells[outer];
        let entry = cell.entries[inner];
        output[lane] = vec4u(entry.dims, entry.tag + settings.flags, u32(cell.transform[1].y) + settings.tail);
    }
    """

private let nestedUniformRender = """
    @vertex fn vertexMain(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
        let reach = f32(settings.cells[0].entries[1].dims.x);
        if id == 0u { return vec4f(-1.0f, -1.0f, 0.0f, 1.0f); }
        if id == 1u { return vec4f(reach, -1.0f, 0.0f, 1.0f); }
        return vec4f(-1.0f, reach, 0.0f, 1.0f);
    }
    @fragment fn fragmentMain() -> @location(0) vec4f {
        let cell = settings.cells[1];
        let correct = cell.entries[0].dims.y == 8u && settings.tail == 9u;
        return select(vec4f(1.0f, 0.0f, 0.0f, 1.0f), vec4f(0.0f, 1.0f, 0.0f, 1.0f), correct);
    }
    """

private let nestedUniformInput: [UInt32] =
    [3, 901, 902, 903, 1, 2, 3, 904, 4, 5, 6, 905] + [Float(1), 2, 3, 4].map(\.bitPattern) +
    [7, 8, 9, 906, 10, 11, 12, 907] + [Float(5), 6, 7, 8].map(\.bitPattern) + [9, 908, 909, 910]

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesNestedUniformBuffers(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let compute = try compiler.compileCompute(.wgsl(nestedUniformDeclarations + nestedUniformCompute), to: format)
    #expect(compute.uniformBindings.map(\.minimumByteSize) == [128])
    let render = ShaderInput.wgsl(nestedUniformDeclarations + nestedUniformRender)
    let vertex = try compiler.compileRenderStage(render, entryPoint: "vertexMain", to: format)
    let fragment = try compiler.compileRenderStage(render, entryPoint: "fragmentMain", to: format)
    #expect(vertex.uniformBindings.map(\.minimumByteSize) == [128])
    #expect(fragment.uniformBindings.map(\.minimumByteSize) == [128])
    if format == .wgsl {
        for artifact in [compute.artifact, vertex.artifact, fragment.artifact] {
            guard case .text(_, let generated) = artifact else { Issue.record("Expected source"); return }
            _ = try compiler.parseAndValidate(.wgsl(generated))
        }
    }
}

#if canImport(Metal)
@Test func readsRangedNestedUniformBuffersOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let computeShader = try compiler.compileCompute(.wgsl(nestedUniformDeclarations + nestedUniformCompute), to: .msl)
    let renderShader = ShaderInput.wgsl(nestedUniformDeclarations + nestedUniformRender)
    let vertex = try compiler.compileRenderStage(renderShader, entryPoint: "vertexMain", to: .msl)
    let fragment = try compiler.compileRenderStage(renderShader, entryPoint: "fragmentMain", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let compute = try await device.createComputePipeline(computeShader)
        let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let uniform = try await device.createBuffer(.init(size: 384, usage: [.uniform, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let target = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm,
                                                           usage: [.renderAttachment, .copySource]))
        let read = try await device.createBuffer(.init(size: 512, usage: [.copyDestination, .mapRead]))
        let range = ComputeBufferBinding(buffer: uniform, offset: 256, size: 128)
        let ticket = try await device.submitCommands([
            .write(buffer: uniform, offset: 256, words: nestedUniformInput),
            .dispatchComputeRanges(pipeline: compute, bindings: [.init(binding: 0): range, .init(binding: 1): .init(buffer: output)],
                                   workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 64),
            .read(buffer: read, byteCount: 64),
            .render(pipeline: render, target: target, clear: .init(red: 0, green: 0, blue: 1, alpha: 1),
                    vertexCount: 3, uniforms: [.init(binding: 0): range]),
            .copyTextureToBuffer(source: target, destination: read, bytesPerRow: 256),
            .read(buffer: read, byteCount: 512),
        ])
        let snapshots = try await ticket.wait()
        try #require(snapshots.map(\.count) == [16, 128])
        #expect(snapshots[0] == [0, 0, 3, 9, 4, 5, 9, 13, 0, 0, 3, 17, 0, 0, 3, 9])
        let pixels = snapshots[1]
        #expect([pixels[0], pixels[1], pixels[64], pixels[65]] == [0xff00ff00, 0xff00ff00, 0xff00ff00, 0xff00ff00])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif

private let nestedHalfUniformShader = """
    enable f16;
    struct Packed { low: f16, high: f16, @size(12) tail: f16, }
    struct Settings { pairs: array<Packed, 2>, }
    @group(0) @binding(0) var<uniform> settings: Settings;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f, 2>;
    @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) lane: u32) {
        let packed = settings.pairs[lane];
        output[lane] = vec4f(f32(packed.low), f32(packed.high), f32(packed.tail), 0.0f);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesNestedHalfUniformBuffers(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(nestedHalfUniformShader), to: format)
    #expect(shader.requiresShaderF16)
    #expect(shader.uniformBindings.first?.minimumByteSize == 32)
    guard case .text(_, let generated) = shader.artifact else { Issue.record("Expected source"); return }
    if format == .wgsl {
        _ = try compiler.parseAndValidate(.wgsl(generated), capabilities: .referenceDefault.union(.shaderFloat16))
    }
    if format == .glsl { #expect(generated.contains("unpackFloat2x16(")) }
    if format == .hlsl { #expect(generated.contains("asfloat16(")) }
}

#if canImport(Metal)
@Test func readsNestedHalfUniformBuffersOnMetal() async throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl(nestedHalfUniformShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(shader)
        let uniform = try await device.createBuffer(.init(size: 32, usage: [.uniform, .copyDestination]))
        let output = try await device.createBuffer(.init(size: 32, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 32, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .write(buffer: uniform, offset: 0, words: [0xc0003c00, 0xdead4300, 701, 702, 0x4500c400, 0xbeef4600, 703, 704]),
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): uniform, .init(binding: 1): output],
                             workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 32), .read(buffer: read, byteCount: 32),
        ])
        let expected: [Float] = [1, -2, 3.5, 0, -4, 5, 6, 0]
        #expect(try await ticket.wait() == [expected.map(\.bitPattern)])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
