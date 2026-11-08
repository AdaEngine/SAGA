import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let localArrayShader = """
    struct Cell { data: array<vec2u, 2>, stamp: u32, }
    @group(0) @binding(0) var<storage, read_write> output: array<vec4u, 4>;

    fn update(outer: i32, inner: i32) -> vec4u {
        var cells: array<Cell, 2>;
        cells[0].data[0] = vec2u(3u, 5u);
        cells[1].data[1] = vec2u(7u, 11u);
        cells[0].stamp = 41u;
        cells[1].stamp = 101u;
        let before = cells;
        cells[outer].data[inner].y = 19u;
        cells[outer].stamp += 1u;
        var reset = 0u;
        for (var i = 0u; i < 3u; i += 1u) {
            var frame: array<u32, 2>;
            let slot = i & 1u;
            reset += frame[slot];
            frame[slot] = 123u;
        }
        return vec4u(before[outer].data[inner].y, cells[outer].data[inner].y,
                     cells[0].stamp + reset, cells[1].stamp);
    }

    @compute @workgroup_size(4)
    fn run(@builtin(local_invocation_index) id: u32) {
        let outer = i32(id) - 1i;
        let inner = select(1i, 2i, id == 1u);
        output[id] = update(outer, inner);
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func compilesMutableLocalArrays(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let artifact = try compiler.compileCompute(.wgsl(localArrayShader), to: format)
    guard case .text(_, let source) = artifact.artifact else { Issue.record("Expected source"); return }
    #expect(artifact.workgroupStorageByteSize == 0)
    #expect(source.contains("saga_local_"))
    if format == .wgsl { _ = try compiler.parseAndValidate(.wgsl(source)) }
}

private let largeLocalArrayShader = """
    @group(0) @binding(0) var<storage, read_write> result: array<u32, 2>;
    @compute @workgroup_size(2) fn run(@builtin(local_invocation_index) id: u32) {
        var scratch: array<u32, 4096>;
        scratch[id] = id + 1u;
        result[id] = scratch[id] + scratch[4095u - id];
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func initializesLargeLocalArraysWithBoundedSource(format: ShaderOutputFormat) throws {
    let artifact = try IndependentShaderCompiler().compileCompute(.wgsl(largeLocalArrayShader), to: format)
    guard case .text(_, let source) = artifact.artifact else { Issue.record("Expected source"); return }
    #expect(source.utf8.count < 8_192)
}

@Test func rejectsPipelineIndexOutsideMutableLocalArray() throws {
    let source = ShaderInput.wgsl("""
        override lane = 0i;
        @compute @workgroup_size(1) fn run() {
            var samples: array<u32, 2>;
            samples[lane] = 42u;
        }
        """)
    let compiler = IndependentShaderCompiler()
    _ = try compiler.compileCompute(source, to: .wgsl)
    for invalid in [-1.0, 2.0] {
        #expect(throws: ShaderFailure.self) {
            try compiler.compileCompute(source, to: .wgsl, constants: ["lane": invalid])
        }
    }
}

#if canImport(Metal)
@Test func readsAndWritesMutableLocalArraysOnMetal() async throws {
    let artifact = try IndependentShaderCompiler().compileCompute(.wgsl(localArrayShader), to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createComputePipeline(artifact)
        let output = try await device.createBuffer(.init(size: 64, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 64, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 1)),
            .copy(source: output, destination: read, byteCount: 64), .read(buffer: read, byteCount: 64),
        ])
        #expect(try await ticket.wait() == [[
            0, 0, 41, 101,
            0, 0, 42, 101,
            11, 19, 41, 102,
            0, 0, 41, 101,
        ]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
