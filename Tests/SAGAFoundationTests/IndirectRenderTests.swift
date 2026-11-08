import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

#if canImport(Metal)
@Test(arguments: [IndexFormat.uint16, .uint32])
func gpuAuthoredDrawsReachMetalWithFirstInstanceAndSignedBaseVertex(format: IndexFormat) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.indirectFirstInstance])
    let compiler = IndependentShaderCompiler()
    let producer = try await device.createComputePipeline(compiler.compileCompute(.wgsl("""
        @group(0) @binding(0) var<storage, read_write> args: array<u32, 29>;
        @compute @workgroup_size(1)
        fn prepare() {
            args[1] = 3u; args[2] = 1u; args[3] = 1u; args[4] = 1u;
            args[5] = 3u; args[6] = 1u; args[7] = 1u; args[8] = 0xffffffffu; args[9] = 1u;
            args[10] = 3u; args[11] = 1u; args[12] = 2u; args[13] = 0xffffffffu; args[14] = 1u;
            args[15] = 3u; args[16] = 0u; args[17] = 1u; args[18] = 1u;
            args[19] = 3u; args[20] = 1u; args[21] = 2u; args[22] = 1u;
            args[23] = 3u; args[24] = 1u; args[25] = 1u; args[26] = 2u;
            args[27] = 0xabcdefu; args[28] = 0x123456u;
        }
        """), to: .msl))
    let source = ShaderInput.wgsl("""
        struct Raster { @builtin(position) position: vec4f, @location(0) tint: f32, }
        @vertex fn vs(@location(0) position: vec2f, @location(1) tint: f32) -> Raster {
            return Raster(vec4f(position, 0.0, 1.0), tint);
        }
        @fragment fn fs(in: Raster) -> @location(0) vec4f {
            return vec4f(1.0 - in.tint, in.tint, 0.0, 1.0);
        }
        """)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [
            .init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)]),
            .init(stride: 4, stepMode: .instance, attributes: [.init(location: 1, offset: 0, format: .float32)]),
        ])
    let arguments = try await device.createBuffer(.init(size: 116, usage: [.indirect, .storage, .copySource]))
    let wrongUsage = try await device.createBuffer(.init(size: 20, usage: .storage))
    let positions = try await device.createBuffer(.init(size: 32, usage: [.vertex, .copyDestination]))
    let instances = try await device.createBuffer(.init(size: 8, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 24, usage: [.index, .copyDestination]))
    let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm,
                                                       usage: [.renderAttachment, .copySource]))
    let readback = try await device.createBuffer(.init(size: 256, usage: [.copyDestination, .mapRead]))
    let index = IndexBufferBinding(buffer: indices, format: format, offset: UInt64(format.byteSize),
                                   size: 4 * UInt64(format.byteSize))
    let streams = [VertexBufferBinding(buffer: positions), VertexBufferBinding(buffer: instances)]
    let clear = RenderColorLoadAction.clear(.init(red: 0, green: 0, blue: 1, alpha: 1))
    func draw(_ offset: UInt64, index: IndexBufferBinding? = nil, buffer: ResourceID? = nil) -> BufferCommand {
        .renderIndirect(pipeline: pipeline, target: target, colorLoad: clear, indirectBuffer: buffer ?? arguments,
                        indirectOffset: offset, indexBuffer: index, vertexBuffers: streams)
    }
    for invalid in [draw(2), draw(104), draw(100, index: index), draw(0, buffer: wrongUsage),
                    draw(20, index: .init(buffer: indices, format: format, offset: 1))] {
        await #expect(throws: GPUError.self) { try await device.submitCommands([invalid]) }
    }
    let storageFragment = try compiler.compileRenderStage(.wgsl("""
        @group(0) @binding(0) var<storage, read_write> feedback: u32;
        @fragment fn fs() -> @location(0) vec4f {
            feedback = 1u;
            return vec4f(0.0, 1.0, 0.0, 1.0);
        }
        """), entryPoint: "fs", to: .msl)
    let storagePipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: storageFragment, colorFormat: .rgba8Unorm, vertexBuffers: pipeline.vertexBuffers)
    let aliased = try await device.createBuffer(.init(size: 512, usage: [.storage, .indirect]))
    await #expect(throws: GPUError.validation("Aliased writable render buffers")) {
        try await device.submitCommands([
            .renderIndirect(pipeline: storagePipeline, target: target, colorLoad: clear, indirectBuffer: aliased,
                            vertexBuffers: streams, storageBuffers: [.init(binding: 0): .init(buffer: aliased, offset: 256)]),
        ])
    }
    let indexWords: [UInt32] = format == .uint16 ? [0x00630063, 0x00030002, 0x00630004] : [99, 99, 2, 3, 4, 99]
    let capture: [BufferCommand] = [
        .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
        .read(buffer: readback, byteCount: 4),
    ]
    let prepare: [BufferCommand] = [
        .write(buffer: positions, offset: 0, words: [Float(4), 4, -1, -1, 3, -1, -1, 3].map(\.bitPattern)),
        .write(buffer: instances, offset: 0, words: [0, Float(1).bitPattern]),
        .write(buffer: indices, offset: 0, words: indexWords),
        .dispatchCompute(pipeline: producer, bindings: [.init(binding: 0): arguments], workgroups: .init(x: 1)),
    ]
    let ticket = try await device.submitCommands(prepare + [
        draw(4), draw(20, index: index), draw(40, index: index), draw(60), draw(76), draw(92),
        // All three indices fit the binding, but the first fetch would be out of range.
        draw(20, index: .init(buffer: indices, format: format, size: 5 * UInt64(format.byteSize))),
        draw(20, index: .init(buffer: indices, format: format, offset: 24, size: 0)),
    ].flatMap { [$0] + capture } + [
        .copy(source: arguments, destination: readback, byteCount: 116),
        .read(buffer: readback, byteCount: 116),
    ])
    let words = try await ticket.wait()
    let green: UInt32 = 0xff00ff00
    let blue: UInt32 = 0xffff0000
    #expect(words.dropLast().flatMap { $0 } == [green, green, blue, blue, blue, blue, blue, blue])
    #expect(words.last == [0, 3, 1, 1, 1, 3, 1, 1, .max, 1, 3, 1, 2, .max, 1,
                           3, 0, 1, 1, 3, 1, 2, 1, 3, 1, 1, 2, 0xabcdef, 0x123456])
    await device.close()
}
#endif
