import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let occlusionQueryShader = """
    @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f {
        return vec4f(point, 0.5, 1.0);
    }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }
    """

@Test func occlusionQueriesRequireAnOwnedRenderPassSlot() throws {
    let query = ResourceID(device: 1, serial: 1)
    let timestamp = ResourceID(device: 1, serial: 2)
    let foreign = ResourceID(device: 2, serial: 1)
    let color = ResourceID(device: 1, serial: 3)
    let pipeline = RenderPipeline(id: .init(device: 1, serial: 4), colorFormat: .rgba8Unorm)
    let draw = BufferCommand.render(pipeline: pipeline, target: color, clear: .transparentBlack, vertexCount: 3)
    let sets: [ResourceID: QuerySetDescriptor] = [
        query: .init(type: .occlusion, count: 2), timestamp: .init(type: .timestamp, count: 2),
        foreign: .init(type: .occlusion, count: 2),
    ]
    let textures: [ResourceID: TextureDescriptor] = [
        color: .init(width: 2, height: 2, format: .rgba8Unorm, usage: .renderAttachment),
    ]
    let measured = BufferCommand.occlusionQueried(pass: draw, query: .init(querySet: query, queryIndex: 1))
    let timed = BufferCommand.timestamped(pass: measured, writes: .init(querySet: timestamp, endOfPassWriteIndex: 1))
    #expect(Set(timed.resources) == [query, timestamp, color])
    try timed.validate(device: 1, descriptors: [:], textures: textures, querySets: sets)
    let invalid: [BufferCommand] = [
        .occlusionQueried(pass: .read(buffer: color, byteCount: 4), query: .init(querySet: query, queryIndex: 0)),
        .occlusionQueried(pass: draw, query: .init(querySet: query, queryIndex: 2)),
        .occlusionQueried(pass: draw, query: .init(querySet: timestamp, queryIndex: 0)),
        .occlusionQueried(pass: draw, query: .init(querySet: foreign, queryIndex: 0)),
        .occlusionQueried(pass: measured, query: .init(querySet: query, queryIndex: 0)),
        .occlusionQueried(pass: timed, query: .init(querySet: query, queryIndex: 0)),
        .timestamped(pass: measured, writes: .init(querySet: query, endOfPassWriteIndex: 0)),
    ]
    for command in invalid {
        #expect(throws: GPUError.self) {
            try command.validate(device: 1, descriptors: [:], textures: textures, querySets: sets)
        }
    }
}

#if canImport(Metal)
@Test func resolvesVisibleOccludedAndOverwrittenOcclusionSlotsOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl(occlusionQueryShader)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let render = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
        vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])],
        depthState: .init(compare: .less))
    let query = try await device.createQuerySet(.init(type: .occlusion, count: 4))
    let result = try await device.createBuffer(.init(size: 288, usage: [.queryResolve, .copyDestination, .copySource]))
    let readback = try await device.createBuffer(.init(size: 288, usage: [.copyDestination, .mapRead]))
    let vertices = try await device.createBuffer(.init(size: 24, usage: [.vertex, .copyDestination]))
    let indices = try await device.createBuffer(.init(size: 8, usage: [.index, .copyDestination]))
    let indirect = try await device.createBuffer(.init(size: 16, usage: [.indirect, .copyDestination]))
    let color = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
    let depth = try await device.createTexture(.init(width: 2, height: 2, format: .depth32Float, usage: .renderAttachment))
    let clear = RenderColorLoadAction.clear(.transparentBlack)
    let streams = [VertexBufferBinding(buffer: vertices)]
    let depthClear = RenderDepthAttachment(texture: depth, loadAction: .clear(1))
    let depthLoad = RenderDepthAttachment(texture: depth, loadAction: .load)
    let visible = BufferCommand.render(pipeline: render, target: color, colorLoad: clear, vertexCount: 3,
                                       vertexBuffers: streams, depth: depthClear)
    let occluded = BufferCommand.renderIndexed(pipeline: render, target: color, colorLoad: clear,
        indexBuffer: .init(buffer: indices, format: .uint16), draw: .init(indexCount: 3), vertexBuffers: streams, depth: depthLoad)
    let indirectDraw = BufferCommand.renderIndirect(pipeline: render, target: color, colorLoad: clear,
        indirectBuffer: indirect, vertexBuffers: streams, depth: depthClear)
    let empty = BufferCommand.render(pipeline: render, target: color, colorLoad: clear, vertexCount: 0,
                                     vertexBuffers: streams, depth: depthClear)
    let capture: [BufferCommand] = [
        .resolveQuerySet(querySet: query, firstQuery: 0, queryCount: 4, destination: result, destinationOffset: 256),
        .copy(source: result, destination: readback, byteCount: 288),
        .read(buffer: readback, byteCount: 288),
    ]
    let initial: [BufferCommand] = [
        .write(buffer: result, offset: 0, words: [0x11223344]),
        .write(buffer: result, offset: 256, words: Array(repeating: 0xffffffff, count: 8)),
        .write(buffer: vertices, offset: 0, words: [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)),
        .write(buffer: indices, offset: 0, words: [0x00010000, 2]),
        .write(buffer: indirect, offset: 0, words: [3, 1, 0, 0]),
    ]
    var queries: [BufferCommand] = [
        .occlusionQueried(pass: visible, query: .init(querySet: query, queryIndex: 1)),
        .occlusionQueried(pass: occluded, query: .init(querySet: query, queryIndex: 2)),
        .occlusionQueried(pass: indirectDraw, query: .init(querySet: query, queryIndex: 3)),
    ]
    var captureTimestamp: [BufferCommand] = []
    if device.capabilities.features.contains(.timestampQuery) {
        let timestamps = try await device.createQuerySet(.init(type: .timestamp, count: 1))
        queries[0] = .occlusionQueried(
            pass: .timestamped(pass: visible, writes: .init(querySet: timestamps, endOfPassWriteIndex: 0)),
            query: .init(querySet: query, queryIndex: 1))
        captureTimestamp = [
            .resolveQuerySet(querySet: timestamps, firstQuery: 0, queryCount: 1, destination: result, destinationOffset: 256),
            .copy(source: result, destination: readback, byteCount: 264),
            .read(buffer: readback, byteCount: 264),
        ]
    }
    let overwrite = BufferCommand.occlusionQueried(pass: empty, query: .init(querySet: query, queryIndex: 1))
    let commands = initial + capture + queries + capture + [overwrite] + capture + captureTimestamp
    let snapshots = try await device.submitCommands(commands).wait()
    if !captureTimestamp.isEmpty {
        let words = try #require(snapshots.last)
        let timestamp = UInt64(words[64]) | (UInt64(words[65]) << 32)
        #expect(timestamp > 0 && timestamp < UInt64.max)
    }
    let repeated = try await device.submitCommands(capture).wait()
    let occlusionSnapshots = snapshots.prefix(3)
    #expect(occlusionSnapshots.map { $0[0] } == [0x11223344, 0x11223344, 0x11223344])
    #expect(repeated == [try #require(occlusionSnapshots.last)])
    let visibleSlots = occlusionSnapshots.map { words in
        stride(from: 64, to: 72, by: 2).map { words[$0] != 0 || words[$0 + 1] != 0 }
    }
    #expect(visibleSlots == [[false, false, false, false], [false, true, false, true], [false, false, false, true]])
    await device.close()
}
#endif
