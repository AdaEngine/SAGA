import Testing
import SAGATypes

@Test func meshQueryPassValidatesAndPinsResources() throws {
    let pipeline = MeshPipeline(id: .init(device: 1, serial: 1), colorFormat: .rgba8Unorm)
    let target = ResourceID(device: 1, serial: 2)
    let occlusion = ResourceID(device: 1, serial: 3)
    let timestamp = ResourceID(device: 1, serial: 4)
    let draw = BufferCommand.drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1))
    let measured = BufferCommand.occlusionQueried(pass: .timestamped(pass: draw, writes: .init(querySet: timestamp, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1)), query: .init(querySet: occlusion, queryIndex: 0))
    let sets: [ResourceID: QuerySetDescriptor] = [occlusion: .init(type: .occlusion, count: 1), timestamp: .init(type: .timestamp, count: 2)]
    let textures: [ResourceID: TextureDescriptor] = [target: .init(width: 1, height: 1, format: .rgba8Unorm, usage: .renderAttachment)]
    #expect(Set(measured.resources) == [target, pipeline.id, occlusion, timestamp])
    try measured.validate(device: 1, descriptors: [:], textures: textures, querySets: sets)
    let invalid = BufferCommand.occlusionQueried(pass: draw, query: .init(querySet: occlusion, queryIndex: 1))
    #expect(throws: GPUError.self) { try invalid.validate(device: 1, descriptors: [:], textures: textures, querySets: sets) }
}

#if canImport(Metal)
import Metal
import SAGACompiler
import SAGAShaderIR

import SAGACore
import SAGAMetal

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalMeshQueriesResolveAndReset() async throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_mesh_shader;
        struct V { @builtin(position) position: vec4f, }
        struct P { @builtin(triangle_indices) indices: vec3u, }
        struct Output { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
        var<workgroup> output: Output;
        @mesh(output) @workgroup_size(1) fn mesh() {
          output.vertex_count = 3u; output.primitive_count = 1u;
          output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
          output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
          output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
          output.primitives[0].indices = vec3u(0u, 1u, 2u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let pipeline = try await device.createMeshPipeline(mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
    let color = try await device.createTexture(.init(width: 2, height: 2, format: .rgba8Unorm, usage: .renderAttachment))
    let query = try await device.createQuerySet(.init(type: .occlusion, count: 1))
    let result = try await device.createBuffer(.init(size: 264, usage: [.queryResolve, .copySource]))
    let readback = try await device.createBuffer(.init(size: 264, usage: [.copyDestination, .mapRead]))
    let timestamps = device.capabilities.features.contains(.timestampQuery) ? try await device.createQuerySet(.init(type: .timestamp, count: 2)) : nil
    func measured(_ count: UInt32) -> BufferCommand {
        var draw = BufferCommand.drawMesh(pipeline: pipeline, target: color, colorLoad: .clear(.transparentBlack), workgroups: .init(x: count))
        if let timestamps { draw = .timestamped(pass: draw, writes: .init(querySet: timestamps, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1)) }
        return .occlusionQueried(pass: draw, query: .init(querySet: query, queryIndex: 0))
    }
    let capture: [BufferCommand] = [.resolveQuerySet(querySet: query, firstQuery: 0, queryCount: 1, destination: result, destinationOffset: 256), .copy(source: result, destination: readback, byteCount: 264), .read(buffer: readback, byteCount: 264)]
    var commands = [measured(1)] + capture + [measured(0)] + capture
    if let timestamps { commands += [.resolveQuerySet(querySet: timestamps, firstQuery: 0, queryCount: 1, destination: result, destinationOffset: 256), .copy(source: result, destination: readback, byteCount: 264), .read(buffer: readback, byteCount: 264)] }
    let values = try await device.submitCommands(commands).wait()
    #expect(values[0][64] != 0 || values[0][65] != 0)
    #expect(values[1][64] == 0 && values[1][65] == 0)
    if timestamps != nil { #expect(values[2][64] != 0 || values[2][65] != 0) }
    // Query availability must survive retirement of the submission that wrote it.
    _ = try await device.submitCommands([measured(1)]).wait()
    let later = try await device.submitCommands(capture).wait()
    #expect(later[0][64] != 0 || later[0][65] != 0)
    if let timestamps {
        let resolved = try await device.submitCommands([
            .resolveQuerySet(querySet: timestamps, firstQuery: 0, queryCount: 1, destination: result, destinationOffset: 256),
            .copy(source: result, destination: readback, byteCount: 264),
            .read(buffer: readback, byteCount: 264),
        ]).wait()
        #expect(resolved[0][64] != 0 || resolved[0][65] != 0)
    }
    await device.close()
}
#endif
