import SAGABackend
import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

@Test func perVertexPipelineRequiresFeatureTrianglesAndMatchingElementType() throws {
    func stage(_ stage: ShaderStage, _ scalar: IRScalar, _ interpolation: IRInterpolation) -> RenderStageArtifact {
        .init(artifact: .text(format: .wgsl, source: ""), stage: stage, sourceEntryPoint: "main", entryPoint: "main",
              varyings: [.init(location: 0, componentCount: 2, scalar: scalar, interpolation: interpolation, sampling: .center)])
    }
    let vertex = stage(.vertex, .f32, .perspective)
    let fragment = stage(.fragment, .f32, .perVertex)
    let incompatible = stage(.fragment, .i32, .perVertex)
    let unsupported = try DeviceCapabilities(maxBufferSize: 4096)
    let supported = try DeviceCapabilities(maxBufferSize: 4096, features: [.shaderPerVertex])

    #expect(throws: GPUError.validation("Feature not advertised")) {
        try unsupported.validateInterstageInterfaces(vertex: vertex, fragment: fragment, topology: .triangleList)
    }
    for topology in [RenderPrimitiveState.Topology.triangleList, .triangleStrip] {
        try supported.validateInterstageInterfaces(vertex: vertex, fragment: fragment, topology: topology)
    }
    #expect(throws: GPUError.validation("Per-vertex fragment inputs require triangle topology")) {
        try supported.validateInterstageInterfaces(vertex: vertex, fragment: fragment, topology: .lineList)
    }
    #expect(throws: GPUError.validation("Fragment inputs do not match vertex outputs")) {
        try supported.validateInterstageInterfaces(vertex: vertex, fragment: incompatible, topology: .triangleList)
    }
}

#if canImport(Metal)
@Test func perVertexPipelineUsesTheMetalDeviceFeature() async throws {
    let source = ShaderInput.wgsl("""
        enable wgpu_per_vertex;
        struct Raster { @builtin(position) position: vec4f, @location(0) value: f32 }
        @vertex fn vs(@builtin(vertex_index) i: u32) -> Raster {
            let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
            return Raster(vec4f(points[i], 0, 1), f32(i + 1u));
        }
        @fragment fn fs(@location(0) @interpolate(per_vertex) values: array<f32, 3>) -> @location(0) vec4u {
            return vec4u(u32(values[0]), u32(values[1]), u32(values[2]), 255u);
        }
        """)
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    #expect(fragment.requiresPerVertex)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        if device.capabilities.features.contains(.shaderPerVertex) {
            let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint)
            let texture = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
                                                              usage: [.renderAttachment, .copySource]))
            let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
            let result = try await device.submitCommands([
                .render(pipeline: pipeline, target: texture, clear: .transparentBlack, vertexCount: 3),
                .copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256),
                .read(buffer: readback, byteCount: 4),
            ]).wait()
            #expect(result == [[0xff030201]])
        } else {
            await #expect(throws: GPUError.validation("Feature not advertised")) {
                try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint)
            }
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
