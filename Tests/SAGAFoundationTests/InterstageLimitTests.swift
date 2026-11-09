import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private func interstageLimitShader(location: UInt32) -> ShaderInput {
    .wgsl("""
        struct Raster {
            @builtin(position) position: vec4f,
            @location(\(location)) @interpolate(flat) color: vec4u,
        }
        @vertex fn vs(@builtin(vertex_index) i: u32) -> Raster {
            let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
            return Raster(vec4f(points[i], 0, 1), vec4u(19, 37, 73, 255));
        }
        @fragment fn fs(input: Raster) -> @location(0) vec4u { return input.color; }
        """)
}

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func interstageLimitsReflectHighLocationsBeforeDeviceSelection(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let source = interstageLimitShader(location: 30)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: format)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: format)
    #expect(vertex.varyings.map(\.location) == [30])
    #expect(fragment.varyings == vertex.varyings)
    if format == .wgsl {
        for shader in [vertex, fragment] {
            guard case .text(_, let text) = shader.artifact else { throw ShaderFailure.invalidHandle }
            _ = try compiler.parseAndValidate(.wgsl(text))
        }
    }
}

@Test func interstageLimitsAccountForTopologyClipDistancesAndFragmentBuiltins() async throws {
    let host = RenderTestBackend(capabilities: try .init(maxBufferSize: 4096,
        features: [.clipDistances], maxInterStageShaderVariables: 4))
    let device = try await Device.make(backend: host)
    let compiler = IndependentShaderCompiler()
    typealias Scenario = (locations: [UInt32], clips: Int, front: Bool, topology: RenderPrimitiveState.Topology, error: String?)
    let vertexError = "Vertex outputs exceed the inter-stage limit"
    let clipError = "Vertex outputs exceed the inter-stage limit with clip distances"
    let scenarios: [Scenario] = [
        ([0, 1, 2, 3], 0, false, .triangleList, nil),
        ([0, 1, 2, 3], 0, false, .pointList, vertexError),
        ([3], 0, false, .pointList, nil),
        ([4], 0, false, .triangleList, vertexError),
        ([2], 4, false, .triangleList, nil),
        ([2], 5, false, .triangleList, clipError),
        ([0, 1, 2], 4, false, .pointList, clipError),
        ([0, 1, 2, 3], 0, true, .triangleList, "Fragment inputs exceed the inter-stage limit"),
        ([0, 1, 2], 0, true, .triangleList, nil),
    ]
    do {
        for scenario in scenarios {
            let members = scenario.locations.map { "@location(\($0)) value\($0): f32," }.joined(separator: "\n")
            let clips = scenario.clips == 0 ? "" : "@builtin(clip_distances) planes: array<f32, \(scenario.clips)>,"
            let front = scenario.front ? "@builtin(front_facing) front: bool," : ""
            let enable = scenario.clips == 0 ? "" : "enable clip_distances;"
            let source = ShaderInput.wgsl("""
                \(enable)
                struct VertexOutput { @builtin(position) pos: vec4f, \(clips) \(members) }
                struct FragmentInput { @builtin(position) pos: vec4f, \(front) \(members) }
                @vertex fn vs() -> VertexOutput { return VertexOutput(); }
                @fragment fn fs(input: FragmentInput) -> @location(0) vec4f { return vec4f(1); }
                """)
            let vertex = try compiler.compileWebGPURenderStage(source, entryPoint: "vs")
            let fragment = try compiler.compileWebGPURenderStage(source, entryPoint: "fs")
            let primitive = RenderPrimitiveState(topology: scenario.topology)
            if let error = scenario.error {
                await #expect(throws: GPUError.validation(error)) {
                    try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                          primitive: primitive)
                }
            } else {
                _ = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                          primitive: primitive)
            }
        }
        #expect(await host.allocatedRenderPipelines == 4)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

#if canImport(Metal)
@Test func interstageLimitsRenderAtTheNativeMetalBoundary() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let compiler = IndependentShaderCompiler()
    let location = device.capabilities.maxInterStageShaderVariables - 1
    let source = interstageLimitShader(location: location)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    do {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Uint)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Uint,
                                                           usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let snapshots = try await device.submitCommands([
            .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(snapshots == [[0xff492513]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
