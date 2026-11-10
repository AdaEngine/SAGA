#if canImport(Metal)
import Metal
import Testing
import SAGACompiler
import SAGAShaderIR

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalTaskDispatchLimitSuppressesInvalidMeshGrids() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let queue = try #require(device.makeCommandQueue())
    let compiler = IndependentShaderCompiler()
    let meshSource = """
    #include <metal_stdlib>
    using namespace metal;
    struct Vertex { float4 position [[position]]; };
    [[mesh]] void mesh_main(mesh<Vertex, void, 3, 1, topology::triangle> output,
                            device atomic_uint *counter [[buffer(0)]]) {
        atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        output.set_primitive_count(0);
    }
    fragment float4 fragment_main() { return float4(0.0); }
    """
    let options = MTLCompileOptions()
    options.languageVersion = .version3_0
    options.preprocessorMacros = ["SAGA_MESH_GRID_LIMIT": NSNumber(value: 4)]
    let meshLibrary = try device.makeLibrary(source: meshSource, options: options)
    let meshFunction = try #require(meshLibrary.makeFunction(name: "mesh_main"))
    let fragmentFunction = try #require(meshLibrary.makeFunction(name: "fragment_main"))
    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
    textureDescriptor.usage = .renderTarget
    let target = try #require(device.makeTexture(descriptor: textureDescriptor))
    let counter = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

    // Each invalid case is still small if the guard regresses; never submit an overflowing grid.
    for (grid, expected) in [("4u, 1u, 1u", UInt32(4)), ("5u, 1u, 1u", UInt32(0)), ("2u, 3u, 1u", UInt32(0))] {
        let source = """
        enable wgpu_mesh_shader;
        struct Payload { value: u32, }
        var<task_payload> payload: Payload;
        @task @payload(payload) @workgroup_size(1)
        fn task() -> @builtin(mesh_task_size) vec3u { return vec3u(\(grid)); }
        """
        let artifact = try compiler.compileReflectedTaskStage(.wgsl(source), entryPoint: "task", to: .msl)
        guard case .text(.msl, let msl) = artifact.artifact else { throw ShaderFailure.incompatibleArtifact }
        let taskLibrary = try device.makeLibrary(source: msl, options: options)
        let taskFunction = try #require(taskLibrary.makeFunction(name: artifact.entryPoint))
        let descriptor = MTLMeshRenderPipelineDescriptor()
        descriptor.objectFunction = taskFunction
        descriptor.meshFunction = meshFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.colorAttachments[0].pixelFormat = .rgba8Unorm
        let (pipeline, _) = try device.makeRenderPipelineState(descriptor: descriptor, options: [])
        counter.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        let command = try #require(queue.makeCommandBuffer())
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        let encoder = try #require(command.makeRenderCommandEncoder(descriptor: pass))
        encoder.setRenderPipelineState(pipeline)
        encoder.setMeshBuffer(counter, offset: 0, index: 0)
        encoder.drawMeshThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerObjectThreadgroup: MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerMeshThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        #expect(command.status == .completed, "\(String(describing: command.error))")
        #expect(counter.contents().load(as: UInt32.self) == expected, "grid \(grid)")
    }
}
#endif
