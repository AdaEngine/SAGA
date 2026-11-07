import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private let textureBarrierShader = """
    @group(2) @binding(0) var image: texture_storage_2d<r32uint, read_write>;
    @group(0) @binding(0) var<storage, read_write> output: array<u32, 384>;
    fn syncImage() { textureBarrier(); }
    fn rendezvous() { syncImage(); }
    @compute @workgroup_size(128)
    fn run(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
        let start = group.x * 128u;
        let index = start + lane;
        textureStore(image, vec2u(index, 0u), vec4u(group.x * 1000u + lane + 10u));
        rendezvous();
        let neighbor = start + (lane + 64u) % 128u;
        output[index] = textureLoad(image, vec2u(neighbor, 0u)).r + 7u;
    }
    """

@Test(arguments: [ShaderOutputFormat.wgsl, .msl, .hlsl, .glsl])
func emitsTextureBarrierWithImageMemorySemantics(format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let shader = try compiler.compileCompute(.wgsl(textureBarrierShader), to: format)
    #expect(shader.usesTextureBarrier)
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected source"); return }
    switch format {
    case .wgsl:
        #expect(body.contains("textureBarrier();"))
        _ = try compiler.parseAndValidate(.wgsl(body))
    case .msl: #expect(body.contains("threadgroup_barrier(mem_flags::mem_texture);"))
    case .hlsl: #expect(body.contains("DeviceMemoryBarrierWithGroupSync();"))
    case .glsl:
        #expect(body.contains("memoryBarrierImage();\n    barrier();"))
        #expect(body.contains("coherent uniform uimage2D"))
        #expect(!body.contains("coherent buffer"))
    default: Issue.record("Unexpected target")
    }
}

@Test func unreachableTextureBarrierDoesNotQualifyGLSLImages() throws {
    let shader = try IndependentShaderCompiler().compileCompute(.wgsl("""
        @group(0) @binding(0) var image: texture_storage_2d<r32uint, write>;
        fn unused() { textureBarrier(); }
        @compute @workgroup_size(1) fn run() { textureStore(image, vec2u(0), vec4u(1)); }
        """), to: .glsl)
    #expect(!shader.usesTextureBarrier)
    guard case .text(_, let body) = shader.artifact else { Issue.record("Expected source"); return }
    #expect(!body.contains("coherent"))
    #expect(!body.contains("memoryBarrierImage"))
}

@Test(arguments: [
    ("@fragment fn run() -> @location(0) vec4f { sync(); return vec4f(1); }", ShaderDiagnosticCode.invalidControlFlow),
    ("""
     @compute @workgroup_size(4) fn run(@builtin(local_invocation_index) lane: u32) {
         if lane != 0u { sync(); }
     }
     """, ShaderDiagnosticCode.nonUniformControlFlow),
])
func rejectsTextureBarriersOutsideUniformComputeControl(entry: String, expected: ShaderDiagnosticCode) throws {
    let source = "fn barrierHelper() { textureBarrier(); }\nfn sync() { barrierHelper(); }\n" + entry
    do {
        _ = try IndependentShaderCompiler().parseAndValidate(.wgsl(source))
        Issue.record("Invalid texture barrier accepted")
    } catch ShaderFailure.diagnostics(let diagnostics) {
        #expect(diagnostics.first?.code == expected)
    }
}

#if canImport(Metal)
@Test func textureBarrierMakesNeighborTexelWritesVisibleOnMetal() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let shader = try IndependentShaderCompiler().compileCompute(.wgsl(textureBarrierShader), to: .msl)
        let pipeline = try await device.createComputePipeline(shader)
        let image = try await device.createTexture(.init(width: 384, height: 1, format: .r32Uint,
            usage: [.copySource, .storageBinding]))
        let output = try await device.createBuffer(.init(size: 1536, usage: [.storage, .copySource]))
        let read = try await device.createBuffer(.init(size: 1536, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .dispatchCompute(pipeline: pipeline, bindings: [.init(binding: 0): output], workgroups: .init(x: 3),
                             textures: [.init(group: 2, binding: 0): image]),
            .copy(source: output, destination: read, byteCount: 1536),
            .read(buffer: read, byteCount: 1536),
            .copyTextureToBuffer(source: image, destination: read, bytesPerRow: 1536),
            .read(buffer: read, byteCount: 1536),
        ])
        let writes = (0..<384).map { (index: Int) -> UInt32 in
            let group = UInt32(index / 128)
            let lane = UInt32(index % 128)
            return group * 1000 + lane + 10
        }
        let neighbors = (0..<384).map { (index: Int) -> UInt32 in
            let start = index / 128 * 128
            let rotated = (index + 64) % 128
            return writes[start + rotated] + 7
        }
        let expected = [neighbors, writes]
        #expect(try await ticket.wait() == expected)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
