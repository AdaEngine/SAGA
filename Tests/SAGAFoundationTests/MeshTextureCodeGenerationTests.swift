import Testing
import SAGACompiler
import SAGAShaderIR

@Test func meshMetalSampledTexturesHaveStageLocalReflection() throws {
    let source = """
    enable wgpu_mesh_shader;
    struct V { @builtin(position) position: vec4f, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> out: O;
    @group(1) @binding(3) var image: texture_2d<f32>;
    @group(1) @binding(4) var smp: sampler;
    @mesh(out) @workgroup_size(1) fn main() {
      out.vertices[0].position = textureSampleLevel(image, smp, vec2f(0.5), 0.0);
      out.vertex_count = 3u; out.primitive_count = 1u; out.primitives[0].indices = vec3u(0u, 1u, 2u);
    }
    """
    let result = try IndependentShaderCompiler().compileReflectedMeshStage(.wgsl(source), entryPoint: "main", to: .msl)
    #expect(result.sampledTextures.count == 1)
    #expect(result.samplerBindings.count == 1)
    guard case .text(_, let text) = result.artifact else { Issue.record("Expected Metal source"); return }
    #expect(text.contains("[[texture(0)]]"))
    #expect(text.contains("[[sampler(0)]]"))
}

#if canImport(Metal)
import Metal
import SAGAMetal
import SAGACore
import SAGATypes

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) == true))
func metalTaskMeshSampledTexturesReadback() async throws {
    let source = ShaderInput.wgsl("""
    enable wgpu_mesh_shader;
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var smp: sampler;
    struct Payload { value: f32, }
    var<task_payload> payload: Payload;
    @task @payload(payload) @workgroup_size(1) fn task() -> @builtin(mesh_task_size) vec3u {
      payload.value = textureSampleLevel(image, smp, vec2f(0.5), 0.0).r;
      return vec3u(1u);
    }
    struct V { @builtin(position) position: vec4f, @location(0) value: f32, }
    struct P { @builtin(triangle_indices) indices: vec3u, }
    struct O { @builtin(vertices) vertices: array<V, 3>, @builtin(primitives) primitives: array<P, 1>, @builtin(vertex_count) vertex_count: u32, @builtin(primitive_count) primitive_count: u32, }
    var<workgroup> output: O;
    @mesh(output) @payload(payload) @workgroup_size(1) fn mesh() {
      let value = payload.value + textureSampleLevel(image, smp, vec2f(0.5), 0.0).r;
      output.vertex_count = 3u; output.primitive_count = 1u;
      output.vertices[0].position = vec4f(-1.0, -1.0, 0.0, 1.0);
      output.vertices[1].position = vec4f(3.0, -1.0, 0.0, 1.0);
      output.vertices[2].position = vec4f(-1.0, 3.0, 0.0, 1.0);
      output.vertices[0].value = value; output.vertices[1].value = value; output.vertices[2].value = value;
      output.primitives[0].indices = vec3u(0u, 1u, 2u);
    }
    """)
    let compiler = IndependentShaderCompiler()
    let task = try compiler.compileReflectedTaskStage(source, entryPoint: "task", to: .msl)
    let mesh = try compiler.compileReflectedMeshStage(source, entryPoint: "mesh", to: .msl)
    let fragment = try compiler.compileRenderStage(.wgsl("@fragment fn fs(@location(0) v: f32) -> @location(0) vec4f { return vec4f(v, 0.0, 0.0, 1.0); }"), entryPoint: "fs", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let pipeline = try await device.createMeshPipeline(task: task, mesh: mesh, fragment: fragment, colorFormat: .rgba8Unorm)
        let image = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.textureBinding, .copyDestination]))
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let sampler = try await device.createSampler()
        let upload = try await device.createBuffer(.init(size: 4, usage: [.copySource, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let slot = ComputeBindingSlot(binding: 0), samplerSlot = ComputeBindingSlot(binding: 1)
        let result = try await device.submitCommands([
            .write(buffer: upload, offset: 0, words: [0xff000040]),
            .copyBufferToTexture(source: upload, layout: .init(), destination: image, region: .init(width: 1, height: 1)),
            .drawMesh(pipeline: pipeline, target: target, colorLoad: .clear(.transparentBlack), workgroups: .init(x: 1), taskTextures: [slot: image], meshTextures: [slot: image], taskSamplers: [samplerSlot: sampler], meshSamplers: [samplerSlot: sampler]),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect((127...129).contains(try #require(result.first?.first) & 0xff))
    } catch { await device.close(); throw error }
    await device.close()
}
#endif
