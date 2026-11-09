import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes
import Testing

#if canImport(Metal)
@Test func glslShadowGradientsSelectDifferentDepthMipsOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let clearShader = ShaderInput.wgsl("""
        @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
            let positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
            return vec4f(positions[index], 0.0, 1.0);
        }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0); }
        """)
    let sampleShader = ShaderInput.glsl("""
        #version 450
        layout(binding = 0) uniform texture2D image;
        layout(binding = 1) uniform samplerShadow state;
        layout(location = 0) out vec4 color;
        void main() {
            vec3 coords = vec3(0.25, 0.25, 0.5);
            float base = textureGrad(sampler2DShadow(image, state), coords, vec2(0.01, 0), vec2(0, 0.01));
            float next = textureGrad(sampler2DShadow(image, state), coords, vec2(0.5, 0), vec2(0, 0.5));
            float exact = textureLod(sampler2DShadow(image, state), coords, 1.0);
            color = vec4(base, next, exact, 1.0);
        }
        """, stage: .fragment, defines: [:])
    let vertex = try compiler.compileRenderStage(clearShader, entryPoint: "vs", to: .msl)
    let clearFragment = try compiler.compileRenderStage(clearShader, entryPoint: "fs", to: .msl)
    guard case .spirv(let words) = try compiler.compile(sampleShader, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let clearPipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: clearFragment, colorFormat: .rgba8Unorm, depthState: .init(compare: .always))
        let color = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource], mipLevelCount: 2))
        let depth = try await device.createTexture(.init(width: 4, height: 4, format: .depth32Float,
            usage: [.renderAttachment, .textureBinding], mipLevelCount: 2))
        let sampler = try await device.createSampler(.init(mipmapFilter: .nearest, compare: .less))
        let read = try await device.createBuffer(.init(size: 1024, usage: [.copyDestination, .mapRead]))
        let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        let initialization = try await device.submitCommands([
            .render(pipeline: clearPipeline, target: color, clear: clear, vertexCount: 0,
                    depth: .init(texture: depth, loadAction: .clear(0.25))),
            .render(pipeline: clearPipeline, target: color, clear: clear, vertexCount: 0, targetMipLevel: 1,
                    depth: .init(texture: depth, mipLevel: 1, loadAction: .clear(0.75))),
        ])
        _ = try await initialization.wait()
        for input in [sampleShader, .spirv(words)] {
            let fragment = try compiler.compileRenderStage(input, entryPoint: "main", to: .msl)
            let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
            let ticket = try await device.submitCommands([
                .render(pipeline: pipeline, target: color, clear: clear, vertexCount: 3,
                        textures: [.init(binding: 0): depth], samplers: [.init(binding: 1): sampler]),
                .copyTextureToBuffer(source: color, destination: read, bytesPerRow: 256),
                .read(buffer: read, byteCount: 4),
            ])
            #expect(try await ticket.wait() == [[0xffffff00]])
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
