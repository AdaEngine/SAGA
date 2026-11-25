#if os(Windows)
import Testing
import SAGATypes
import SAGACore
import SAGAShaderIR
import SAGACompiler
import SAGAD3D12

private let attachmentShader = ShaderInput.wgsl("""
    @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
        return vec4f(points[index], 0.5, 1);
    }
    struct ColorOutput {
        @location(0) first: vec4f,
        @location(1) second: vec4f,
    }
    @fragment fn fs() -> ColorOutput {
        return ColorOutput(vec4f(1, 0, 0, 1), vec4f(0, 0, 1, 1));
    }
    """)

@Test(.enabled(if: d3d12Available), .gpuExclusive)
func d3d12UploadPagesKeepEarlierCopySourcesAlive() async throws {
    let device = try await Device.make(backend: D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories))
    do {
        let source = try await device.createBuffer(.init(size: 65_536, usage: [.copySource, .copyDestination]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let first = [UInt8](repeating: 0x11, count: 32_768)
        let second = [UInt8](repeating: 0x22, count: 32_768)
        let overlap = [UInt8](repeating: 0x33, count: 32_768)
        let read: BufferCommand = .read(buffer: readback, byteCount: 4)
        let result = try await device.submitCommands([
            .writeBytes(buffer: source, offset: 0, bytes: first),
            .writeBytes(buffer: source, offset: 32_768, bytes: second),
            .copy(source: source, sourceOffset: 32_764, destination: readback, byteCount: 4), read,
            .writeBytes(buffer: source, offset: 32_764, bytes: overlap),
            .copy(source: source, sourceOffset: 32_764, destination: readback, byteCount: 4), read,
            .copy(source: source, sourceOffset: 65_532, destination: readback, byteCount: 4), read,
        ]).wait()
        #expect(result == [[0x11111111], [0x33333333], [0x22222222]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test(.enabled(if: d3d12Available), .gpuExclusive)
func d3d12DrawsPackedVerticesAndIndicesAfterReusingCommandResources() async throws {
    let device = try await Device.make(backend: D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories))
    do {
        let compiler = IndependentShaderCompiler()
        let shader = ShaderInput.wgsl("""
            @vertex fn vs(@location(0) point: vec2f) -> @builtin(position) vec4f { return vec4f(point, 0, 1); }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
            """)
        let vertex = try compiler.compileRenderStage(shader, entryPoint: "vs", to: .hlsl)
        let fragment = try compiler.compileRenderStage(shader, entryPoint: "fs", to: .hlsl)
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
            vertexBuffers: [.init(stride: 8, attributes: [.init(location: 0, offset: 0, format: .float32x2)])]
        )
        let packed = try await device.createBuffer(.init(size: 36, usage: [.vertex, .index, .copyDestination]))
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let vertices = [Float(-1), -1, 3, -1, -1, 3].map(\.bitPattern)
        let indexOrders: [(indices: [UInt32], pixel: UInt32)] = [([0, 1, 2], 0xff00ff00), ([0, 0, 0], 0)]
        for (indices, pixel) in indexOrders {
            let upload = try await device.submitCommands([.write(buffer: packed, offset: 0, words: vertices + indices)])
            let draw = try await device.submitCommands([
                .renderIndexed(pipeline: pipeline, target: target, clear: .transparentBlack,
                               indexBuffer: .init(buffer: packed, format: .uint32, offset: 24, size: 12),
                               draw: .init(indexCount: 3), vertexBuffers: [.init(buffer: packed)]),
                .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
                .read(buffer: readback, byteCount: 4),
            ])
            #expect(try await draw.wait() == [[pixel]])
            _ = try await upload.wait()
        }
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test(.enabled(if: d3d12Available), .gpuExclusive)
func d3d12KeepsSamplingAfterACommandListFillsItsFirstDescriptorHeaps() async throws {
    let device = try await Device.make(backend: D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories))
    do {
        let compiler = IndependentShaderCompiler()
        let vertex = try compiler.compileRenderStage(attachmentShader, entryPoint: "vs", to: .hlsl)
        let fragment = try compiler.compileRenderStage(.wgsl("""
            @group(0) @binding(0) var image: texture_2d<f32>;
            @group(0) @binding(1) var imageSampler: sampler;
            @fragment fn fs() -> @location(0) vec4f {
                return textureSample(image, imageSampler, vec2f(0.5));
            }
            """), entryPoint: "fs", to: .hlsl)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let input = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: [.textureBinding, .copyDestination])
        let output = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource])
        let green = try await device.createTexture(input)
        let red = try await device.createTexture(input)
        let first = try await device.createTexture(output)
        let second = try await device.createTexture(output)
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        func draw(_ image: ResourceID, to target: ResourceID, using sampler: ResourceID) -> BufferCommand {
            .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3,
                    textures: [.init(binding: 0): image], samplers: [.init(binding: 1): sampler])
        }
        var greenDraws: [BufferCommand] = []
        for _ in 0..<512 {
            let sampler = try await device.createSampler(.init())
            greenDraws.append(draw(green, to: first, using: sampler))
        }
        let finalSampler = try await device.createSampler(.init())
        let result = try await device.submitCommands([
            .writeTexture(destination: green, data: [0, 255, 0, 255], region: .init(width: 1, height: 1)),
            .writeTexture(destination: red, data: [255, 0, 0, 255], region: .init(width: 1, height: 1)),
        ] + greenDraws + [
            draw(red, to: second, using: finalSampler),
            .copyTextureToBuffer(source: first, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
            .copyTextureToBuffer(source: second, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[0xff00ff00], [0xff0000ff]])

        // A later recording can recycle the completed heap. Bind red first so the old green
        // table's offset would point at the wrong image if the previous cache survived the reset.
        let compute = try await device.createComputePipeline(
            compiler.compileCompute(.wgsl("@compute @workgroup_size(1) fn main() {}"), to: .hlsl)
        )
        let reused = try await device.submitCommands([
            draw(red, to: first, using: finalSampler),
            .dispatchCompute(pipeline: compute, bindings: [:], workgroups: .init(x: 1)),
            greenDraws[0],
            .copyTextureToBuffer(source: first, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
            .writeTexture(destination: green, data: [0, 0, 255, 255], region: .init(width: 1, height: 1)),
            draw(green, to: second, using: finalSampler),
            .copyTextureToBuffer(source: second, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(reused == [[0xff00ff00], [0xffff0000]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test(.enabled(if: d3d12Available), .gpuExclusive)
func d3d12LinksAFragmentThatReadsOnlyOneVertexVarying() async throws {
    let device = try await Device.make(backend: D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories))
    do {
        let compiler = IndependentShaderCompiler()
        let vertex = try compiler.compileRenderStage(.wgsl("""
            struct Output {
                @builtin(position) position: vec4f,
                @location(0) unusedColor: vec4f,
                @location(1) uv: vec2f,
            }
            @vertex fn vs(@builtin(vertex_index) index: u32) -> Output {
                let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
                return Output(vec4f(points[index], 0.5, 1), vec4f(0, 1, 0, 1), vec2f(1, 0));
            }
            """), entryPoint: "vs", to: .hlsl)
        let fragment = try compiler.compileRenderStage(.wgsl("""
            @fragment fn fs(@location(1) uv: vec2f) -> @location(0) vec4f {
                return vec4f(uv, 0, 1);
            }
            """), entryPoint: "fs", to: .hlsl)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let target = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .render(pipeline: pipeline, target: target, clear: .transparentBlack, vertexCount: 3),
            .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[0xff0000ff]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test(.enabled(if: d3d12Available), .gpuExclusive)
func d3d12MultipleColorAttachmentsRespectIndependentWriteMasks() async throws {
    let device = try await Device.make(backend: D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories))
    do {
        let compiler = IndependentShaderCompiler()
        let vertex = try compiler.compileRenderStage(attachmentShader, entryPoint: "vs", to: .hlsl)
        let fragment = try compiler.compileRenderStage(attachmentShader, entryPoint: "fs", to: .hlsl)
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
            additionalColorTargets: [1: .init(format: .rgba8Unorm, colorState: .init(writeMask: .blue))]
        )
        let descriptor = TextureDescriptor(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource])
        let first = try await device.createTexture(descriptor)
        let second = try await device.createTexture(descriptor)
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let draw = BufferCommand.render(
            pipeline: pipeline, target: first, clear: .transparentBlack, vertexCount: 3,
            additionalColors: [1: .init(texture: second, loadAction: .clear(.init(red: 0, green: 1, blue: 0, alpha: 1)))])
        // A single submission with more than 64 attachment descriptors must still render.
        let result = try await device.submitCommands(Array(repeating: draw, count: 40) + [
            .copyTextureToBuffer(source: first, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
            .copyTextureToBuffer(source: second, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[0xff0000ff], [0xffffff00]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test(.enabled(if: d3d12Available), .gpuExclusive)
func d3d12DepthOnlyPipelineWritesDepthWithoutAColorTarget() async throws {
    let device = try await Device.make(backend: D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories))
    do {
        let vertex = try IndependentShaderCompiler().compileRenderStage(attachmentShader, entryPoint: "vs", to: .hlsl)
        let pipeline = try await device.createRenderPipeline(
            vertex: vertex, fragment: nil, colorFormat: nil,
            depthState: .init(format: .depth32Float, compare: .always, writeEnabled: true)
        )
        let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float, usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 4, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .render(pipeline: pipeline, target: nil, colorLoad: .load, vertexCount: 0, depth: .init(texture: depth, loadAction: .clear(0.75))),
            .copyTextureToBuffer(source: depth, destination: readback, bytesPerRow: 256, aspect: .depthOnly),
            .read(buffer: readback, byteCount: 4),
            .render(pipeline: pipeline, target: nil, colorLoad: .load, vertexCount: 3, depth: .init(texture: depth, loadAction: .load)),
            .copyTextureToBuffer(source: depth, destination: readback, bytesPerRow: 256, aspect: .depthOnly),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[Float(0.75).bitPattern], [Float(0.5).bitPattern]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

@Test(.enabled(if: d3d12Available), .gpuExclusive)
func d3d12DrawsToTheSixColorTargetsUsedByStudio() async throws {
    let device = try await Device.make(backend: D3D12Backend.makeDevice(shaderCompilerDirectories: dxcDirectories))
    do {
        let compiler = IndependentShaderCompiler()
        let vertex = try compiler.compileRenderStage(attachmentShader, entryPoint: "vs", to: .hlsl)
        let fragment = try compiler.compileRenderStage(.wgsl("""
            struct Output {
                @location(0) input0: vec4f,
                @location(1) input1: vec4f,
                @location(2) input2: vec4f,
                @location(3) input3: vec4f,
                @location(4) input4: vec4f,
                @location(5) input5: vec4f,
            }
            @fragment fn fs() -> Output {
                let value = vec4f(1, 0, 0, 1);
                return Output(value, value, value, value, value, vec4f(0, 1, 0, 1));
            }
            """), entryPoint: "fs", to: .hlsl)
        let formats: [TextureFormat] = Array(repeating: .rgba16Float, count: 5) + [.rgba8Unorm]
        var targets: [ResourceID] = []
        var declarations: [UInt32: RenderColorTarget] = [:]
        var attachments: [UInt32: RenderColorAttachment] = [:]
        for (index, format) in formats.enumerated() {
            let texture = try await device.createTexture(.init(width: 1, height: 1, format: format, usage: [.renderAttachment, .copySource]))
            targets.append(texture)
            if index > 0 {
                declarations[UInt32(index)] = .init(format: format)
                attachments[UInt32(index)] = .init(texture: texture, loadAction: .clear(.transparentBlack))
            }
        }
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment,
            colorFormat: formats[0], additionalColorTargets: declarations)
        let readback = try await device.createBuffer(.init(size: 8, usage: [.copyDestination, .mapRead]))
        let result = try await device.submitCommands([
            .render(pipeline: pipeline, target: targets[0], clear: .transparentBlack, vertexCount: 3, additionalColors: attachments),
            .copyTextureToBuffer(source: targets[0], destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 8),
            .copyTextureToBuffer(source: targets[5], destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
        ]).wait()
        #expect(result == [[0x00003c00, 0x3c000000], [0xff00ff00]])
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
