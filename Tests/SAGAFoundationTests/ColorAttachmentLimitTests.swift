import SAGACore
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private func colorAttachmentLimitShader(targets: Int) -> ShaderInput {
    let fields = (0..<targets).map { "@location(\($0)) color\($0): vec4f," }.joined(separator: "\n")
    let colors = (0..<targets).map { "vec4f(\($0).25, 0.5, 0.75, 1)" }.joined(separator: ", ")
    return .wgsl("""
        struct Outputs { \(fields) }
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            let points = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
            return vec4f(points[i], 0, 1);
        }
        @fragment fn fs() -> Outputs { return Outputs(\(colors)); }
        """)
}

@Test func colorAttachmentLimitsUseTheRenderCostOfRG11B10Float() {
    #expect(TextureFormat.rg11b10Ufloat.colorAttachmentByteCost == 8)
    #expect(TextureFormat.rg11b10Ufloat.colorAttachmentAlignment == 4)
}

@Test(arguments: [ShaderOutputFormat.wgsl, .glsl, .hlsl, .msl])
func colorAttachmentLimitsKeepTargetSpecificOutputRegisters(_ format: ShaderOutputFormat) throws {
    let source = ShaderInput.wgsl("@fragment fn fs() -> @location(8) vec4f { return vec4f(1); }")
    let compiler = IndependentShaderCompiler()
    if format == .hlsl || format == .msl {
        #expect(throws: ShaderFailure.self) { try compiler.compileRenderStage(source, entryPoint: "fs", to: format) }
    } else {
        let shader = try compiler.compileRenderStage(source, entryPoint: "fs", to: format)
        #expect(shader.colorOutputKinds == [8: .float])
    }
}

@Test(arguments: [UInt32(11), 12])
func colorAttachmentLimitsValidateAlignmentOrderAndSparseSlots(_ budget: UInt32) async throws {
    let limits = try DeviceCapabilities(maxBufferSize: 4096, features: [.rg11b10UfloatRenderable],
                                        maxColorAttachments: 3, maxColorAttachmentBytesPerSample: budget)
    let host = RenderTestBackend(capabilities: limits)
    let device = try await Device.make(backend: host)
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        struct Outputs { @location(0) first: vec4f, @location(2) second: vec4f, }
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); }
        @fragment fn fs() -> Outputs { return Outputs(vec4f(0), vec4f(1)); }
        """)
    let vertex = try compiler.compileWebGPURenderStage(source, entryPoint: "vs")
    let fragment = try compiler.compileWebGPURenderStage(source, entryPoint: "fs")
    let packed = RenderColorTarget(format: .rg11b10Ufloat)
    do {
        await #expect(throws: GPUError.validation("Color target location exceeds the device attachment limit")) {
            try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .r8Unorm,
                                                  additionalColorTargets: [2: packed, 3: .init(format: .r8Unorm)])
        }
        #expect(await host.allocatedRenderPipelines == 0)
        if budget == 12 {
            _ = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .r8Unorm,
                                                      additionalColorTargets: [2: packed])
        } else {
            await #expect(throws: GPUError.validation("Color attachments exceed the device bytes-per-sample limit")) {
                try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .r8Unorm,
                                                      additionalColorTargets: [2: packed])
            }
            #expect(await host.allocatedRenderPipelines == 0)
        }
        _ = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rg11b10Ufloat,
                                                  additionalColorTargets: [2: .init(format: .r8Unorm)])
        #expect(await host.allocatedRenderPipelines == (budget == 12 ? 2 : 1))
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}

#if canImport(Metal)
@Test func colorAttachmentLimitsRenderFloat32TargetsAtTheMetalBudget() async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    let targetCount = Int(min(device.capabilities.maxColorAttachments, device.capabilities.maxColorAttachmentBytesPerSample / 16))
    let compiler = IndependentShaderCompiler()
    let source = colorAttachmentLimitShader(targets: targetCount)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    do {
        let formats = Dictionary(uniqueKeysWithValues: (UInt32(1)..<UInt32(targetCount)).map {
            ($0, RenderColorTarget(format: .rgba32Float))
        })
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba32Float,
                                                             additionalColorTargets: formats)
        var textures: [ResourceID] = []
        for _ in 0..<targetCount {
            textures.append(try await device.createTexture(.init(width: 1, height: 1, format: .rgba32Float,
                                                                  usage: [.renderAttachment, .copySource])))
        }
        let extras = Dictionary(uniqueKeysWithValues: textures.dropFirst().enumerated().map { (index, texture) in
            (UInt32(index + 1), RenderColorAttachment(texture: texture, loadAction: .clear(.transparentBlack)))
        })
        let readback = try await device.createBuffer(.init(size: 16, usage: [.copyDestination, .mapRead]))
        var commands: [BufferCommand] = [
            .render(pipeline: pipeline, target: textures[0], clear: .transparentBlack, vertexCount: 3, additionalColors: extras),
        ]
        for texture in textures {
            commands.append(.copyTextureToBuffer(source: texture, destination: readback, bytesPerRow: 256))
            commands.append(.read(buffer: readback, byteCount: 16))
        }
        let snapshots = try await device.submitCommands(commands).wait()
        let expected = (0..<targetCount).map { [Float($0) + 0.25, 0.5, 0.75, 1].map(\.bitPattern) }
        #expect(snapshots == expected)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
