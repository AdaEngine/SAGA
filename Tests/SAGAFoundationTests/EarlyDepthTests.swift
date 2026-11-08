import Testing
import SAGACompiler
import SAGAShaderIR
import SAGATypes
import SAGACore
import SAGAMetal

private func earlyDepthSource(_ mode: String) -> ShaderInput {
    .wgsl("""
        @group(0) @binding(0) var<storage, read_write> counter: u32;
        struct Outputs {
            @builtin(frag_depth) depth: f32,
            @location(0) color: vec4f,
        }
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            if i == 0u { return vec4f(-1, -1, 0.75, 1); }
            if i == 1u { return vec4f(3, -1, 0.75, 1); }
            return vec4f(-1, 3, 0.75, 1);
        }
        @fragment @early_depth_test(\(mode))
        fn fs(@builtin(position) p: vec4f) -> Outputs {
            counter += 1u;
            return Outputs(\(mode == "unchanged" ? "p.z" : mode == "greater_equal" ? "0.875" : "0.25"), vec4f(0, 1, 0, 1));
        }
        @fragment @early_depth_test(\(mode))
        fn depthOnly(@builtin(position) p: vec4f) -> @builtin(frag_depth) f32 {
            return p.z;
        }
        """)
}

@Test(arguments: ["force", "greater_equal", "less_equal", "unchanged"], [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsEarlyDepthModesForDirectAndStructuredOutputs(mode: String, format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let combined = try compiler.compileRenderStage(earlyDepthSource(mode), entryPoint: "fs", to: format)
    let direct = try compiler.compileRenderStage(earlyDepthSource(mode), entryPoint: "depthOnly", to: format)
    let suppressesOutput = (format == .msl && ["force", "unchanged"].contains(mode)) ||
        (format == .hlsl && mode == "unchanged")
    #expect(combined.writesDepth == !suppressesOutput)
    #expect(direct.writesDepth == !suppressesOutput)
    #expect(combined.earlyDepthTest != nil && direct.earlyDepthTest == combined.earlyDepthTest)
    #expect(combined.colorOutputKind == .float)
    #expect(direct.colorOutputKind == nil)
    guard case .text(_, let combinedText) = combined.artifact, case .text(_, let directText) = direct.artifact else {
        Issue.record("Expected source"); return
    }
    if format == .hlsl && mode == "unchanged" {
        #expect(directText.contains("float saga_ignored_depth ="))
    }
    for text in [combinedText, directText] {
        switch format {
        case .msl:
            #expect(text.contains("[[early_fragment_tests]]") == (mode == "force"))
            #expect(text.contains("[[depth(greater)]]") == (mode == "greater_equal"))
            #expect(text.contains("[[depth(less)]]") == (mode == "less_equal"))
            #expect(!text.contains("[[depth(any)]]"))
        case .hlsl:
            #expect(text.contains("[earlydepthstencil]") == (mode == "force"))
            #expect(text.contains("SV_DepthGreaterEqual") == (mode == "greater_equal"))
            #expect(text.contains("SV_DepthLessEqual") == (mode == "less_equal"))
            #expect(text.contains("SV_Depth") == (mode != "unchanged"))
            #expect(!text.contains("(void)("))
        case .glsl:
            let layout = switch mode {
            case "force": "layout(early_fragment_tests) in;"
            case "greater_equal": "layout(depth_greater) out float gl_FragDepth;"
            case "less_equal": "layout(depth_less) out float gl_FragDepth;"
            default: "layout(depth_unchanged) out float gl_FragDepth;"
            }
            #expect(text.contains(layout))
        case .wgsl:
            #expect(text.contains("@early_depth_test(\(mode))"))
            let module = try compiler.parseAndValidate(.wgsl(text), capabilities: [.earlyDepthTest])
            #expect(module.module.entryPoints.first?.earlyDepthTest != nil)
        default: Issue.record("Unexpected shader target")
        }
    }
}

@Test func earlyDepthValidationRequiresCapabilityAndFragmentStage() throws {
    let compiler = IndependentShaderCompiler()
    #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(earlyDepthSource("force")) }
    let module = try compiler.parseAndValidate(earlyDepthSource("force"), capabilities: [.earlyDepthTest])
    #expect(module.module.entryPoints.first(where: { $0.name == "fs" })?.earlyDepthTest == .force)
    for invalid in [
        "@early_depth_test(force) fn helper() {}",
        "@early_depth_test(force) @compute @workgroup_size(1) fn run() {}",
        "@early_depth_test(force) @vertex fn vs() -> @builtin(position) vec4f { return vec4f(1); }",
        "@early_depth_test(any) @fragment fn fs() {}",
        "@early_depth_test(force,) @fragment fn fs() {}",
        "@early_depth_test @fragment fn fs() {}",
        "@early_depth_test(force) @early_depth_test(unchanged) @fragment fn fs() {}",
    ] {
        #expect(throws: ShaderFailure.self) { try compiler.parseAndValidate(.wgsl(invalid), capabilities: [.earlyDepthTest]) }
    }
}

#if canImport(Metal)
@Test(arguments: ["force", "greater_equal", "less_equal", "unchanged"])
func earlyDepthControlsRasterDepthAndFragmentEffectsOnMetal(mode: String) async throws {
    let device = try await Device.make(backend: MetalBackend.makeDevice(), requiredFeatures: [.shaderEarlyDepthTest])
    let compiler = IndependentShaderCompiler()
    let source = earlyDepthSource(mode)
    let vertex = try compiler.compileRenderStage(source, entryPoint: "vs", to: .msl)
    let fragment = try compiler.compileRenderStage(source, entryPoint: "fs", to: .msl)
    let direct = try compiler.compileRenderStage(source, entryPoint: "depthOnly", to: .msl)
    _ = try await device.createRenderPipeline(
        vertex: vertex, fragment: direct, colorFormat: .rgba8Unorm, depthState: .init(compare: .less), sampleMask: 1)
    let pipeline = try await device.createRenderPipeline(
        vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm, depthState: .init(compare: .less))
    let color = try await device.createTexture(.init(width: 1, height: 1, format: .rgba8Unorm, usage: [.renderAttachment, .copySource]))
    let depth = try await device.createTexture(.init(width: 1, height: 1, format: .depth32Float, usage: [.renderAttachment, .copySource]))
    let storage = try await device.createBuffer(.init(size: 4, usage: [.storage, .copySource]))
    let readback = try await device.createBuffer(.init(size: 4, usage: [.mapRead, .copyDestination]))
    let bindings: [ComputeBindingSlot: ComputeBufferBinding] = [.init(binding: 0): .init(buffer: storage)]
    let blue = RenderClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    var commands: [BufferCommand] = []
    for clearDepth: Double in mode == "force" ? [0.5, 1] : [1] {
        commands += [
            .render(pipeline: pipeline, target: color, clear: blue, vertexCount: 3,
                    depth: .init(texture: depth, loadAction: .clear(clearDepth)), storageBuffers: bindings),
            .copyTextureToBuffer(source: color, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
            .copyTextureToBuffer(source: depth, destination: readback, bytesPerRow: 256),
            .read(buffer: readback, byteCount: 4),
            .copy(source: storage, destination: readback, byteCount: 4),
            .read(buffer: readback, byteCount: 4),
        ]
    }
    let snapshots = try await device.submitCommands(commands).wait()
    let expectedDepth: Float = switch mode {
    case "greater_equal": 0.875
    case "less_equal": 0.25
    default: 0.75
    }
    let rejected: [[UInt32]] = mode == "force" ? [[0xffff0000], [Float(0.5).bitPattern], [0]] : []
    #expect(snapshots == rejected + [[0xff00ff00], [expectedDepth.bitPattern], [1]])
    await device.close()
}
#endif
