import Testing
import SAGACompiler
import SAGAShaderIR

private let triangleVertex = """
    @vertex fn vertexMain(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        if index == 0u { return vec4f(-1.0f, 1.0f, 0.25f, 1.0f); }
        if index == 1u { return vec4f(1.0f, 1.0f, 0.25f, 1.0f); }
        return vec4f(-1.0f, -1.0f, 0.25f, 1.0f);
    }
    """

private let triangleFragment = """
    @fragment fn fragmentMain() -> @location(0) vec4f {
        return vec4f(0.0f, 1.0f, 0.0f, 1.0f);
    }
    """

let triangleProgram = triangleVertex + "\n" + triangleFragment

@Test(arguments: [ShaderOutputFormat.msl, .hlsl, .glsl, .wgsl])
func emitsSelectedVertexAndFragmentStages(to format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let vertex = try compiler.compileRenderStage(.wgsl(triangleProgram), entryPoint: "vertexMain", to: format)
    let fragment = try compiler.compileRenderStage(.wgsl(triangleProgram), entryPoint: "fragmentMain", to: format)
    #expect(vertex.stage == .vertex)
    #expect(fragment.stage == .fragment)
    #expect(vertex.sourceEntryPoint == "vertexMain")
    #expect(fragment.sourceEntryPoint == "fragmentMain")
    #expect(try compiler.compile(.wgsl(triangleVertex), to: format) == vertex.artifact)
    #expect(try compiler.compile(.wgsl(triangleFragment), to: format) == fragment.artifact)
    guard case .text(let vertexFormat, let vs) = vertex.artifact,
          case .text(let fragmentFormat, let fs) = fragment.artifact else {
        Issue.record("Expected stage-specific text artifacts")
        return
    }
    #expect(vertexFormat == format && fragmentFormat == format)
    #expect(!vs.contains("saga_fragment") && !fs.contains("saga_vertex"))
    #expect(!vs.contains("== 0u))"))
    #expect(!vs.contains("== 1u))"))
    switch format {
    case .msl:
        #expect(vs.contains("[[vertex_id]]") && vs.contains("[[position]]"))
        #expect(fs.contains("[[color(0)]]"))
    case .hlsl:
        #expect(vs.contains("SV_VertexID") && vs.contains("SV_Position"))
        #expect(fs.contains("SV_Target0"))
    case .glsl:
        #expect(vertex.entryPoint == "main" && fragment.entryPoint == "main")
        #expect(vs.contains("uint(gl_VertexID)"))
        #expect(vs.contains("gl_Position.y = -gl_Position.y"))
        #expect(vs.contains("gl_Position.z * 2.0 - gl_Position.w"))
        #expect(fs.contains("layout(location = 0) out vec4"))
    case .wgsl:
        #expect(vs.contains("@vertex") && vs.contains("@builtin(vertex_index)"))
        #expect(fs.contains("@fragment") && fs.contains("@location(0)"))
    default: Issue.record("Unexpected target")
    }
    if format == .wgsl {
        #expect(try compiler.parseAndValidate(.wgsl(vs)).module.entryPoints.first?.stage == .vertex)
        #expect(try compiler.parseAndValidate(.wgsl(fs)).module.entryPoints.first?.stage == .fragment)
    }
    if format == .glsl {
        #expect(try compiler.parseAndValidate(.glsl(vs, stage: .vertex, defines: [:])).module.entryPoints.first?.stage == .vertex)
        #expect(try compiler.parseAndValidate(.glsl(fs, stage: .fragment, defines: [:])).module.entryPoints.first?.stage == .fragment)
    }
}

@Test(arguments: ["gl_VertexID", "gl_VertexIndex", "gl_InstanceID", "gl_InstanceIndex"])
func glslVertexBuiltinsExposeUnsignedIRInputsAndSignedGLSLValues(name: String) throws {
    let source = """
        #version 450 core
        void main() {
            if (\(name) < 1) { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }
            else { gl_Position = vec4(1.0, 1.0, 0.0, 1.0); }
        }
        """
    let module = try IndependentShaderCompiler().parseAndValidate(.glsl(source, stage: .vertex, defines: [:])).module
    let entry = try #require(module.entryPoints.first)
    let function = try module.functions.value(for: entry.function)
    let argument = try #require(function.arguments.first)
    #expect(argument.binding == .builtIn(name.hasPrefix("gl_Vertex") ? .vertexIndex : .instanceIndex))
    #expect(try module.types.value(for: argument.type).kind == .scalar(.u32))
    #expect(function.expressions.values.contains {
        if case .bitcast = $0.kind { return true }
        return false
    })
}

@Test func renderSelectionRejectsMissingEntryAndUnsupportedStageInterface() throws {
    let compiler = IndependentShaderCompiler()
    for (source, entry) in [
        (triangleProgram, "missing"),
        ("@compute @workgroup_size(1) fn run() {}", "run"),
        ("@fragment fn run() -> @location(8) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }", "run"),
    ] {
        #expect(throws: ShaderFailure.self) {
            try compiler.compileRenderStage(.wgsl(source), entryPoint: entry, to: .msl)
        }
    }
}

#if canImport(Metal)
import Metal

@Test func rendersEmittedVertexAndFragmentShadersToMetalTexture() throws {
    let compiler = IndependentShaderCompiler()
    let device = try #require(MTLCreateSystemDefaultDevice())
    func function(_ entry: String) throws -> any MTLFunction {
        let stage = try compiler.compileRenderStage(.wgsl(triangleProgram), entryPoint: entry, to: .msl)
        guard case .text(.msl, let code) = stage.artifact else { throw ShaderFailure.incompatibleArtifact }
        let library = try device.makeLibrary(source: code, options: nil)
        return try #require(library.makeFunction(name: stage.entryPoint))
    }
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = try function("vertexMain")
    descriptor.fragmentFunction = try function("fragmentMain")
    descriptor.colorAttachments[0].pixelFormat = .rgba8Unorm
    let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                                      width: 8, height: 8, mipmapped: false)
    textureDescriptor.usage = .renderTarget
    textureDescriptor.storageMode = .shared
    let texture = try #require(device.makeTexture(descriptor: textureDescriptor))
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = texture
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].storeAction = .store
    pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 1, alpha: 1)
    let queue = try #require(device.makeCommandQueue())
    let command = try #require(queue.makeCommandBuffer())
    let encoder = try #require(command.makeRenderCommandEncoder(descriptor: pass))
    encoder.setRenderPipelineState(pipeline)
    encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    try #require(command.status == .completed)
    var pixels = [UInt8](repeating: 0, count: 8 * 8 * 4)
    pixels.withUnsafeMutableBytes {
        texture.getBytes($0.baseAddress!, bytesPerRow: 32, from: MTLRegionMake2D(0, 0, 8, 8), mipmapLevel: 0)
    }
    let wrongPixels = (0..<64).filter { index in
        let diagonal = index / 8 + index % 8
        guard diagonal != 7 else { return false }
        let expected: [UInt8] = diagonal < 7 ? [0, 255, 0, 255] : [0, 0, 255, 255]
        return Array(pixels[index * 4..<index * 4 + 4]) != expected
    }
    #expect(wrongPixels.isEmpty)
}
#endif
