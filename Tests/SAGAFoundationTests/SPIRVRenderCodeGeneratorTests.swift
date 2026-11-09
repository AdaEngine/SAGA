import SAGACompiler
import SAGACore
import SAGAShaderIR
import SAGATypes
import Testing
#if canImport(Metal)
import SAGAMetal
#endif

private let spirvRenderInput = ShaderInput.wgsl("""
    struct VertexOutput {
        @location(2) tint: vec2f,
        @builtin(position) clip: vec4f,
        @location(5) @interpolate(flat) tag: u32,
    }
    struct FragmentInput {
        @location(5) @interpolate(flat) tag: u32,
        @builtin(front_facing) front: bool,
        @location(2) tint: vec2f,
    }

    @vertex fn shapeVertex(@builtin(vertex_index) index: u32) -> VertexOutput {
        var corner = vec2f(-1.0, -0.625);
        if index == 0u { corner = vec2f(-1.0, 1.0); }
        if index == 1u { corner = vec2f(0.625, 1.0); }
        return VertexOutput(vec2f(1.0, 0.0), vec4f(corner, 0.25, 1.0), 23u);
    }

    @fragment fn shadeFragment(input: FragmentInput) -> @location(0) vec4f {
        let green = select(0.0, 1.0, input.front);
        let blue = select(0.0, 1.0, input.tag == 23u);
        return vec4f(input.tint.x, input.tint.y + green, blue, 1.0);
    }
    """)

private func exportRenderModule(_ compiler: IndependentShaderCompiler) throws -> [UInt32] {
    guard case .spirv(let words) = try compiler.compile(spirvRenderInput, to: .spirv) else {
        throw ShaderFailure.incompatibleArtifact
    }
    return words
}

private func expectRenderInterfaces(_ words: [UInt32]) throws {
    try #require(words.count > 5)
    var interface: [UInt32: (function: UInt32, variables: Set<UInt32>)] = [:]
    var origins: Set<UInt32> = []
    var builtins: [UInt32: UInt32] = [:]
    var locations: [UInt32: UInt32] = [:]
    var flat: Set<UInt32> = []
    var offset = 5
    while offset < words.count {
        let count = Int(words[offset] >> 16)
        try #require(count > 0 && count <= words.count - offset)
        let operands = Array(words[(offset + 1)..<(offset + count)])
        switch words[offset] & 0xffff {
        case 15 where operands.count >= 3 && [0, 4].contains(operands[0]): // OpEntryPoint
            let name = operands[0] == 0 ? "shapeVertex" : "shadeFragment"
            let nameWordCount = (name.utf8.count + 4) / 4
            interface[operands[0]] = (operands[1], Set(operands.dropFirst(2 + nameWordCount)))
        case 16 where operands.count == 2 && operands[1] == 7: // OriginUpperLeft
            origins.insert(operands[0])
        case 71 where operands.count == 3 && operands[1] == 11: // BuiltIn
            builtins[operands[0]] = operands[2]
        case 71 where operands.count == 3 && operands[1] == 30: // Location
            locations[operands[0]] = operands[2]
        case 71 where operands.count == 2 && operands[1] == 14: // Flat
            flat.insert(operands[0])
        default: break
        }
        offset += count
    }
    let vertex = try #require(interface[0])
    let fragment = try #require(interface[4])
    #expect(origins == [fragment.function])
    #expect(Set(vertex.variables.compactMap { builtins[$0] }) == [0, 42]) // Position and VertexIndex
    #expect(Set(fragment.variables.compactMap { builtins[$0] }) == [17]) // FrontFacing
    #expect(Set(vertex.variables.compactMap { locations[$0] }) == [2, 5])
    #expect(Set(fragment.variables.compactMap { locations[$0] }) == [0, 2, 5])
    for variables in [vertex.variables, fragment.variables] {
        #expect(Set(variables.intersection(flat).compactMap { locations[$0] }) == [5])
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvRenderCodeGeneratorExportsLinkedStagesAndVaryings(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    let words = try exportRenderModule(compiler)
    try expectRenderInterfaces(words)
    let input = ShaderInput.spirv(words)
    let module = try compiler.parseAndValidate(input).module
    #expect(Set(module.entryPoints.map(\.name)) == ["shapeVertex", "shadeFragment"])
    #expect(module.entryPoints.contains { $0.name == "shapeVertex" && $0.stage == .vertex })
    #expect(module.entryPoints.contains { $0.name == "shadeFragment" && $0.stage == .fragment })
    let vertex = try compiler.compileRenderStage(input, entryPoint: "shapeVertex", to: format)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "shadeFragment", to: format)
    #expect(vertex.varyings == fragment.varyings)
    #expect(vertex.varyings.map(\.location) == [2, 5])
    #expect(vertex.varyings.map(\.componentCount) == [2, 1])
    #expect(vertex.varyings.map(\.interpolation) == [.perspective, .flat])
    #expect(fragment.colorOutputComponentCounts == [0: 4])
    if format == .wgsl {
        for shader in [vertex, fragment] {
            guard case .text(.wgsl, let text) = shader.artifact else { throw ShaderFailure.incompatibleArtifact }
            _ = try compiler.parseAndValidate(.wgsl(text))
        }
    }
}

#if canImport(Metal)
@Test func spirvRenderCodeGeneratorPreservesCoverageAndFacingOnMetal() async throws {
    let compiler = IndependentShaderCompiler()
    let input = ShaderInput.spirv(try exportRenderModule(compiler))
    let vertex = try compiler.compileRenderStage(input, entryPoint: "shapeVertex", to: .msl)
    let fragment = try compiler.compileRenderStage(input, entryPoint: "shadeFragment", to: .msl)
    let device = try await Device.make(backend: MetalBackend.makeDevice())
    do {
        let counterClockwise = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm)
        let clockwise = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .rgba8Unorm,
                                                              primitive: .init(frontFace: .clockwise))
        let target = try await device.createTexture(.init(width: 4, height: 4, format: .rgba8Unorm,
            usage: [.renderAttachment, .copySource]))
        let readback = try await device.createBuffer(.init(size: 784, usage: [.copyDestination, .mapRead]))
        let clear = RenderClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        var commands: [BufferCommand] = []
        for pipeline in [counterClockwise, clockwise] {
            commands += [
                .render(pipeline: pipeline, target: target, clear: clear, vertexCount: 3),
                .copyTextureToBuffer(source: target, destination: readback, bytesPerRow: 256),
                .read(buffer: readback, byteCount: 784),
            ]
        }
        let snapshots = try await device.submitCommands(commands).wait()
        try #require(snapshots.count == 2 && snapshots.allSatisfy { $0.count == 196 })
        let pixels = snapshots.map { words in (0..<4).flatMap { row in Array(words[(row * 64)..<(row * 64 + 4)]) } }
        let expected: [[UInt32]] = [UInt32(0xffff00ff), 0xffffffff].map { color in
            (0..<4).flatMap { row in (0..<4).map { column in row + column < 3 ? color : 0xff000000 } }
        }
        #expect(pixels == expected)
    } catch {
        await device.close()
        throw error
    }
    await device.close()
}
#endif
