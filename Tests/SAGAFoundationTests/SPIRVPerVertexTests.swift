import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    private static func perVertexModule(
        scalar: Bool = false, count: UInt32 = 3, capability: Bool = true, extensionName: Bool = true,
        model: UInt32 = 4, input: Bool = true, flat: Bool = false
    ) -> [UInt32] {
        var words = head + i(17, 1)
        if capability { words += i(17, 5284) } // FragmentBarycentricKHR
        if extensionName {
            // SPV_KHR_fragment_shader_barycentric
            words += i(10, 0x5f565053, 0x5f52484b, 0x67617266, 0x746e656d, 0x6168735f,
                       0x5f726564, 0x79726162, 0x746e6563, 0x00636972)
        }
        words += i(14, 0, 1) + i(15, model, 20, 0x6e69616d, 0, 12, 13)
        if model == 4 { words += i(16, 20, 7) }
        words += i(71, 12, 30, 0) + i(71, 12, 5285) + i(71, 13, 30, 0)
        if flat { words += i(71, 12, 14) }
        let element: UInt32 = scalar ? 2 : 4
        let storage: UInt32 = input ? 1 : 3
        for instruction in [
            i(19, 1), i(21, 2, 32, 0), i(22, 3, 32), i(23, 4, 3, 2), i(43, 2, 5, count), i(28, 6, element, 5),
            i(32, 7, storage, 6), i(32, 8, 3, 2), i(33, 9, 1), i(59, 7, 12, storage), i(59, 8, 13, 3),
            i(54, 1, 20, 0, 9), i(248, 21), i(61, 6, 22, 12), i(81, element, 23, 22, count >= 3 ? 2 : 0),
        ] { words += instruction }
        if !scalar { words += i(81, 3, 24, 23, 0) + i(124, 2, 25, 24) }
        return words + i(62, 13, scalar ? 23 : 25) + i(253) + i(56)
    }

    @Test(arguments: [false, true])
    func spirvPerVertexPreservesAnArrayOfThreeVertexValues(scalar: Bool) throws {
        let words = Self.perVertexModule(scalar: scalar)
        let module = try IndependentShaderCompiler().parseAndValidate(.spirv(words), capabilities: [.perVertex]).module
        let entry = try module.functions.value(for: module.entryPoints[0].function)
        let input = try #require(entry.arguments.first)
        #expect(input.binding == .location(index: 0, interpolation: .perVertex, sampling: nil,
                                             blendSource: nil, perPrimitive: false))
        guard case .array(let element, .constant(3), _) = try module.types.value(for: input.type).kind else {
            Issue.record("Expected exactly three per-vertex values")
            return
        }
        let elementKind = try module.types.value(for: element).kind
        #expect(elementKind == (scalar ? .scalar(.u32) : .vector(size: .two, scalar: .f32)))
    }

    @Test func spirvPerVertexRejectsInvalidDeclarations() throws {
        try fails(Self.perVertexModule(capability: false), code: .unsupportedCapability)
        try fails(Self.perVertexModule(extensionName: false), code: .unsupportedCapability)
        try fails(Self.perVertexModule(count: 2))
        try fails(Self.perVertexModule(model: 0))
        try fails(Self.perVertexModule(input: false))
        try fails(Self.perVertexModule(flat: true))
    }
}

@Test func spirvPerVertexWritesDecorationsAndRoundTripsTheBinding() throws {
    let compiler = IndependentShaderCompiler()
    let source = ShaderInput.wgsl("""
        enable wgpu_per_vertex;
        struct Inputs {
            @location(0) @interpolate(per_vertex) points: array<vec2f, 3>,
            @location(1) @interpolate(per_vertex) ids: array<u32, 3>,
        }
        @fragment fn main(input: Inputs) -> @location(0) vec4u {
            return vec4u(bitcast<u32>(input.points[2].x), input.ids[0], input.ids[1], 255u);
        }
        """)
    guard case .spirv(let words) = try compiler.compile(source, to: .spirv) else { throw ShaderFailure.incompatibleArtifact }
    var cursor = 5
    var decorated = 0
    var barycentricCapability = false
    while cursor < words.count {
        let count = Int(words[cursor] >> 16)
        try #require(count > 0 && count <= words.count - cursor)
        let opcode = words[cursor] & 0xffff
        if opcode == 17 && count == 2 && words[cursor + 1] == 5284 { barycentricCapability = true }
        if opcode == 71 && count == 3 && words[cursor + 2] == 5285 { decorated += 1 }
        cursor += count
    }
    #expect(barycentricCapability)
    #expect(decorated == 2)
    let module = try compiler.parseAndValidate(.spirv(words), capabilities: [.perVertex]).module
    let entry = try module.functions.value(for: module.entryPoints[0].function)
    #expect(entry.arguments.count == 2)
    for (index, input) in entry.arguments.enumerated() {
        #expect(input.binding == .location(index: UInt32(index), interpolation: .perVertex, sampling: nil,
                                             blendSource: nil, perPrimitive: false))
    }
}
