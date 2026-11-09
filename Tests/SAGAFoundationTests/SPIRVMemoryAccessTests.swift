import SAGACompiler
import SAGAShaderIR
import Testing

extension SPIRVFrontendTests {
    private static func appendAccessOperands(_ operands: [UInt32], to opcode: UInt32, in words: [UInt32]) -> [UInt32] {
        var words = words
        var offset = 5
        while offset < words.count {
            let count = Int(words[offset] >> 16)
            if words[offset] & 0xffff == opcode {
                words[offset] = UInt32(count + operands.count) << 16 | opcode
                words.insert(contentsOf: operands, at: offset + count)
                break
            }
            offset += count
        }
        return words
    }

    static func pointerMemoryAccessModule(vulkan: Bool, load: [UInt32] = [], store: [UInt32] = []) -> [UInt32] {
        var words = head + i(17, 1)
        if vulkan {
            words[1] = 0x00010500
            words += i(17, 5345)
        }
        words += i(14, 0, vulkan ? 3 : 1)
        words += vulkan ? i(15, 5, 20, 0x6e69616d, 0, 10) : i(15, 5, 20, 0x6e69616d, 0)
        for instruction in [
            i(16, 20, 17, 1, 1, 1), i(71, 3, 2), i(72, 3, 0, 35, 0), i(72, 3, 1, 35, 4),
            i(71, 10, 34, 0), i(71, 10, 33, 0),
            i(19, 1), i(21, 2, 32, 0), i(30, 3, 2, 2), i(32, 4, 12, 3), i(32, 5, 12, 2), i(33, 6, 1),
            i(43, 2, 11, 0), i(43, 2, 12, 1), i(43, 2, 13, 2), i(59, 4, 10, 12),
            i(54, 1, 20, 0, 6), i(248, 21), i(65, 5, 22, 10, 11), i(65, 5, 23, 10, 12),
            i(61, 2, 24, 22), i(62, 23, 24), i(253), i(56),
        ] { words += instruction }
        return appendAccessOperands(store, to: 62, in: appendAccessOperands(load, to: 61, in: words))
    }

    static func imageMemoryAccessModule(version: UInt32 = 0x00010600, opcode: UInt32, operands: [UInt32]) -> [UInt32] {
        let base = storageImageModule()
        var words = Array(base.prefix(5))
        words[1] = version
        var offset = 5
        while offset < base.count {
            let count = Int(base[offset] >> 16)
            switch base[offset] & 0xffff {
            case 14: words += i(17, 5345) + i(14, 0, 3)
            case 15: words += i(15, 5, 40, 0x6e69616d, 0, 15, 16)
            case 54:
                words += i(43, 2, 27, 2) // Workgroup scope for optional image operands.
                words += base[offset..<(offset + count)]
            default: words += base[offset..<(offset + count)]
            }
            offset += count
        }
        return appendAccessOperands(operands, to: opcode, in: words)
    }

    @Test(arguments: [[2], [2, 3], [0, 4], [0x30], [0x10, 13], [0x28, 13]])
    func spirvPointerMemoryAccessRejectsMalformedMasks(_ load: [UInt32]) throws {
        try fails(Self.pointerMemoryAccessModule(vulkan: true, load: load))
    }

    @Test func spirvMemoryAccessDoesNotEraseOrderingOrVolatile() throws {
        for load: [UInt32] in [[1], [0x20], [0x30, 13]] {
            try fails(Self.pointerMemoryAccessModule(vulkan: true, load: load), code: .unsupportedConstruct)
        }
        try fails(Self.pointerMemoryAccessModule(vulkan: true, store: [0x28, 13]), code: .unsupportedConstruct)
        try fails(Self.pointerMemoryAccessModule(vulkan: false, load: [0x20]), code: .unsupportedCapability)
        try fails(Self.pointerMemoryAccessModule(vulkan: true, load: [0x30, 12]), code: .unsupportedCapability)
        for read: [UInt32] in [[0x400], [0x800], [0x600, 27]] {
            try fails(Self.imageMemoryAccessModule(opcode: 98, operands: read), code: .unsupportedConstruct)
        }
        try fails(Self.imageMemoryAccessModule(opcode: 98, operands: [0x600]))
        try fails(Self.imageMemoryAccessModule(opcode: 98, operands: [0x500, 27]))
        try fails(Self.imageMemoryAccessModule(opcode: 99, operands: [0x500, 27]), code: .unsupportedConstruct)
        try fails(Self.imageMemoryAccessModule(version: 0x00010500, opcode: 98, operands: [0x4000]),
                  code: .unsupportedCapability)
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl], [false, true])
func spirvPointerMemoryAccessAcceptsNoneAndAlignmentCacheHints(_ format: ShaderOutputFormat, _ vulkan: Bool) throws {
    let compiler = IndependentShaderCompiler()
    for operands: [UInt32] in [[0], [6, 4]] {
        let words = SPIRVFrontendTests.pointerMemoryAccessModule(vulkan: vulkan, load: operands, store: operands)
        let shader = try compiler.compileCompute(.spirv(words), to: format)
        #expect(shader.storageBindings.map(\.minimumByteSize) == [8])
        if format == .wgsl, case .text(.wgsl, let text) = shader.artifact {
            _ = try compiler.parseAndValidate(.wgsl(text))
        }
    }
}

@Test(arguments: [ShaderOutputFormat.msl, .wgsl, .hlsl, .glsl])
func spirvStorageImageMemoryAccessAcceptsNontemporalHint(_ format: ShaderOutputFormat) throws {
    let compiler = IndependentShaderCompiler()
    for opcode: UInt32 in [98, 99] {
        let words = SPIRVFrontendTests.imageMemoryAccessModule(opcode: opcode, operands: [0x4000])
        let shader = try compiler.compileCompute(.spirv(words), to: format)
        #expect(shader.storageTextures.map(\.format) == [.r32Uint, .r32Uint])
        if format == .wgsl, case .text(.wgsl, let text) = shader.artifact {
            _ = try compiler.parseAndValidate(.wgsl(text))
        }
    }
}
