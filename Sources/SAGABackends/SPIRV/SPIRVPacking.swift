import SAGAShaderIR

extension SPIRVBinaryWriter {
    mutating func writePacking(
        _ builtin: IRPackingBuiltin, operand: UInt32, type: IRHandle<IRType>
    ) throws -> UInt32 {
        let result = try typeID(type)
        if builtin.encoding == .binary16 {
            return try glslMathInstruction(builtin.packs ? 58 : 62, type: result, operands: [operand])
        }
        if !builtin.packs {
            switch (builtin.encoding, builtin.size) {
            case (.snorm, .four): return try glslMathInstruction(63, type: result, operands: [operand])
            case (.unorm, .four): return try glslMathInstruction(64, type: result, operands: [operand])
            case (.snorm, .two): return try glslMathInstruction(60, type: result, operands: [operand])
            case (.unorm, .two): return try glslMathInstruction(61, type: result, operands: [operand])
            case (.sint, .four), (.uint, .four):
                var lanes: [UInt32] = []
                for index in 0..<UInt32(builtin.size.rawValue) {
                    lanes.append(try packedIntegerLane(operand, index: index, signed: builtin.encoding == .sint))
                }
                return try rawValueInstruction(80, type: result, operands: lanes) // OpCompositeConstruct
            default: throw ShaderFailure.invalidHandle
            }
        }

        let signed = builtin.encoding.isSignedIntegerField
        let integer = try vectorTypeID(size: builtin.size, scalar: signed ? .i32 : .u32)
        let lanes: UInt32
        switch builtin.encoding {
        case .snorm, .unorm:
            let float = try vectorTypeID(size: builtin.size, scalar: .f32)
            let lower = try splatLiteralID(.f32(signed ? -1 : 0), type: float, size: builtin.size)
            let upper = try splatLiteralID(.f32(1), type: float, size: builtin.size)
            let clamped = try glslMathInstruction(43, type: float, operands: [operand, lower, upper]) // FClamp
            let scale = try splatLiteralID(.f32(Float(builtin.normalizedScale)), type: float, size: builtin.size)
            let scaled = try rawValueInstruction(133, type: float, operands: [clamped, scale]) // OpFMul
            let half = try splatLiteralID(.f32(0.5), type: float, size: builtin.size)
            // GLSL Pack*Norm does not specify WGSL's direction for ties.
            let biased = try rawValueInstruction(129, type: float, operands: [scaled, half]) // OpFAdd
            let rounded = try glslMathInstruction(8, type: float, operands: [biased]) // Floor
            lanes = try rawValueInstruction(signed ? 110 : 109, type: integer, operands: [rounded])
        case .sintClamp:
            let lower = try splatLiteralID(.i32(-128), type: integer, size: builtin.size)
            let upper = try splatLiteralID(.i32(127), type: integer, size: builtin.size)
            lanes = try glslMathInstruction(45, type: integer, operands: [operand, lower, upper]) // SClamp
        case .uintClamp:
            let upper = try splatLiteralID(.u32(255), type: integer, size: builtin.size)
            lanes = try glslMathInstruction(38, type: integer, operands: [operand, upper]) // UMin
        case .sint, .uint:
            lanes = operand
        case .binary16:
            throw ShaderFailure.invalidHandle
        }
        let bits = try signed ? rawValueInstruction(124, type: vectorTypeID(size: builtin.size, scalar: .u32), operands: [lanes]) : lanes
        let count = try unsignedConstantID(UInt32(builtin.bitsPerLane))
        var word = try zeroID(typeID: result)
        for index in 0..<UInt32(builtin.size.rawValue) {
            let lane = try rawValueInstruction(81, type: result, operands: [bits, index]) // OpCompositeExtract
            let offset = try unsignedConstantID(index * UInt32(builtin.bitsPerLane))
            word = try rawValueInstruction(201, type: result, operands: [word, lane, offset, count]) // OpBitFieldInsert
        }
        return word
    }

    mutating func packedIntegerLane(_ word: UInt32, index: UInt32, signed: Bool) throws -> UInt32 {
        let type = try scalarTypeID(signed ? .i32 : .u32)
        let base = try signed ? rawValueInstruction(124, type: type, operands: [word]) : word
        let offset = try unsignedConstantID(index * 8)
        let count = try unsignedConstantID(8)
        return try rawValueInstruction(signed ? 202 : 203, type: type, operands: [base, offset, count])
    }
}
