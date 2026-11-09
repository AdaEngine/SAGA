import SAGAShaderIR

private struct SPIRVFloatIntegerBounds {
    let magnitudeMask: UInt32
    let infinity: UInt32
    let upper: (bits: UInt32, value: IRLiteral)
    let lower: (bits: UInt32, value: IRLiteral)?

    init(source: IRScalar, target: IRScalar) throws {
        let half = source == .f16
        magnitudeMask = half ? 0x7fff : 0x7fff_ffff
        infinity = half ? 0x7c00 : 0x7f80_0000
        switch target {
        case .i32:
            upper = half ? (0x7bff, .i32(65_504)) : (0x4eff_ffff, .i32(2_147_483_520))
            lower = half ? (0xfbff, .i32(-65_504)) : (0xcf00_0000, .i32(.min))
        case .u32:
            upper = half ? (0x7bff, .u32(65_504)) : (0x4f7f_ffff, .u32(4_294_967_040))
            lower = nil
        case .i64:
            upper = half ? (0x7bff, .i64(65_504)) : (0x5eff_ffff, .i64(9_223_371_487_098_961_920))
            lower = half ? (0xfbff, .i64(-65_504)) : (0xdf00_0000, .i64(.min))
        case .u64:
            upper = half ? (0x7bff, .u64(65_504)) : (0x5f7f_ffff, .u64(18_446_742_974_197_923_840))
            lower = nil
        default:
            throw ShaderFailure.invalidHandle
        }
    }
}

extension SPIRVBinaryWriter {
    mutating func writeSaturatingFloat64Conversion(
        _ operand: UInt32, source: IRBitcastShape, target: IRScalar, type: IRHandle<IRType>
    ) throws -> UInt32 {
        let bounds = try Float64IntegerBounds(target: target)
        let float = try scalarOrVectorTypeID(.f64, size: source.size)
        let predicate = try scalarOrVectorTypeID(.bool, size: source.size)
        let lower = try splatLiteralID(.f64(bounds.lower), type: float, size: source.size)
        let upper = try splatLiteralID(.f64(bounds.upper), type: float, size: source.size)
        let zero = try zeroID(typeID: float)

        // Ordered comparisons reject NaN without requiring any integer bitcasts.
        let below = try rawValueInstruction(184, type: predicate, operands: [operand, lower]) // OpFOrdLessThan
        let above = try rawValueInstruction(186, type: predicate, operands: [operand, upper]) // OpFOrdGreaterThan
        let atMost = try rawValueInstruction(188, type: predicate, operands: [operand, upper]) // OpFOrdLessThanEqual
        let excess = try rawValueInstruction(169, type: float, operands: [above, upper, zero]) // OpSelect
        let capped = try rawValueInstruction(169, type: float, operands: [atMost, operand, excess])
        let safe = try rawValueInstruction(169, type: float, operands: [below, lower, capped])
        let opcode: UInt32 = target.kind == .sint ? 110 : 109 // OpConvertFToS / OpConvertFToU
        return try valueInstruction(opcode, type: type, operands: [safe])
    }

    mutating func writeSaturatingFloatConversion(
        _ operand: UInt32, source: IRBitcastShape, target: IRScalar, type: IRHandle<IRType>
    ) throws -> UInt32 {
        let bounds = try SPIRVFloatIntegerBounds(source: source.scalar, target: target)
        let float = try scalarOrVectorTypeID(source.scalar, size: source.size)
        let word = try scalarOrVectorTypeID(.u32, size: source.size)
        let predicate = try scalarOrVectorTypeID(.bool, size: source.size)
        let result = try typeID(type)
        let bits = try floatConversionBits(operand, source: source, word: word)
        var limit = try splatLiteralID(.u32(bounds.upper.bits), type: word, size: source.size)
        var saturation = try splatLiteralID(bounds.upper.value, type: result, size: source.size)
        let classification: UInt32
        if let lower = bounds.lower {
            let mask = try splatLiteralID(.u32(bounds.magnitudeMask), type: word, size: source.size)
            let negative = try rawValueInstruction(172, type: predicate, operands: [bits, mask])
            let negativeLimit = try splatLiteralID(.u32(lower.bits), type: word, size: source.size)
            let minimum = try splatLiteralID(lower.value, type: result, size: source.size)
            limit = try rawValueInstruction(169, type: word, operands: [negative, negativeLimit, limit])
            saturation = try rawValueInstruction(169, type: result, operands: [negative, minimum, saturation])
            classification = try rawValueInstruction(199, type: word, operands: [bits, mask])
        } else {
            // For unsigned conversion, sign-bit values and positive NaNs both exceed +infinity.
            classification = bits
        }
        let outside = try rawValueInstruction(172, type: predicate, operands: [bits, limit])
        let safe = try rawValueInstruction(169, type: float, operands: [outside, zeroID(typeID: float), operand])
        let opcode: UInt32 = target.kind == .sint ? 110 : 109 // OpConvertFToS / OpConvertFToU
        let converted = try rawValueInstruction(opcode, type: result, operands: [safe])
        let bounded = try rawValueInstruction(169, type: result, operands: [outside, saturation, converted])
        let infinity = try splatLiteralID(.u32(bounds.infinity), type: word, size: source.size)
        let clear = try rawValueInstruction(172, type: predicate, operands: [classification, infinity])
        return try rawValueInstruction(169, type: result, operands: [clear, zeroID(typeID: result), bounded])
    }

    private mutating func floatConversionBits(_ operand: UInt32, source: IRBitcastShape, word: UInt32) throws -> UInt32 {
        if source.scalar == .f32 { return try rawValueInstruction(124, type: word, operands: [operand]) }
        // A pair of identical lanes preserves the half bits without requiring Int16.
        let f16 = try scalarTypeID(.f16)
        let pairType = try vectorTypeID(size: .two, scalar: .f16)
        let u32 = try scalarTypeID(.u32)
        let mask = try unsignedConstantID(0xffff)
        var lanes: [UInt32] = []
        for index in 0..<UInt32(source.size?.rawValue ?? 1) {
            let half = try source.size == nil ? operand : rawValueInstruction(81, type: f16, operands: [operand, index])
            let pair = try rawValueInstruction(80, type: pairType, operands: [half, half])
            let packed = try rawValueInstruction(124, type: u32, operands: [pair])
            lanes.append(try rawValueInstruction(199, type: u32, operands: [packed, mask]))
        }
        if source.size == nil { return lanes[0] }
        return try rawValueInstruction(80, type: word, operands: lanes)
    }
}
