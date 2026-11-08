import SAGAShaderIR

func integerBitStatements(
    _ function: IRMathFunction, arguments: [String], type: String, scalar: IRScalar, width: UInt8?,
    named name: String, format: ShaderOutputFormat
) throws -> [String] {
    guard let arity = function.integerBitArity, arity == arguments.count else {
        throw ShaderFailure.unsupported("Integer bit builtin arity")
    }
    if format == .wgsl {
        return ["let \(name) = \(function.rawValue)(\(arguments.joined(separator: ", ")));"]
    }
    let signed = scalar.kind == .sint
    let wide = scalar.width == 8
    func unsignedBits(_ operand: String) throws -> String {
        guard signed else { return operand }
        if wide { return try wideIntegerBits(operand, signed: false, width: width ?? 1, in: format) }
        return integerBits(operand, signed: false, width: width, in: format)
    }
    func resultBits(_ operand: String) throws -> String {
        guard signed else { return operand }
        if wide { return try wideIntegerBits(operand, signed: true, width: width ?? 1, in: format) }
        return integerBits(operand, signed: true, width: width, in: format)
    }
    let value = arguments[0]
    let bits = try unsignedBits(value)
    if arity == 1 {
        if wide {
            return try wideIntegerBitUnary(function, bits: bits, type: type, signed: signed, width: width,
                                           named: name, format: format)
        }
        let glslUint = width.map { "uvec\($0)" } ?? "uint"
        let expression: String
        switch (format, function) {
        case (.msl, .countLeadingZeros): expression = "clz(\(bits))"
        case (.msl, .countTrailingZeros): expression = "ctz(\(bits))"
        case (.msl, .countOneBits): expression = "popcount(\(bits))"
        case (.msl, .reverseBits): expression = "reverse_bits(\(bits))"
        case (.msl, .firstLeadingBit):
            let search = signed ? "select(\(bits), ~(\(bits)), (\(value)) < \(type)(0))" : bits
            expression = "(31u - clz(\(search)))"
        case (.msl, .firstTrailingBit): expression = "(((ctz(\(bits)) + 1u) % 33u) - 1u)"
        case (.hlsl, .countLeadingZeros): expression = "(31u - firstbithigh(\(bits)))"
        case (.hlsl, .countTrailingZeros): expression = "min(firstbitlow(\(bits)), 32u)"
        case (.hlsl, .countOneBits): expression = "countbits(\(bits))"
        case (.hlsl, .reverseBits): expression = "reversebits(\(bits))"
        case (.hlsl, .firstLeadingBit): expression = "firstbithigh(\(value))"
        case (.hlsl, .firstTrailingBit): expression = "firstbitlow(\(bits))"
        case (.glsl, .countLeadingZeros): expression = "(31u - \(glslUint)(findMSB(\(bits))))"
        case (.glsl, .countTrailingZeros): expression = "min(\(glslUint)(findLSB(\(bits))), 32u)"
        case (.glsl, .countOneBits): expression = "\(glslUint)(bitCount(\(bits)))"
        case (.glsl, .reverseBits): expression = "bitfieldReverse(\(bits))"
        case (.glsl, .firstLeadingBit): expression = "\(glslUint)(findMSB(\(value)))"
        case (.glsl, .firstTrailingBit): expression = "\(glslUint)(findLSB(\(bits)))"
        default: throw ShaderFailure.unsupported("Integer bit builtin target")
        }
        return ["\(type) \(name) = \(try resultBits(expression));"]
    }

    let offset = "\(name)_offset"
    let count = "\(name)_count"
    let shift = "\(name)_shift"
    let mask = "\(name)_mask"
    let bitWidth = Int(scalar.width) * 8
    let lastBit = bitWidth - 1
    let maskType = wide ? unsigned64Type(in: format) : "uint"
    let zero = wide ? unsigned64Literal(0, in: format) : "0u"
    let one = wide ? unsigned64Literal(1, in: format) : "1u"
    let allBits = wide ? unsigned64Literal(.max, in: format) : "0xffffffffu"
    // Clamp the requested range and keep every shift defined, including empty and full-width fields.
    var statements = [
        "uint \(offset) = min(\(arguments[arity - 2]), \(bitWidth)u);",
        "uint \(count) = min(\(arguments[arity - 1]), \(bitWidth)u - \(offset));",
        "uint \(shift) = min(\(offset), \(lastBit)u);",
        "\(maskType) \(mask) = \(count) == 0u ? \(zero) : (\(allBits) >> ((\(bitWidth)u - \(count)) & \(lastBit)u));",
    ]
    let expression: String
    if function == .extractBits {
        let extracted = "(((\(bits)) >> \(shift)) & \(mask))"
        if signed {
            let sign = "\(name)_sign"
            statements.append("\(maskType) \(sign) = (\(one) << ((\(count) - 1u) & \(lastBit)u)) & \(mask);")
            expression = "((\(extracted) ^ \(sign)) - \(sign))"
        } else {
            expression = extracted
        }
    } else {
        let inserted = try unsignedBits(arguments[1])
        expression = "(((\(bits)) & ~(\(mask) << \(shift))) | (((\(inserted)) & \(mask)) << \(shift)))"
    }
    statements.append("\(type) \(name) = \(try resultBits(expression));")
    return statements
}

private func wideIntegerBitUnary(
    _ function: IRMathFunction, bits: String, type: String, signed: Bool, width: UInt8?,
    named name: String, format: ShaderOutputFormat
) throws -> [String] {
    let wideType = try shaderVectorType(width ?? 1, scalar: .u64, in: format)
    let wordType = try shaderVectorType(width ?? 1, scalar: .u32, in: format)
    let wideScalar = unsigned64Type(in: format)
    let source = "\(name)_bits"
    let low = "\(name)_low", high = "\(name)_high"
    let lowResult = "\(name)_low_result", highResult = "\(name)_high_result"
    var statements = ["\(wideType) \(source) = \(bits);"]
    if signed, function == .firstLeadingBit {
        let zero = unsigned64Literal(0, in: format)
        statements.append("\(source) ^= \(zero) - (\(source) >> 63u);")
    }
    statements += [
        "\(wordType) \(low) = \(wordType)(\(source));",
        "\(wordType) \(high) = \(wordType)(\(source) >> 32u);",
    ]
    // The word intrinsics are available across all native targets.
    statements += try integerBitStatements(function, arguments: [low], type: wordType, scalar: .u32, width: width,
                                           named: lowResult, format: format)
    statements += try integerBitStatements(function, arguments: [high], type: wordType, scalar: .u32, width: width,
                                           named: highResult, format: format)
    let missing = unsigned64Literal(.max, in: format)
    let lanes = width.map { "xyzw".prefix(Int($0)).map { ".\($0)" } } ?? [""]
    let values = try lanes.map { lane -> String in
        let lo = low + lane, hi = high + lane
        let a = lowResult + lane, b = highResult + lane
        switch function {
        case .countLeadingZeros: return "\(wideScalar)(\(hi) == 0u ? 32u + \(a) : \(b))"
        case .countTrailingZeros: return "\(wideScalar)(\(lo) == 0u ? 32u + \(b) : \(a))"
        case .countOneBits: return "\(wideScalar)(\(a) + \(b))"
        case .reverseBits: return "(\(wideScalar)(\(b)) | (\(wideScalar)(\(a)) << 32u))"
        case .firstLeadingBit:
            return "(\(hi) != 0u ? \(wideScalar)(32u + \(b)) : (\(lo) != 0u ? \(wideScalar)(\(a)) : \(missing)))"
        case .firstTrailingBit:
            return "(\(lo) != 0u ? \(wideScalar)(\(a)) : (\(hi) != 0u ? \(wideScalar)(32u + \(b)) : \(missing)))"
        default: throw ShaderFailure.invalidHandle
        }
    }
    let combined = width == nil ? values[0] : "\(wideType)(\(values.joined(separator: ", ")))"
    let value = signed ? try wideIntegerBits(combined, signed: true, width: width ?? 1, in: format) : combined
    statements.append("\(type) \(name) = \(value);")
    return statements
}
