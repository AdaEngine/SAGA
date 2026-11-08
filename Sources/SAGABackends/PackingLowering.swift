import SAGAShaderIR

func packingStatements(
    _ function: IRMathFunction, builtin: IRPackingBuiltin, operand: String, resultType: String,
    named name: String, format: ShaderOutputFormat
) throws -> [String] {
    // Some WebGPU backends round negative snorm ties down. WGSL requires floor(0.5 + x).
    if format == .wgsl && !(builtin.packs && builtin.encoding == .snorm) {
        return ["let \(name) = \(function.rawValue)(\(operand));"]
    }
    let input = "\(name)_input"
    let inputType = builtin.packs ? try shaderVectorType(builtin.size.rawValue, scalar: builtin.encoding.scalar, in: format) : "uint"
    let inputBinding = format == .wgsl ? "let \(input)" : "\(inputType) \(input)"
    let resultBinding = format == .wgsl ? "let \(name)" : "\(resultType) \(name)"
    let statement = "\(inputBinding) = \(operand);"
    let expression: String
    if builtin.encoding == .binary16 {
        switch (format, builtin.packs) {
        case (.msl, true): expression = "as_type<uint>(half2(\(input)))"
        case (.msl, false): expression = "float2(as_type<half2>(\(input)))"
        case (.hlsl, true): expression = "((f32tof16(\(input).x) & 65535u) | (f32tof16(\(input).y) << 16u))"
        case (.hlsl, false): expression = "float2(f16tof32(\(input) & 65535u), f16tof32(\(input) >> 16u))"
        case (.glsl, true): expression = "packHalf2x16(\(input))"
        case (.glsl, false): expression = "unpackHalf2x16(\(input))"
        default: throw ShaderFailure.unsupported("Binary16 packing target")
        }
        return [statement, "\(resultBinding) = \(expression);"]
    }

    let suffix = format == .msl || format == .hlsl ? "f" : ""
    let scale = "\(builtin.normalizedScale).0\(suffix)"
    if builtin.packs {
        var fields: [String] = []
        for (index, lane) in "xyzw".prefix(Int(builtin.size.rawValue)).enumerated() {
            let value = "\(input).\(lane)"
            let bits: String
            switch builtin.encoding {
            case .snorm, .unorm:
                let lower = builtin.encoding == .snorm ? "-1.0\(suffix)" : "0.0\(suffix)"
                let rounded = "floor(0.5\(suffix) + clamp(\(value), \(lower), 1.0\(suffix)) * \(scale))"
                let signedType = format == .wgsl ? "i32" : "int"
                bits = builtin.encoding == .snorm ?
                    integerBits("\(signedType)(\(rounded))", signed: false, in: format) : "uint(\(rounded))"
            case .sint: bits = integerBits(value, signed: false, in: format)
            case .sintClamp: bits = integerBits("clamp(\(value), -128, 127)", signed: false, in: format)
            case .uint: bits = value
            case .uintClamp: bits = "min(\(value), 255u)"
            case .binary16: throw ShaderFailure.invalidHandle
            }
            fields.append("((\(bits) & \(builtin.mask)u) << \(index * builtin.bitsPerLane)u)")
        }
        expression = fields.joined(separator: " | ")
    } else {
        var lanes: [String] = []
        for index in 0..<Int(builtin.size.rawValue) {
            let field = "((\(input) >> \(index * builtin.bitsPerLane)u) & \(builtin.mask)u)"
            let number: String
            if builtin.encoding.isSignedIntegerField {
                let sign = 1 << (builtin.bitsPerLane - 1)
                number = integerBits("((\(field) ^ \(sign)u) - \(sign)u)", signed: true, in: format)
            } else {
                number = field
            }
            switch builtin.encoding {
            case .snorm: lanes.append("max(-1.0\(suffix), float(\(number)) / \(scale))")
            case .unorm: lanes.append("(float(\(number)) / \(scale))")
            case .sint, .uint: lanes.append(number)
            default: throw ShaderFailure.invalidHandle
            }
        }
        expression = "\(resultType)(\(lanes.joined(separator: ", ")))"
    }
    return [statement, "\(resultBinding) = \(expression);"]
}
