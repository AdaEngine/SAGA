import SAGAShaderIR

func numericBuiltinStatements(
    _ function: IRMathFunction, arguments input: [String], argumentTypes: [IRTypeKind],
    result: IRTypeKind, resultType outputType: String, named output: String, format: ShaderOutputFormat
) throws -> [String] {
    let outputScalar: IRScalar
    let width: UInt8?
    switch result {
    case .scalar(let lane): outputScalar = lane; width = nil
    case .vector(let size, let lane): outputScalar = lane; width = size.rawValue
    default: throw ShaderFailure.unsupported("Numeric builtin result shape")
    }
    let promote = format != .wgsl && outputScalar == .f16
    let scalar = promote ? IRScalar.f32 : outputScalar
    let resultType = promote ? floatType(width ?? 1, in: format) : outputType
    let name = promote ? "\(output)_wide" : output
    let arguments = promote ? zip(input, argumentTypes).map { value, type in
        let wideType: String
        if case .scalar = type { wideType = floatType(1, in: format) }
        else { wideType = resultType }
        return "\(wideType)(\(value))"
    } : input
    let expression: String
    var statements: [String] = []
    if function == .quantizeToF16, format != .wgsl {
        let argument = arguments[0]
        switch format {
        case .msl:
            expression = "\(resultType)(\(halfType(width ?? 1, in: .msl))(\(argument)))"
        case .hlsl:
            expression = "f16tof32(f32tof16(\(argument)))"
        case .glsl:
            let source = "\(name)_operand"
            statements.append("\(resultType) \(source) = \(argument);")
            func pair(_ value: String) -> String { "unpackHalf2x16(packHalf2x16(\(value)))" }
            switch width {
            case nil: expression = pair("vec2(\(source), 0.0)") + ".x"
            case 2: expression = pair(source)
            case 3: expression = "vec3(\(pair(source + ".xy")), \(pair(source + ".zz")).x)"
            case 4: expression = "vec4(\(pair(source + ".xy")), \(pair(source + ".zw")))"
            default: throw ShaderFailure.invalidHandle
            }
        default: throw ShaderFailure.unsupported("quantizeToF16 target")
        }
    } else if function == .clamp {
        // The ordered min/max form also defines dynamic integer clamps with inverted bounds.
        expression = "min(max(\(arguments[0]), \(arguments[1])), \(arguments[2]))"
    } else if format != .wgsl, function == .mix {
        let factor = "\(name)_factor"
        let factorType: String
        if case .scalar = argumentTypes[2] { factorType = "float" }
        else { factorType = resultType }
        statements.append("\(factorType) \(factor) = \(arguments[2]);")
        let one = format == .glsl ? "1.0" : "1.0f"
        expression = "(\(arguments[0])) * (\(one) - \(factor)) + (\(arguments[1])) * \(factor)"
    } else if format != .wgsl, function == .smoothStep {
        // GLSL smoothstep leaves reversed edges undefined. WGSL defines the same formula in either order.
        let edge = "\(name)_edge"
        let t = "\(name)_t"
        let suffix = format == .glsl ? "" : "f"
        let zero = format == .msl ? "\(resultType)(0.0f)" : "0.0\(suffix)"
        let one = format == .msl ? "\(resultType)(1.0f)" : "1.0\(suffix)"
        statements.append("\(resultType) \(edge) = \(arguments[0]);")
        statements.append("\(resultType) \(t) = clamp(((\(arguments[2])) - \(edge)) / ((\(arguments[1])) - \(edge)), \(zero), \(one));")
        expression = "\(t) * \(t) * (3.0\(suffix) - 2.0\(suffix) * \(t))"
    } else if format != .wgsl, function == .abs, scalar.kind == .uint {
        expression = arguments[0]
    } else if format != .wgsl, function == .abs, scalar == .i64 {
        let bits = "\(name)_bits", mask = "\(name)_mask"
        let type = try shaderVectorType(width ?? 1, scalar: .u64, in: format)
        let unsigned = try wideIntegerBits(arguments[0], signed: false, width: width ?? 1, in: format)
        statements.append("\(type) \(bits) = \(unsigned);")
        statements.append("\(type) \(mask) = \(unsigned64Literal(0, in: format)) - (\(bits) >> 63u);")
        expression = try wideIntegerBits("(\(bits) ^ \(mask)) - \(mask)", signed: true, width: width ?? 1, in: format)
    } else if format != .wgsl, function == .abs, scalar == .i32 {
        let bits = "\(name)_bits", mask = "\(name)_mask"
        let uintType = try width.map { try shaderVectorType($0, scalar: .u32, in: format) } ?? "uint"
        statements.append("\(uintType) \(bits) = \(integerBits(arguments[0], signed: false, width: width, in: format));")
        statements.append("\(uintType) \(mask) = 0u - (\(bits) >> 31u);")
        expression = integerBits("(\(bits) ^ \(mask)) - \(mask)", signed: true, width: width, in: format)
    } else if format != .wgsl, function == .sign, scalar == .i64 {
        let operand = "\(name)_operand"
        statements.append("\(resultType) \(operand) = \(arguments[0]);")
        let zero = "\(resultType)(\(signed64Literal(0, in: format)))"
        let positive = try wideIntegerBinary(.greater, scalar: scalar, width: width ?? 1, left: operand, right: zero, in: format)
        let negative = try wideIntegerBinary(.less, scalar: scalar, width: width ?? 1, left: operand, right: zero, in: format)
        expression = "(\(resultType)(\(positive)) - \(resultType)(\(negative)))"
    } else if format == .msl, function == .sign, scalar == .i16 || scalar == .i32 {
        let operand = "\(name)_operand"
        statements.append("\(resultType) \(operand) = \(arguments[0]);")
        let zero = "\(resultType)(0)"
        expression = "(\(resultType)(\(operand) > \(zero)) - \(resultType)(\(operand) < \(zero)))"
    } else if format == .glsl, function == .saturate {
        expression = "clamp(\(arguments[0]), 0.0, 1.0)"
    } else if format == .hlsl, [.asinh, .acosh, .atanh].contains(function) {
        let operand = "\(name)_operand"
        statements.append("\(resultType) \(operand) = \(arguments[0]);")
        switch function {
        case .asinh:
            let magnitude = "\(name)_magnitude", scale = "\(name)_scale", ratio = "\(name)_ratio", inverse = "\(name)_inverse"
            statements += [
                "\(resultType) \(magnitude) = abs(\(operand));",
                "\(resultType) \(scale) = max(\(magnitude), 1.0f);",
                "\(resultType) \(ratio) = \(magnitude) / \(scale);",
                "\(resultType) \(inverse) = 1.0f / \(scale);",
            ]
            expression = "\(resultType)(sign(\(operand))) * (log(\(scale)) + log(\(ratio) + sqrt(\(ratio) * \(ratio) + \(inverse) * \(inverse))))"
        case .acosh:
            let inverse = "\(name)_inverse"
            statements.append("\(resultType) \(inverse) = 1.0f / \(operand);")
            expression = "log(\(operand)) + log(1.0f + sqrt(1.0f - \(inverse) * \(inverse)))"
        default:
            expression = "0.5f * (log(1.0f + \(operand)) - log(1.0f - \(operand)))"
        }
    } else if format == .msl, function == .radians || function == .degrees {
        let factor = function == .radians ? "0.017453292519943295f" : "57.29577951308232f"
        expression = "(\(arguments[0])) * \(factor)"
    } else {
        let operation = switch (format, function) {
        case (.msl, .round): "rint"
        case (.glsl, .round): "roundEven"
        case (.hlsl, .fract): "frac"
        case (.hlsl, .fma): "mad"
        case (.hlsl, .inverseSqrt), (.msl, .inverseSqrt): "rsqrt"
        case (.glsl, .inverseSqrt): "inversesqrt"
        case (.glsl, .atan2): "atan"
        default: function.wgslName
        }
        let call = "\(operation)(\(arguments.joined(separator: ", ")))"
        expression = format == .hlsl && function == .sign && (scalar == .f32 || scalar == .i16)
            ? "\(resultType)(\(call))" : call
    }
    let binding = format == .wgsl ? "let \(name)" : "\(resultType) \(name)"
    statements.append("\(binding) = \(expression);")
    if promote {
        let value = format == .hlsl ? "(\(outputType))(\(name))" : "\(outputType)(\(name))"
        statements.append("\(outputType) \(output) = \(value);")
    }
    return statements
}
