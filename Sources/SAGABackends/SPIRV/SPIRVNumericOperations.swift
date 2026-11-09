import SAGAShaderIR

extension SPIRVBinaryWriter {
    mutating func writeNumericBuiltin(
        _ builtin: IRMathFunction, operands: [IRHandle<IRExpression>], type: IRHandle<IRType>,
        in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        guard let signature = builtin.numericBuiltin, operands.count == signature.arity,
              let shape = IRBitcastShape(try module.types.value(for: type).kind) else {
            throw ShaderFailure.invalidHandle
        }
        var arguments: [UInt32] = []
        for operand in operands { arguments.append(try value(operand, in: &state)) }
        let first = arguments[0]
        let result = try typeID(type)
        if builtin == .quantizeToF16 {
            return try rawValueInstruction(116, type: result, operands: [first]) // OpQuantizeToF16
        }
        if builtin == .abs, shape.scalar.kind == .uint { return first }

        let extrema: (minimum: UInt32, maximum: UInt32) = switch shape.scalar.kind {
        case .sint: (39, 42)
        case .uint: (38, 41)
        default: (37, 40)
        }
        switch builtin {
        case .clamp:
            // WGSL permits inverted runtime bounds. GLSL.std.450 Clamp is poison in that case.
            let lower = try glslMathInstruction(extrema.maximum, type: result, operands: [first, arguments[1]])
            return try glslMathInstruction(extrema.minimum, type: result, operands: [lower, arguments[2]])
        case .saturate:
            return try clampToUnit(first, shape: shape, type: result)
        case .smoothStep:
            // WGSL defines descending edges as well. Emit its interpolation formula.
            let shifted = try rawValueInstruction(131, type: result, operands: [arguments[2], first]) // OpFSub
            let span = try rawValueInstruction(131, type: result, operands: [arguments[1], first])
            let ratio = try rawValueInstruction(136, type: result, operands: [shifted, span]) // OpFDiv
            let t = try clampToUnit(ratio, shape: shape, type: result)
            let two = try splatLiteralID(shape.scalar == .f16 ? .f16Bits(0x4000) : .f32(2), type: result, size: shape.size)
            let three = try splatLiteralID(shape.scalar == .f16 ? .f16Bits(0x4200) : .f32(3), type: result, size: shape.size)
            let twice = try rawValueInstruction(133, type: result, operands: [two, t]) // OpFMul
            let factor = try rawValueInstruction(131, type: result, operands: [three, twice])
            let squared = try rawValueInstruction(133, type: result, operands: [t, t])
            return try rawValueInstruction(133, type: result, operands: [squared, factor])
        case .mix:
            if let size = shape.size, case .scalar = try valueKind(operands[2], in: state) {
                arguments[2] = try rawValueInstruction(80, type: result,
                    operands: Array(repeating: arguments[2], count: Int(size.rawValue))) // OpCompositeConstruct
            }
            return try glslMathInstruction(46, type: result, operands: arguments)
        default:
            break
        }

        let instruction: UInt32 = switch builtin {
        case .round: 2 // RoundEven
        case .trunc: 3
        case .abs: shape.scalar.kind == .sint ? 5 : 4
        case .sign: shape.scalar.kind == .sint ? 7 : 6
        case .floor: 8
        case .ceil: 9
        case .fract: 10
        case .radians: 11
        case .degrees: 12
        case .sin: 13
        case .cos: 14
        case .tan: 15
        case .asin: 16
        case .acos: 17
        case .atan: 18
        case .sinh: 19
        case .cosh: 20
        case .tanh: 21
        case .asinh: 22
        case .acosh: 23
        case .atanh: 24
        case .atan2: 25
        case .pow: 26
        case .exp: 27
        case .log: 28
        case .exp2: 29
        case .log2: 30
        case .sqrt: 31
        case .inverseSqrt: 32
        case .min: extrema.minimum
        case .max: extrema.maximum
        case .step: 48
        case .fma: 50
        default: throw unsupported("does not lower this numeric builtin")
        }
        return try glslMathInstruction(instruction, type: result, operands: arguments)
    }

    mutating func writeDot(
        _ left: IRHandle<IRExpression>, _ right: IRHandle<IRExpression>, type: IRHandle<IRType>,
        in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        let vector = try state.function.expressions.value(for: left).type
        guard case .vector(let size, let scalar) = try module.types.value(for: vector).kind else {
            throw ShaderFailure.invalidHandle
        }
        let operands = try [value(left, in: &state), value(right, in: &state)]
        if scalar.kind == .float {
            return try valueInstruction(148, type: type, operands: operands) // OpDot
        }
        let products = try valueInstruction(132, type: vector, operands: operands) // OpIMul
        let result = try typeID(type)
        var sum = try rawValueInstruction(81, type: result, operands: [products, 0]) // OpCompositeExtract
        for lane in 1..<UInt32(size.rawValue) {
            let product = try rawValueInstruction(81, type: result, operands: [products, lane])
            sum = try rawValueInstruction(128, type: result, operands: [sum, product]) // OpIAdd
        }
        return sum
    }

    mutating func writeGeometry(
        _ builtin: IRMathFunction, operands: [IRHandle<IRExpression>], type: IRHandle<IRType>,
        in state: inout SPIRVFunctionState
    ) throws -> UInt32 {
        guard let first = operands.first, builtin.geometricArity == operands.count,
              let shape = IRBitcastShape(try valueKind(first, in: state)) else {
            throw ShaderFailure.invalidHandle
        }
        let instruction: UInt32 = switch builtin {
        case .length: 66
        case .distance: 67
        case .cross: 68
        case .normalize: 69
        case .faceForward: 70
        case .reflect: 71
        case .refract: 72
        default: throw unsupported("does not lower this geometric builtin")
        }
        let result = try typeID(type)
        var arguments: [UInt32] = []
        for operand in operands { arguments.append(try value(operand, in: &state)) }
        let promote = shape.scalar == .f16 && [.length, .distance, .normalize].contains(builtin)
        let intermediate: UInt32
        if promote {
            // Half-precision squaring can overflow or underflow before taking a norm.
            let wide = try scalarOrVectorTypeID(.f32, size: shape.size)
            for index in arguments.indices {
                arguments[index] = try rawValueInstruction(115, type: wide, operands: [arguments[index]]) // OpFConvert
            }
            intermediate = try builtin == .normalize ? wide : scalarTypeID(.f32)
        } else {
            intermediate = result
        }
        let evaluated: UInt32
        if shape.size == nil, builtin == .length || builtin == .distance {
            let difference = try builtin == .length ? arguments[0] :
                rawValueInstruction(131, type: intermediate, operands: arguments) // OpFSub
            evaluated = try glslMathInstruction(4, type: intermediate, operands: [difference]) // FAbs
        } else {
            evaluated = try glslMathInstruction(instruction, type: intermediate, operands: arguments)
        }
        return try promote ? rawValueInstruction(115, type: result, operands: [evaluated]) : evaluated
    }

    mutating func glslMathInstruction(_ instruction: UInt32, type: UInt32, operands: [UInt32]) throws -> UInt32 {
        try rawValueInstruction(12, type: type, operands: [glslInstructionSetID(), instruction] + operands)
    }

    private mutating func clampToUnit(_ value: UInt32, shape: IRBitcastShape, type: UInt32) throws -> UInt32 {
        let zero = try zeroID(typeID: type)
        let one = try splatLiteralID(shape.scalar == .f16 ? .f16Bits(0x3c00) : .f32(1), type: type, size: shape.size)
        return try glslMathInstruction(43, type: type, operands: [value, zero, one]) // FClamp
    }
}
