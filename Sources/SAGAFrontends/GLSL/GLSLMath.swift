import SAGAShaderIR

extension GLSLParser {
    private static let glslMathFunctions: [String: IRMathFunction] = [
        "abs": .abs, "sign": .sign, "min": .min, "max": .max, "clamp": .clamp,
        "ceil": .ceil, "floor": .floor, "round": .round, "roundEven": .round, "fract": .fract, "trunc": .trunc,
        "sqrt": .sqrt, "inversesqrt": .inverseSqrt, "sin": .sin, "cos": .cos, "tan": .tan,
        "asin": .asin, "acos": .acos, "sinh": .sinh, "cosh": .cosh, "tanh": .tanh,
        "asinh": .asinh, "acosh": .acosh, "atanh": .atanh, "exp": .exp, "exp2": .exp2,
        "log": .log, "log2": .log2, "radians": .radians, "degrees": .degrees, "pow": .pow,
        "step": .step, "smoothstep": .smoothStep, "fma": .fma, "mix": .mix,
        "length": .length, "normalize": .normalize, "transpose": .transpose, "determinant": .determinant,
        "inverse": .inverse, "dot": .dot, "cross": .cross, "distance": .distance,
        "reflect": .reflect, "faceforward": .faceForward, "refract": .refract,
    ]
    private static let glslPackingFunctions: [String: IRMathFunction] = [
        "packSnorm4x8": .pack4x8snorm, "unpackSnorm4x8": .unpack4x8snorm,
        "packUnorm4x8": .pack4x8unorm, "unpackUnorm4x8": .unpack4x8unorm,
        "packSnorm2x16": .pack2x16snorm, "unpackSnorm2x16": .unpack2x16snorm,
        "packUnorm2x16": .pack2x16unorm, "unpackUnorm2x16": .unpack2x16unorm,
        "packHalf2x16": .pack2x16float, "unpackHalf2x16": .unpack2x16float,
    ]

    func glslMathBuiltin(_ token: GLSLToken, arguments: [GLSLValue]) throws -> GLSLValue? {
        let function: IRMathFunction
        switch token.text {
        case "atan": function = arguments.count == 2 ? .atan2 : .atan
        case "mod": return try glslFloatModulus(arguments, at: token)
        case "ldexp": return try glslLdexp(arguments, at: token)
        case "bitCount", "bitfieldReverse", "findLSB", "findMSB", "bitfieldExtract", "bitfieldInsert":
            return try glslBitBuiltin(arguments, at: token)
        default:
            if let packing = Self.glslPackingFunctions[token.text] {
                return try glslPackingBuiltin(packing, arguments: arguments, at: token)
            }
            guard let builtin = Self.glslMathFunctions[token.text] else { return nil }
            function = builtin
        }
        if let signature = function.numericBuiltin {
            guard arguments.count == signature.arity else {
                throw glslError(token, .typeMismatch, "Invalid math builtin argument count")
            }
            let anchor: Int
            let scalarGroups: [[Int]]
            switch function {
            case .min, .max: (anchor, scalarGroups) = (0, [[1]])
            case .clamp: (anchor, scalarGroups) = (0, [[1, 2]])
            case .step: (anchor, scalarGroups) = (1, [[0]])
            case .smoothStep: (anchor, scalarGroups) = (2, [[0, 1]])
            case .mix: (anchor, scalarGroups) = (0, [[2]])
            default: (anchor, scalarGroups) = (0, [])
            }
            let floats = signature.domain == .floating
            let unsigned = signature.domain == .ordered && function != .abs
            let (type, operands) = try glslNumericMathOperands(arguments, anchor: anchor, scalarGroups: scalarGroups,
                                                               floatOnly: floats, allowsUnsigned: unsigned, at: token)
            guard scalar(type) != .f64 || function.hasGLSLDoubleOverload else {
                throw glslError(token, .typeMismatch, "Math builtin has no double-precision overload")
            }
            return try glslMathExpression(function, type: type, arguments: operands, at: token)
        }
        return try glslGeometricBuiltin(function, arguments: arguments, at: token)
    }

    private func glslNumericMathOperands(
        _ arguments: [GLSLValue], anchor: Int, scalarGroups: [[Int]], floatOnly: Bool, allowsUnsigned: Bool,
        at token: GLSLToken
    ) throws -> (IRHandle<IRType>, [GLSLValue]) {
        let shapes = try arguments.map { argument -> IRBitcastShape in
            guard let shape = IRBitcastShape(kind(argument.type)), [.i32, .u32, .f32, .f64].contains(shape.scalar) else {
                throw glslError(token, .typeMismatch, "Expected numeric scalar or vector math operands")
            }
            return shape
        }
        let width = shapes[anchor].size
        var splats: Set<Int> = []
        for group in scalarGroups where width != nil && group.allSatisfy({ shapes[$0].size == nil }) {
            splats.formUnion(group)
        }
        guard shapes.indices.allSatisfy({ shapes[$0].size == width || splats.contains($0) }) else {
            throw glslError(token, .typeMismatch, "Math builtin operand shapes disagree")
        }
        let hasUnsigned = shapes.contains { $0.scalar == .u32 }
        let hasFloat = shapes.contains { $0.scalar == .f32 }
        let scalar: IRScalar = if shapes.contains(where: { $0.scalar == .f64 }) { .f64 }
            else if floatOnly || hasFloat || (hasUnsigned && !allowsUnsigned) { .f32 }
            else if hasUnsigned { .u32 }
            else { .i32 }
        let type = try addType(IRBitcastShape(size: width, scalar: scalar).type)
        let operands = try arguments.enumerated().map { index, argument in
            if let width, splats.contains(index) {
                let element = try glslMathConvert(argument, to: .scalar(scalar), at: token)
                return try expressionNode(type, .splat(size: width, value: element.expression), at: token)
            }
            return try glslMathConvert(argument, to: kind(type), at: token)
        }
        return (type, operands)
    }

    private func glslMathConvert(_ value: GLSLValue, to kind: IRTypeKind, at token: GLSLToken) throws -> GLSLValue {
        guard let converted = try implicitlyConvert(value, to: addType(kind), at: token) else {
            throw glslError(token, .typeMismatch, "Invalid implicit math builtin operand conversion")
        }
        return converted
    }

    private func glslMathExpression(
        _ function: IRMathFunction, type: IRHandle<IRType>, arguments: [GLSLValue], at token: GLSLToken
    ) throws -> GLSLValue {
        let operands = arguments.map(\.expression)
        return try expressionNode(type, .math(function: function, argument: operands[0],
            argument1: operands.count > 1 ? operands[1] : nil,
            argument2: operands.count > 2 ? operands[2] : nil,
            argument3: operands.count > 3 ? operands[3] : nil), at: token)
    }

    private func glslGeometricBuiltin(
        _ function: IRMathFunction, arguments: [GLSLValue], at token: GLSLToken
    ) throws -> GLSLValue {
        guard let first = arguments.first, arguments.count == function.geometricArity else {
            throw glslError(token, .typeMismatch, "Invalid geometric builtin argument count")
        }
        let input = kind(first.type)
        let operand: IRTypeKind
        if let shape = IRBitcastShape(input), [.f32, .i32, .u32].contains(shape.scalar) {
            operand = IRBitcastShape(size: shape.size, scalar: .f32).type
        } else {
            operand = input
        }
        guard let signature = function.geometricSignature(for: operand) else {
            throw glslError(token, .typeMismatch, "Unsupported geometric builtin operand shape")
        }
        let operands = try zip(arguments, signature.arguments).map { try glslMathConvert($0, to: $1, at: token) }
        return try glslMathExpression(function, type: addType(signature.result), arguments: operands, at: token)
    }

    private func glslFloatModulus(_ arguments: [GLSLValue], at token: GLSLToken) throws -> GLSLValue {
        guard arguments.count == 2 else { throw glslError(token, .typeMismatch, "mod requires two operands") }
        let (type, operands) = try glslNumericMathOperands(arguments, anchor: 0, scalarGroups: [[1]],
                                                           floatOnly: true, allowsUnsigned: false, at: token)
        let left = operands[0].expression
        let right = operands[1].expression
        let quotient = try expressionNode(type, .binary(op: .divide, left: left, right: right), at: token)
        let rounded = try glslMathExpression(.floor, type: type, arguments: [quotient], at: token)
        let multiple = try expressionNode(type, .binary(op: .multiply, left: right, right: rounded.expression), at: token)
        return try expressionNode(type, .binary(op: .subtract, left: left, right: multiple.expression), at: token)
    }

    private func glslLdexp(_ arguments: [GLSLValue], at token: GLSLToken) throws -> GLSLValue {
        guard arguments.count == 2, let shape = IRBitcastShape(kind(arguments[0].type)) else {
            throw glslError(token, .typeMismatch, "ldexp requires a mantissa and exponent")
        }
        let mantissaKind = IRBitcastShape(size: shape.size, scalar: shape.scalar == .f64 ? .f64 : .f32).type
        guard let exponentKind = mantissaKind.ldexpExponentType else { throw ShaderFailure.invalidHandle }
        let mantissa = try glslMathConvert(arguments[0], to: mantissaKind, at: token)
        let exponent = try glslMathConvert(arguments[1], to: exponentKind, at: token)
        return try glslMathExpression(.ldexp, type: mantissa.type, arguments: [mantissa, exponent], at: token)
    }

    private func glslBitBuiltin(_ arguments: [GLSLValue], at token: GLSLToken) throws -> GLSLValue {
        let function: IRMathFunction = switch token.text {
        case "bitCount": .countOneBits
        case "bitfieldReverse": .reverseBits
        case "findLSB": .firstTrailingBit
        case "findMSB": .firstLeadingBit
        case "bitfieldExtract": .extractBits
        default: .insertBits
        }
        guard let arity = function.integerBitArity, arguments.count == arity,
              let shape = IRBitcastShape(kind(arguments[0].type)),
              shape.scalar == .i32 || shape.scalar == .u32 else {
            throw glslError(token, .typeMismatch, "Invalid integer bit builtin operands or arity")
        }
        if arity > 1 {
            let data = Array(arguments.dropLast(2))
            guard data.allSatisfy({ scalar($0.type) == .i32 || scalar($0.type) == .u32 }),
                  arguments.suffix(2).allSatisfy({ kind($0.type) == .scalar(.i32) }) else {
                throw glslError(token, .typeMismatch, "Bitfield bounds require scalar int operands")
            }
            let (type, values) = try glslNumericMathOperands(data, anchor: 0, scalarGroups: [],
                                                            floatOnly: false, allowsUnsigned: true, at: token)
            let bounds = try arguments.suffix(2).map { try glslMathConvert($0, to: .scalar(.u32), at: token) }
            return try glslMathExpression(function, type: type, arguments: values + bounds, at: token)
        }
        let value = try glslMathExpression(function, type: arguments[0].type, arguments: arguments, at: token)
        guard function != .reverseBits, shape.scalar == .u32 else { return value }
        let type = try addType(IRBitcastShape(size: shape.size, scalar: .i32).type)
        return try expressionNode(type, .bitcast(expression: value.expression, type: type), at: token)
    }

    private func glslPackingBuiltin(
        _ function: IRMathFunction, arguments: [GLSLValue], at token: GLSLToken
    ) throws -> GLSLValue {
        guard let signature = function.packingBuiltin, arguments.count == 1 else {
            throw glslError(token, .typeMismatch, "Packing builtin requires one operand")
        }
        let operand = try glslMathConvert(arguments[0], to: signature.argumentType, at: token)
        return try glslMathExpression(function, type: addType(signature.resultType), arguments: [operand], at: token)
    }
}
