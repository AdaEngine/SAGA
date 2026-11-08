import SAGAShaderIR

struct SPVGLSLShape: Hashable {
    let size: IRVectorSize?
    let scalar: IRScalar

    init(size: IRVectorSize? = nil, scalar: IRScalar) {
        self.size = size
        self.scalar = scalar
    }

    init?(_ kind: IRTypeKind) {
        guard let shape = IRBitcastShape(kind) else { return nil }
        self.init(size: shape.size, scalar: shape.scalar)
    }

    var kind: IRTypeKind {
        size.map { .vector(size: $0, scalar: scalar) } ?? .scalar(scalar)
    }

    var isInteger: Bool { scalar.kind == .sint || scalar.kind == .uint }
}

private enum SPVGLSL450 {
    // Instruction numbers from the GLSL.std.450 extended instruction grammar.
    static let functions: [UInt32: IRMathFunction] = [
        1: .round, 2: .round, 3: .trunc, 4: .abs, 5: .abs, 6: .sign, 7: .sign,
        8: .floor, 9: .ceil, 10: .fract, 11: .radians, 12: .degrees,
        13: .sin, 14: .cos, 15: .tan, 16: .asin, 17: .acos, 18: .atan,
        19: .sinh, 20: .cosh, 21: .tanh, 22: .asinh, 23: .acosh, 24: .atanh, 25: .atan2,
        26: .pow, 27: .exp, 28: .log, 29: .exp2, 30: .log2, 31: .sqrt, 32: .inverseSqrt, 33: .determinant, 34: .inverse,
        36: .modf, 37: .min, 38: .min, 39: .min, 40: .max, 41: .max, 42: .max, 43: .clamp, 44: .clamp, 45: .clamp,
        46: .mix, 48: .step, 49: .smoothStep, 50: .fma, 52: .frexp, 53: .ldexp,
        54: .pack4x8snorm, 55: .pack4x8unorm, 56: .pack2x16snorm, 57: .pack2x16unorm, 58: .pack2x16float,
        60: .unpack2x16snorm, 61: .unpack2x16unorm, 62: .unpack2x16float, 63: .unpack4x8snorm, 64: .unpack4x8unorm,
        66: .length, 67: .distance, 68: .cross, 69: .normalize, 70: .faceForward, 71: .reflect, 72: .refract,
        73: .firstTrailingBit, 74: .firstLeadingBit, 75: .firstLeadingBit,
    ]

    static func integerInterpretation(_ opcode: UInt32) -> IRScalarKind? {
        switch opcode {
        case 5, 7, 39, 42, 45, 74: .sint
        case 38, 41, 44, 73, 75: .uint
        default: nil
        }
    }
}

extension SPVDecoder {
    mutating func prepareGLSLMathTypes() throws {
        guard hasGLSLMath else { return }
        for (index, type) in types.enumerated() {
            try charge()
            if let shape = SPVGLSLShape(type.kind), glslMathTypes[shape] == nil {
                glslMathTypes[shape] = try IRHandle(index: index)
            }
        }
        for function in functions {
            for instruction in function.body {
                try charge()
                guard instruction.op == 12, let type = typeMap[instruction.a[0]],
                      let function = SPVGLSL450.functions[instruction.a[3]] else { continue }
                let opcode = instruction.a[3]
                let shape = SPVGLSLShape(types[type.index].kind)
                if function.decomposesFloat, case .structure(let members, _) = types[type.index].kind,
                   let first = members.first, let shape = types[first.type.index].kind.concreteFloatShape {
                    try ensureGLSLDecomposition(function, size: shape.size, scalar: shape.scalar)
                } else if let interpretation = SPVGLSL450.integerInterpretation(opcode), let shape, shape.isInteger {
                    try ensureGLSLType(.init(size: shape.size, scalar: .init(kind: interpretation, width: shape.scalar.width)))
                } else if function.packingBuiltin != nil {
                    try ensureGLSLType(.init(scalar: .u32))
                } else if function == .ldexp, let shape, shape.scalar.kind == .float {
                    try ensureGLSLType(.init(size: shape.size, scalar: .i32))
                } else if (69...72).contains(opcode), let shape, shape.size == nil, shape.scalar.kind == .float {
                    try ensureGLSLType(.init(size: .two, scalar: shape.scalar))
                }
            }
        }
    }

    private mutating func ensureGLSLType(_ shape: SPVGLSLShape) throws {
        guard glslMathTypes[shape] == nil else { return }
        try node()
        glslMathTypes[shape] = try IRHandle(index: types.count)
        types.append(.init(kind: shape.kind))
    }

    private mutating func ensureGLSLDecomposition(_ function: IRMathFunction, size: IRVectorSize?, scalar: IRScalar) throws {
        let kind = function.decompositionResult(size: size, scalar: scalar)
        guard !glslDecompositionTypes.contains(where: { $0.kind == kind }) else { return }
        let first = SPVGLSLShape(size: size, scalar: scalar)
        let second = SPVGLSLShape(size: size, scalar: function.decompositionSecondScalar(scalar))
        try ensureGLSLType(first)
        try ensureGLSLType(second)
        guard let firstType = glslMathTypes[first], let secondType = glslMathTypes[second] else {
            throw ShaderFailure.invalidHandle
        }
        let stride = function.decompositionMemberStride(size: size, scalar: scalar)
        let members: [IRStructMember] = [
            .init(name: "fract", type: firstType, offset: 0),
            .init(name: function.decompositionSecondMember, type: secondType, offset: stride),
        ]
        try node()
        let result = try IRHandle<IRType>(index: types.count)
        types.append(.init(kind: .structure(members: members, span: stride * 2)))
        glslDecompositionTypes.append(.init(kind: kind, type: result))
    }
}

extension SPVFunctionLowerer {
    mutating func glslMath(_ instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        try instruction.minimum(4)
        let a = instruction.a
        guard module.glslInstructionSets.contains(a[2]) else { throw instruction.fail("OpExtInst has an invalid instruction set") }
        let opcode = a[3]
        guard let function = SPVGLSL450.functions[opcode] else {
            throw instruction.fail("GLSL.std.450 instruction \(opcode) is not implemented", code: .unsupportedConstruct)
        }
        let arity = function.numericBuiltin?.arity ?? function.geometricArity ?? function.integerBitArity ?? (function == .ldexp ? 2 : 1)
        try instruction.count(4 + arity)
        let resultType = try module.kind(a[0], instruction)
        var handles: [IRHandle<IRExpression>] = []
        var operandTypes: [IRTypeKind] = []
        for id in a.dropFirst(4) {
            let operand = try value(id, instruction, &statements)
            handles.append(operand.expression)
            operandTypes.append(try module.kind(operand.type, instruction))
        }

        if function.decomposesFloat {
            try glslDecomposition(function, operand: handles[0], operandType: operandTypes[0],
                                  declared: resultType, &statements, instruction)
        } else if let interpretation = SPVGLSL450.integerInterpretation(opcode) {
            guard let shape = SPVGLSLShape(resultType), shape.isInteger,
                  !(73...75).contains(opcode) || shape.scalar.width == 4 else {
                throw instruction.fail("Invalid GLSL integer builtin result shape")
            }
            let expected = SPVGLSLShape(size: shape.size, scalar: .init(kind: interpretation, width: shape.scalar.width))
            for index in handles.indices {
                handles[index] = try glslIntegerBits(handles[index], operandTypes[index], as: expected, &statements, instruction)
            }
            try glslMathResult(function, handles, shape: expected, declared: resultType, &statements, instruction)
        } else if let packing = function.packingBuiltin {
            let word = SPVGLSLShape(scalar: .u32)
            if packing.packs {
                guard operandTypes[0] == packing.argumentType, let shape = SPVGLSLShape(resultType),
                      shape.isInteger, shape.scalar.width == 4, shape.size == nil else {
                    throw instruction.fail("Invalid GLSL packing operand or result")
                }
                try glslMathResult(function, handles, shape: word, declared: resultType, &statements, instruction)
            } else {
                guard resultType == packing.resultType else { throw instruction.fail("Invalid GLSL unpacking result") }
                handles[0] = try glslIntegerBits(handles[0], operandTypes[0], as: word, &statements, instruction)
                try result(instruction, glslMathExpression(function, handles), &statements)
            }
        } else if function == .ldexp {
            guard let shape = resultType.concreteFloatShape, resultType == operandTypes[0] else {
                throw instruction.fail("Invalid GLSL Ldexp mantissa or result")
            }
            let exponent = SPVGLSLShape(size: shape.size, scalar: .i32)
            handles[1] = try glslIntegerBits(handles[1], operandTypes[1], as: exponent, &statements, instruction)
            try result(instruction, glslMathExpression(function, handles), &statements)
        } else if (69...72).contains(opcode), case .scalar(let scalar) = operandTypes[0],
                  scalar == .f32 || scalar == .f16 {
            try glslScalarGeometry(function, handles, operandTypes, scalar, resultType, &statements, instruction)
        } else if let signature = function.geometricSignature(for: operandTypes[0]) {
            guard signature.arguments == operandTypes, signature.result == resultType else {
                throw instruction.fail("Invalid GLSL geometric builtin operands or result")
            }
            try result(instruction, glslMathExpression(function, handles), &statements)
        } else {
            guard function.numericBuiltin != nil, let shape = SPVGLSLShape(resultType), shape.scalar.kind == .float,
                  !(11...32).contains(opcode) || shape.scalar.width <= 4,
                  operandTypes.allSatisfy({ $0 == resultType }) else {
                throw instruction.fail("Invalid GLSL floating builtin operands or result")
            }
            try result(instruction, glslMathExpression(function, handles), &statements)
        }
    }

    private func glslMathExpression(_ function: IRMathFunction, _ arguments: [IRHandle<IRExpression>]) -> IRExpressionKind {
        .math(function: function, argument: arguments[0],
              argument1: arguments.indices.contains(1) ? arguments[1] : nil,
              argument2: arguments.indices.contains(2) ? arguments[2] : nil, argument3: nil)
    }

    private mutating func glslDecomposition(
        _ function: IRMathFunction, operand: IRHandle<IRExpression>, operandType: IRTypeKind,
        declared: IRTypeKind, _ statements: inout [IRHandle<IRStatement>], _ instruction: SPVInstruction
    ) throws {
        guard let shape = operandType.concreteFloatShape, case .structure(let members, _) = declared,
              members.count == 2, module.types[members[0].type.index].kind == operandType,
              let second = SPVGLSLShape(module.types[members[1].type.index].kind) else {
            throw instruction.fail("Invalid GLSL decomposition result or operand")
        }
        let secondShape = SPVGLSLShape(size: shape.size, scalar: function.decompositionSecondScalar(shape.scalar))
        if function == .frexp {
            guard second.isInteger, second.scalar.width == 4, second.size == shape.size else {
                throw instruction.fail("FrexpStruct requires a matching 32-bit integer exponent")
            }
        } else if second != secondShape {
            throw instruction.fail("ModfStruct requires a matching whole-part type")
        }
        let kind = function.decompositionResult(size: shape.size, scalar: shape.scalar)
        guard let canonical = module.glslDecompositionTypes.first(where: { $0.kind == kind }) else {
            throw instruction.fail("Missing GLSL decomposition type")
        }
        let parts = try expr(canonical.type, glslMathExpression(function, [operand]), instruction)
        statements.append(try stmt(.emit([parts]), instruction))
        let fraction = try glslTemporary(.init(size: shape.size, scalar: shape.scalar),
                                         .accessIndex(base: parts, index: 0), &statements, instruction)
        var other = try glslTemporary(secondShape, .accessIndex(base: parts, index: 1), &statements, instruction)
        if second != secondShape {
            other = try glslIntegerBits(other, secondShape.kind, as: second, &statements, instruction)
        }
        // Keep the source structure's fields and layout separate from the builtin result type.
        let type = try module.type(instruction.a[0], instruction)
        try result(instruction, .compose(type: type, components: [fraction, other]), &statements)
    }

    private mutating func glslTemporary(
        _ shape: SPVGLSLShape, _ kind: IRExpressionKind,
        _ statements: inout [IRHandle<IRStatement>], _ instruction: SPVInstruction
    ) throws -> IRHandle<IRExpression> {
        guard let type = module.glslMathTypes[shape] else { throw instruction.fail("Missing GLSL builtin type") }
        let expression = try expr(type, kind, instruction)
        statements.append(try stmt(.emit([expression]), instruction))
        return expression
    }

    private mutating func glslIntegerBits(
        _ value: IRHandle<IRExpression>, _ declared: IRTypeKind, as expected: SPVGLSLShape,
        _ statements: inout [IRHandle<IRStatement>], _ instruction: SPVInstruction
    ) throws -> IRHandle<IRExpression> {
        guard let shape = SPVGLSLShape(declared), shape.isInteger,
              shape.size == expected.size, shape.scalar.width == expected.scalar.width else {
            throw instruction.fail("GLSL integer builtin needs matching integer operand widths and shapes")
        }
        if shape == expected { return value }
        guard let type = module.glslMathTypes[expected] else { throw instruction.fail("Missing GLSL integer builtin type") }
        return try glslTemporary(expected, .bitcast(expression: value, type: type), &statements, instruction)
    }

    private mutating func glslMathResult(
        _ function: IRMathFunction, _ arguments: [IRHandle<IRExpression>],
        shape: SPVGLSLShape, declared: IRTypeKind, _ statements: inout [IRHandle<IRStatement>], _ instruction: SPVInstruction
    ) throws {
        let math = glslMathExpression(function, arguments)
        if declared == shape.kind {
            try result(instruction, math, &statements)
        } else {
            let computed = try glslTemporary(shape, math, &statements, instruction)
            let resultType = try module.type(instruction.a[0], instruction)
            try result(instruction, .bitcast(expression: computed, type: resultType), &statements)
        }
    }

    private mutating func glslScalarGeometry(
        _ function: IRMathFunction, _ arguments: [IRHandle<IRExpression>], _ types: [IRTypeKind],
        _ scalar: IRScalar, _ declared: IRTypeKind,
        _ statements: inout [IRHandle<IRStatement>], _ instruction: SPVInstruction
    ) throws {
        guard declared == .scalar(scalar), types.allSatisfy({ $0 == declared }) else {
            throw instruction.fail("Invalid GLSL scalar geometric builtin operands or result")
        }
        let pair = SPVGLSLShape(size: .two, scalar: scalar)
        guard let type = module.glslMathTypes[pair] else { throw instruction.fail("Missing GLSL vector type") }
        let scalarType = try module.type(instruction.a[0], instruction)
        let zero = try expr(scalarType, .zeroValue(scalarType), instruction)
        var promoted: [IRHandle<IRExpression>] = []
        for (index, operand) in arguments.enumerated() {
            if function == .refract, index == 2 { promoted.append(operand) }
            else {
                promoted.append(try glslTemporary(pair, .compose(type: type, components: [operand, zero]), &statements, instruction))
            }
        }
        let computed = try glslTemporary(pair, glslMathExpression(function, promoted), &statements, instruction)
        try result(instruction, .accessIndex(base: computed, index: 0), &statements)
    }
}
