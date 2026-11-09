import SAGAShaderIR

extension SPVFunctionLowerer {
    mutating func lowerRelational(_ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        try i.count(3)
        let operand = try value(i.a[2], i, &statements)
        let operandType = try module.kind(operand.type, i)
        let resultType = try module.kind(i.a[0], i)
        switch i.op {
        case 154, 155:
            guard case .vector(_, .bool) = operandType, resultType == .scalar(.bool) else {
                throw i.fail("OpAny and OpAll require a bool vector operand and a scalar bool result")
            }
            try result(i, .relational(function: i.op == 154 ? .any : .all, argument: operand.expression), &statements)
        case 156, 157:
            guard let shape = SPVFloatWordShape.floatingShape(operandType),
                  resultType == IRBitcastShape(size: shape.size, scalar: .bool).type else {
                throw i.fail("OpIsNan and OpIsInf require a floating operand and a corresponding bool result")
            }
            let words = try floatWordShape(shape)
            let bits = try floatOperandBits(operand.expression, words, i, &statements)
            let mask = try floatWordConstant(words.sign - 1, words, i, &statements)
            let infinity = try floatWordConstant(words.infinity, words, i, &statements)
            // Ignoring sign makes both infinities equal to the threshold and every NaN larger.
            let magnitude = try emitValue(words.type, .binary(op: .and, left: bits, right: mask), &statements, i)
            try result(i, .binary(op: i.op == 156 ? .greater : .equal, left: magnitude, right: infinity), &statements)
        default: throw i.fail("Unsupported relational opcode", code: .unsupportedConstruct)
        }
    }
}
