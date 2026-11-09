import SAGAShaderIR

extension SPVFunctionLowerer {
    mutating func lowerQuantizeToF16(_ instruction: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        try instruction.count(3)
        let resultType = try module.type(instruction.a[0], instruction)
        guard let shape = IRBitcastShape(module.types[resultType.index].kind), shape.scalar == .f32 else {
            throw instruction.fail("OpQuantizeToF16 requires an f32 scalar or vector")
        }
        let input = try typed(instruction.a[2], instruction.a[0], instruction, &statements)
        let scalarType = try internLoweredType(.scalar(.u32))
        let wordType = try shape.size.map { try internLoweredType(.vector(size: $0, scalar: .u32)) } ?? scalarType
        let conditionType = try internLoweredType(IRBitcastShape(size: shape.size, scalar: .bool).type)

        func constant(_ word: UInt32, lowerer: inout Self, statements: inout [IRHandle<IRStatement>]) throws -> IRHandle<IRExpression> {
            let literal = try lowerer.expr(scalarType, .literal(.u32(word)), instruction)
            guard let size = shape.size else { return literal }
            return try lowerer.emitValue(wordType, .splat(size: size, value: literal), &statements, instruction)
        }
        let zero = try constant(0, lowerer: &self, statements: &statements)
        let signMask = try constant(0x80000000, lowerer: &self, statements: &statements)
        let fractionShift = try constant(13, lowerer: &self, statements: &statements)
        let one = try constant(1, lowerer: &self, statements: &statements)
        let roundBias = try constant(0xfff, lowerer: &self, statements: &statements)
        let retainedBits = try constant(0xffffe000, lowerer: &self, statements: &statements)
        let leastNormal = try constant(0x38800000, lowerer: &self, statements: &statements)
        let greatestFinite = try constant(0x477fe000, lowerer: &self, statements: &statements)
        let infinity = try constant(0x7f800000, lowerer: &self, statements: &statements)

        let bits = try emitValue(wordType, .bitcast(expression: input, type: wordType), &statements, instruction)
        let sign = try emitValue(wordType, .binary(op: .and, left: bits, right: signMask), &statements, instruction)
        let magnitude = try emitValue(wordType, .binary(op: .exclusiveOr, left: bits, right: sign), &statements, instruction)

        // Normal half values keep the f32 exponent and ten fraction bits. Round ties to even before masking.
        let retainedFraction = try emitValue(wordType, .binary(op: .shiftRight, left: magnitude, right: fractionShift), &statements, instruction)
        let parity = try emitValue(wordType, .binary(op: .and, left: retainedFraction, right: one), &statements, instruction)
        let bias = try emitValue(wordType, .binary(op: .add, left: roundBias, right: parity), &statements, instruction)
        let biased = try emitValue(wordType, .binary(op: .add, left: magnitude, right: bias), &statements, instruction)
        let rounded = try emitValue(wordType, .binary(op: .and, left: biased, right: retainedBits), &statements, instruction)

        // Classify the original bits so even values that round to a normal half are flushed when below the limit.
        let isNaN = try emitValue(conditionType, .binary(op: .greater, left: magnitude, right: infinity), &statements, instruction)
        let nonFinite = try emitValue(wordType, .select(condition: isNaN, accept: magnitude, reject: infinity), &statements, instruction)
        let tooLarge = try emitValue(conditionType, .binary(op: .greater, left: magnitude, right: greatestFinite), &statements, instruction)
        let bounded = try emitValue(wordType, .select(condition: tooLarge, accept: nonFinite, reject: rounded), &statements, instruction)
        let tooSmall = try emitValue(conditionType, .binary(op: .less, left: magnitude, right: leastNormal), &statements, instruction)
        let flushed = try emitValue(wordType, .select(condition: tooSmall, accept: zero, reject: bounded), &statements, instruction)
        let signedBits = try emitValue(wordType, .binary(op: .inclusiveOr, left: sign, right: flushed), &statements, instruction)
        try result(instruction, .bitcast(expression: signedBits, type: resultType), &statements)
    }

}
