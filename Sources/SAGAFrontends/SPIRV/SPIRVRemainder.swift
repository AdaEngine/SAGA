import SAGAShaderIR

extension SPVFunctionLowerer {
    mutating func lowerSPVSpecialRemainder(_ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        try i.count(4)
        let resultType = try module.type(i.a[0], i)
        let resultKind = module.types[resultType.index].kind
        guard let shape = IRBitcastShape(resultKind) else {
            throw i.fail("Remainder result requires a scalar or vector")
        }
        if i.op == 139 {
            guard shape.scalar.kind == .sint || shape.scalar.kind == .uint else {
                throw i.fail("OpSMod requires an integer result")
            }
            let signed = IRBitcastShape(size: shape.size, scalar: .init(kind: .sint, width: shape.scalar.width))
            let signedType = try internLoweredType(signed.type)
            let left = try signedRemainderOperand(i.a[2], as: signed, type: signedType, i, &statements)
            let right = try signedRemainderOperand(i.a[3], as: signed, type: signedType, i, &statements)
            let modulus = try divisorSignedRemainder(left, right, type: signedType, width: shape.size, i, &statements)
            if resultKind == signed.type {
                try result(i, modulus, &statements)
            } else {
                let value = try emitValue(signedType, modulus, &statements, i)
                try result(i, .bitcast(expression: value, type: resultType), &statements)
            }
        } else {
            guard shape.scalar.kind == .float else { throw i.fail("Floating remainder requires a floating-point result") }
            let left = try typed(i.a[2], i.a[0], i, &statements)
            let right = try typed(i.a[3], i.a[0], i, &statements)
            if i.op == 140 {
                try result(i, .binary(op: .modulo, left: left, right: right), &statements)
            } else {
                let modulus = try divisorSignedRemainder(left, right, type: resultType, width: shape.size, i, &statements)
                try result(i, modulus, &statements)
            }
        }
    }

    private mutating func signedRemainderOperand(
        _ id: UInt32, as signed: IRBitcastShape, type: IRHandle<IRType>,
        _ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> IRHandle<IRExpression> {
        let operand = try value(id, i, &statements)
        guard let shape = IRBitcastShape(try module.kind(operand.type, i)),
              shape.scalar.kind == .sint || shape.scalar.kind == .uint,
              shape.size == signed.size, shape.scalar.width == signed.scalar.width else {
            throw i.fail("OpSMod operands require matching integer shapes and widths")
        }
        if shape.scalar.kind == .sint { return operand.expression }
        return try emitValue(type, .bitcast(expression: operand.expression, type: type), &statements, i)
    }

    private mutating func divisorSignedRemainder(
        _ left: IRHandle<IRExpression>, _ right: IRHandle<IRExpression>, type: IRHandle<IRType>, width: IRVectorSize?,
        _ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> IRExpressionKind {
        // A truncating remainder has the dividend's sign. Shift a nonzero remainder by one divisor
        // only in lanes whose signs differ. Exact multiples must stay zero.
        let conditionType = try internLoweredType(IRBitcastShape(size: width, scalar: .bool).type)
        let zero = try expr(type, .zeroValue(type), i)
        let remainder = try emitValue(type, .binary(op: .modulo, left: left, right: right), &statements, i)
        let nonzero = try emitValue(conditionType, .binary(op: .notEqual, left: remainder, right: zero), &statements, i)
        let negativeRemainder = try emitValue(conditionType, .binary(op: .less, left: remainder, right: zero), &statements, i)
        let negativeDivisor = try emitValue(conditionType, .binary(op: .less, left: right, right: zero), &statements, i)
        let different = try emitValue(conditionType, .binary(op: .notEqual, left: negativeRemainder, right: negativeDivisor),
                                      &statements, i)
        let adjusts = try emitValue(conditionType, .binary(op: .and, left: nonzero, right: different), &statements, i)
        let shifted = try emitValue(type, .binary(op: .add, left: remainder, right: right), &statements, i)
        return .select(condition: adjusts, accept: shifted, reject: remainder)
    }
}
