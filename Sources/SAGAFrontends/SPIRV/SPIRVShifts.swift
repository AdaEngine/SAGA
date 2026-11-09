import SAGAShaderIR

extension SPVFunctionLowerer {
    mutating func lowerSPVShift(_ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        try i.count(4)
        let base = try value(i.a[2], i, &statements)
        let count = try value(i.a[3], i, &statements)
        let resultType = try module.type(i.a[0], i)
        guard let baseShape = IRBitcastShape(try module.kind(base.type, i)),
              let countShape = IRBitcastShape(try module.kind(count.type, i)),
              let resultShape = IRBitcastShape(module.types[resultType.index].kind),
              [baseShape.scalar, countShape.scalar, resultShape.scalar].allSatisfy({
                  $0.kind == .sint || $0.kind == .uint
              }), baseShape.size == countShape.size, baseShape.size == resultShape.size,
              baseShape.scalar.width == resultShape.scalar.width else {
            throw i.fail("Shift requires integer operands with matching lane counts and base/result widths")
        }

        let kind: IRScalarKind = i.op == 195 ? .sint : .uint
        let lane = IRScalar(kind: kind, width: baseShape.scalar.width)
        let shiftLane = IRScalar(kind: kind, width: max(lane.width, 4))
        let shiftType = try internLoweredType(IRBitcastShape(size: baseShape.size, scalar: shiftLane).type)
        var operand = base.expression
        if baseShape.scalar != lane {
            let type = try internLoweredType(IRBitcastShape(size: baseShape.size, scalar: lane).type)
            operand = try emitValue(type, .bitcast(expression: operand, type: type), &statements, i)
        }
        if lane != shiftLane {
            operand = try emitValue(shiftType, .cast(expression: operand, target: shiftLane), &statements, i)
        }

        let (amount, inRange) = try safeSPVShiftCount(count.expression, shape: countShape, bitWidth: UInt32(lane.width) * 8,
                                                      i, &statements)
        let operation: IRBinaryOperator = i.op == 196 ? .shiftLeft : .shiftRight
        let shifted = try emitValue(shiftType, .binary(op: operation, left: operand, right: amount), &statements, i)
        let zero = try expr(shiftType, .zeroValue(shiftType), i)
        let safe = IRExpressionKind.select(condition: inRange, accept: shifted, reject: zero)
        if lane == resultShape.scalar, lane == shiftLane {
            try result(i, safe, &statements)
            return
        }
        let value = try emitValue(shiftType, safe, &statements, i)
        if lane == resultShape.scalar {
            try result(i, .cast(expression: value, target: lane), &statements)
        } else {
            let bits: IRHandle<IRExpression>
            if lane == shiftLane {
                bits = value
            } else {
                let type = try internLoweredType(IRBitcastShape(size: baseShape.size, scalar: lane).type)
                bits = try emitValue(type, .cast(expression: value, target: lane), &statements, i)
            }
            try result(i, .bitcast(expression: bits, type: resultType), &statements)
        }
    }

    private mutating func safeSPVShiftCount(
        _ count: IRHandle<IRExpression>, shape: IRBitcastShape, bitWidth: UInt32,
        _ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> (amount: IRHandle<IRExpression>, inRange: IRHandle<IRExpression>) {
        let lane: IRScalar = shape.scalar.width == 8 ? .u64 : .u32
        let type = try internLoweredType(IRBitcastShape(size: shape.size, scalar: lane).type)
        let unsigned: IRHandle<IRExpression>
        if shape.scalar == lane {
            unsigned = count
        } else {
            let conversion: IRExpressionKind = shape.scalar.width == lane.width
                ? .bitcast(expression: count, type: type) : .cast(expression: count, target: lane)
            unsigned = try emitValue(type, conversion, &statements, i)
        }

        // Compare the full count before narrowing so an invalid wide count cannot alias a valid one.
        let boundType = try internLoweredType(.scalar(lane))
        let boundLiteral: IRLiteral = lane == .u64 ? .u64(UInt64(bitWidth)) : .u32(bitWidth)
        let bound = try expr(boundType, .literal(boundLiteral), i)
        let limit = try shape.size.map { try emitValue(type, .splat(size: $0, value: bound), &statements, i) } ?? bound
        let boolean = try internLoweredType(IRBitcastShape(size: shape.size, scalar: .bool).type)
        let inRange = try emitValue(boolean, .binary(op: .less, left: unsigned, right: limit), &statements, i)

        let amountType = try internLoweredType(IRBitcastShape(size: shape.size, scalar: .u32).type)
        let narrow = lane == .u32 ? unsigned : try emitValue(amountType, .cast(expression: unsigned, target: .u32), &statements, i)
        let zero = try expr(amountType, .zeroValue(amountType), i)
        let amount = try emitValue(amountType, .select(condition: inRange, accept: narrow, reject: zero), &statements, i)
        return (amount, inRange)
    }
}
