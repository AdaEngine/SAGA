import SAGAShaderIR

struct SPVFloatWordShape {
    let source: IRBitcastShape
    let scalarType: IRHandle<IRType>
    let type: IRHandle<IRType>

    var sign: UInt64 { UInt64(1) << (Int(source.scalar.width) * 8 - 1) }

    var infinity: UInt64 {
        switch source.scalar.width {
        case 2: 0x7c00
        case 4: 0x7f800000
        default: 0x7ff0000000000000
        }
    }

    static func floatingShape(_ kind: IRTypeKind) -> IRBitcastShape? {
        guard let shape = IRBitcastShape(kind), shape.scalar.kind == .float, [2, 4, 8].contains(shape.scalar.width) else {
            return nil
        }
        return shape
    }
}

extension SPVFunctionLowerer {
    mutating func lowerFloatComparison(_ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        try i.count(4)
        let left = try value(i.a[2], i, &statements)
        let right = try typed(i.a[3], left.type, i, &statements)
        let resultType = try module.type(i.a[0], i)
        guard let shape = SPVFloatWordShape.floatingShape(try module.kind(left.type, i)),
              module.types[resultType.index].kind == IRBitcastShape(size: shape.size, scalar: .bool).type else {
            throw i.fail("Floating comparison requires matching floating operands and a corresponding bool result")
        }
        let operation: IRBinaryOperator
        switch i.op {
        case 180, 181: operation = .equal
        case 182, 183: operation = .notEqual
        case 184, 185: operation = .less
        case 186, 187: operation = .greater
        case 188, 189: operation = .lessEqual
        case 190, 191: operation = .greaterEqual
        default: throw i.fail("Unsupported floating comparison opcode", code: .unsupportedConstruct)
        }

        let words = try floatWordShape(shape)
        let sign = try floatWordConstant(words.sign, words, i, &statements)
        let magnitudeMask = try floatWordConstant(words.sign - 1, words, i, &statements)
        let infinityBits = try floatWordConstant(words.infinity, words, i, &statements)

        func key(_ operand: IRHandle<IRExpression>, lowerer: inout Self,
                 statements: inout [IRHandle<IRStatement>]) throws -> (orderedBits: IRHandle<IRExpression>, isNaN: IRHandle<IRExpression>) {
            let converted: IRHandle<IRExpression>
            if words.source.scalar == .f16 {
                // Half packing and the ordering keys reuse their operands. Keep the expansion
                // in separate expressions so generated WGSL does not nest all that work.
                let source = lowerer.expressions[operand.index].type
                let snapshot = try lowerer.emitValue(source, .runtimeValue(operand), &statements, i)
                converted = try lowerer.floatOperandBits(snapshot, words, i, &statements)
            } else {
                converted = try lowerer.floatOperandBits(operand, words, i, &statements)
            }
            let bits = try lowerer.emitValue(words.type, .runtimeValue(converted), &statements, i)
            let magnitude = try lowerer.emitValue(words.type, .binary(op: .and, left: bits, right: magnitudeMask), &statements, i)
            let isNaN = try lowerer.emitValue(resultType, .binary(op: .greater, left: magnitude, right: infinityBits), &statements, i)
            // Both signed zeros map to the same key. Negative magnitudes reverse the unsigned order.
            let negative = try lowerer.emitValue(resultType, .binary(op: .notEqual, left: bits, right: magnitude), &statements, i)
            let belowZero = try lowerer.emitValue(words.type, .binary(op: .subtract, left: sign, right: magnitude), &statements, i)
            let aboveZero = try lowerer.emitValue(words.type, .binary(op: .add, left: sign, right: magnitude), &statements, i)
            let orderedBits = try lowerer.emitValue(words.type, .select(condition: negative, accept: belowZero, reject: aboveZero), &statements, i)
            return (orderedBits, isNaN)
        }

        // Integer operations keep the comparison defined even on targets that assume finite floating arithmetic.
        let lhs = try key(left.expression, lowerer: &self, statements: &statements)
        let rhs = try key(right, lowerer: &self, statements: &statements)
        let unordered = try emitValue(resultType, .binary(op: .inclusiveOr, left: lhs.isNaN, right: rhs.isNaN), &statements, i)
        let compare = try emitValue(resultType, .binary(op: operation, left: lhs.orderedBits, right: rhs.orderedBits), &statements, i)
        let comparison: IRExpressionKind
        if i.op.isMultiple(of: 2) {
            let ordered = try emitValue(resultType, .unary(op: .logicalNot, expression: unordered), &statements, i)
            comparison = .binary(op: .and, left: ordered, right: compare)
        } else {
            comparison = .binary(op: .inclusiveOr, left: unordered, right: compare)
        }
        // Keep the IEEE comparison expansion out of callers' nested WGSL expressions.
        let combined = try emitValue(resultType, comparison, &statements, i)
        try result(i, .runtimeValue(combined), &statements)
    }

    mutating func floatWordShape(_ shape: IRBitcastShape) throws -> SPVFloatWordShape {
        let scalar = shape.scalar.width == 8 ? IRScalar.u64 : .u32
        let scalarType = try internLoweredType(.scalar(scalar))
        let type = try shape.size.map { try internLoweredType(.vector(size: $0, scalar: scalar)) } ?? scalarType
        return .init(source: shape, scalarType: scalarType, type: type)
    }

    mutating func floatWordConstant(
        _ bits: UInt64, _ words: SPVFloatWordShape, _ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> IRHandle<IRExpression> {
        let literal = try expr(words.scalarType, .literal(words.source.scalar.width == 8 ? .u64(bits) : .u32(UInt32(bits))), i)
        guard let size = words.source.size else { return literal }
        return try emitValue(words.type, .splat(size: size, value: literal), &statements, i)
    }

    mutating func floatOperandBits(
        _ operand: IRHandle<IRExpression>, _ words: SPVFloatWordShape,
        _ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> IRHandle<IRExpression> {
        let shape = words.source
        guard shape.scalar == .f16 else {
            return try emitValue(words.type, .bitcast(expression: operand, type: words.type), &statements, i)
        }
        // Pack a pair of identical half lanes to read the original half bits without a floating conversion.
        let halfType = try internLoweredType(.scalar(.f16))
        let pairType = try internLoweredType(.vector(size: .two, scalar: .f16))
        let lowHalfMask = try expr(words.scalarType, .literal(.u32(0xffff)), i)
        var lanes: [IRHandle<IRExpression>] = []
        for lane in 0..<UInt32(shape.size?.rawValue ?? 1) {
            let half = shape.size == nil ? operand : try emitValue(halfType, .accessIndex(base: operand, index: lane), &statements, i)
            let pair = try emitValue(pairType, .splat(size: .two, value: half), &statements, i)
            let packed = try emitValue(words.scalarType, .bitcast(expression: pair, type: words.scalarType), &statements, i)
            lanes.append(try emitValue(words.scalarType, .binary(op: .and, left: packed, right: lowHalfMask), &statements, i))
        }
        if shape.size == nil { return lanes[0] }
        return try emitValue(words.type, .compose(type: words.type, components: lanes), &statements, i)
    }
}
