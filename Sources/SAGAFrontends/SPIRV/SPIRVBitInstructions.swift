import SAGAShaderIR

extension SPVFunctionLowerer {
    mutating func lowerBitInstruction(_ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        let opcode = i.op
        try i.count(opcode == 201 ? 6 : opcode <= 203 ? 5 : 3)
        let base = try value(i.a[2], i, &statements)
        let baseType = try module.type(base.type, i)
        let resultType = try module.type(i.a[0], i)
        guard let source = IRBitcastShape(module.types[baseType.index].kind),
              let destination = IRBitcastShape(module.types[resultType.index].kind),
              source.scalar.isSPVBitInteger, destination.scalar.isSPVBitInteger,
              source.size == destination.size, opcode == 205 || baseType == resultType else {
            throw i.fail("Bit instruction requires matching integer shapes and a compatible result type")
        }

        let signedness: IRScalarKind = opcode == 202 ? .sint : .uint
        let scalar = IRScalar(kind: signedness, width: max(4, source.scalar.width))
        let working = IRBitcastShape(size: source.size, scalar: scalar)
        let workingType = try internLoweredType(working.type)
        let operand = try prepareBitOperand(base.expression, source: source, working: working, i, &statements)
        var extra: [IRHandle<IRExpression>] = []
        if opcode == 201 {
            let insert = try typed(i.a[3], i.a[0], i, &statements)
            extra.append(try prepareBitOperand(insert, source: source, working: working, i, &statements))
        }
        if opcode <= 203 {
            let limit = UInt32(source.scalar.width) * 8
            let first = opcode == 201 ? 4 : 3
            var offset = try bitRangeOperand(i.a[first], limit: limit, i, &statements)
            var count = try bitRangeOperand(i.a[first + 1], limit: limit, i, &statements)
            if source.scalar.width < scalar.width {
                // The shared builtin clamps to its working width, so constrain the original narrower field first.
                let word = try internLoweredType(.scalar(.u32))
                let maxBits = try expr(word, .literal(.u32(limit)), i)
                offset = try clampBitRange(offset, to: maxBits, type: word, i, &statements)
                let available = try emitValue(word, .binary(op: .subtract, left: maxBits, right: offset), &statements, i)
                count = try clampBitRange(count, to: available, type: word, i, &statements)
            }
            extra += [offset, count]
        }
        let function: IRMathFunction = switch opcode {
        case 201: .insertBits
        case 202, 203: .extractBits
        case 204: .reverseBits
        default: .countOneBits
        }
        let math = IRExpressionKind.math(function: function, argument: operand, argument1: extra.first,
                                         argument2: extra.count >= 2 ? extra[1] : nil, argument3: extra.count == 3 ? extra[2] : nil)
        if workingType == resultType {
            try result(i, math, &statements)
            return
        }
        var converted = try emitValue(workingType, math, &statements, i)
        if opcode == 204, source.scalar.width == 2 {
            let word = try internLoweredType(.scalar(.u32))
            let bits = try expr(word, .literal(.u32(16)), i)
            let count = try working.size.map { try emitValue(workingType, .splat(size: $0, value: bits), &statements, i) } ?? bits
            converted = try emitValue(workingType, .binary(op: .shiftRight, left: converted, right: count), &statements, i)
        }
        let narrowed = IRScalar(kind: signedness, width: destination.scalar.width)
        if narrowed.width != scalar.width {
            let cast = IRExpressionKind.cast(expression: converted, target: narrowed)
            if narrowed == destination.scalar {
                try result(i, cast, &statements)
                return
            }
            let narrowType = try internLoweredType(IRBitcastShape(size: destination.size, scalar: narrowed).type)
            converted = try emitValue(narrowType, cast, &statements, i)
        }
        try result(i, narrowed == destination.scalar ? .runtimeValue(converted) : .bitcast(expression: converted, type: resultType),
                   &statements)
    }

    private mutating func prepareBitOperand(
        _ operand: IRHandle<IRExpression>, source: IRBitcastShape, working: IRBitcastShape,
        _ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> IRHandle<IRExpression> {
        var value = operand
        if source.scalar.kind != working.scalar.kind {
            let scalar = IRScalar(kind: working.scalar.kind, width: source.scalar.width)
            let type = try internLoweredType(IRBitcastShape(size: source.size, scalar: scalar).type)
            value = try emitValue(type, .bitcast(expression: value, type: type), &statements, i)
        }
        if source.scalar.width != working.scalar.width {
            let type = try internLoweredType(working.type)
            value = try emitValue(type, .cast(expression: value, target: working.scalar), &statements, i)
        }
        return value
    }

    private mutating func bitRangeOperand(
        _ id: UInt32, limit: UInt32, _ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> IRHandle<IRExpression> {
        let operand = try value(id, i, &statements)
        guard case .scalar(let scalar) = try module.kind(operand.type, i), scalar.isSPVBitInteger else {
            throw i.fail("Bit field offset and count must be scalar integers")
        }
        let working = IRBitcastShape(size: nil, scalar: scalar.width == 8 ? .u64 : .u32)
        let unsigned = try prepareBitOperand(operand.expression, source: .init(size: nil, scalar: scalar), working: working, i, &statements)
        guard scalar.width == 8 else { return unsigned }
        // Check the range before narrowing so a large offset cannot alias a valid low word.
        let wideType = try internLoweredType(.scalar(.u64))
        let maximum = try expr(wideType, .literal(.u64(UInt64(limit))), i)
        let bounded = try clampBitRange(unsigned, to: maximum, type: wideType, i, &statements)
        let wordType = try internLoweredType(.scalar(.u32))
        return try emitValue(wordType, .cast(expression: bounded, target: .u32), &statements, i)
    }

    private mutating func clampBitRange(
        _ value: IRHandle<IRExpression>, to maximum: IRHandle<IRExpression>, type: IRHandle<IRType>,
        _ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]
    ) throws -> IRHandle<IRExpression> {
        let boolean = try internLoweredType(.scalar(.bool))
        let exceeds = try emitValue(boolean, .binary(op: .greater, left: value, right: maximum), &statements, i)
        return try emitValue(type, .select(condition: exceeds, accept: maximum, reject: value), &statements, i)
    }
}

private extension IRScalar {
    var isSPVBitInteger: Bool { (kind == .uint || kind == .sint) && [2, 4, 8].contains(width) }
}
