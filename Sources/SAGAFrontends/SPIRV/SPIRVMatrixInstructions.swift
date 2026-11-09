import SAGAShaderIR

extension SPVFunctionLowerer {
    mutating func lowerMatrixInstruction(_ i: SPVInstruction, _ statements: inout [IRHandle<IRStatement>]) throws {
        try i.count(i.op == 84 ? 3 : 4)
        let resultType = try module.type(i.a[0], i)
        let resultKind = module.types[resultType.index].kind
        let left = try value(i.a[2], i, &statements)
        let leftType = try module.type(left.type, i)
        let leftKind = module.types[leftType.index].kind
        if i.op == 84 {
            guard case .matrix(let columns, let rows, let scalar) = leftKind, scalar.kind == .float,
                  resultKind == .matrix(columns: rows, rows: columns, scalar: scalar) else {
                throw i.fail("OpTranspose requires a matrix and a result with swapped dimensions")
            }
            try result(i, .math(function: .transpose, argument: left.expression, argument1: nil, argument2: nil, argument3: nil),
                       &statements)
            return
        }

        let right = try value(i.a[3], i, &statements)
        let rightType = try module.type(right.type, i)
        let rightKind = module.types[rightType.index].kind
        let expected: IRTypeKind
        switch (i.op, leftKind, rightKind) {
        case (142, .vector(_, let scalar), .scalar(let other)),
             (143, .matrix(_, _, let scalar), .scalar(let other)):
            guard scalar.kind == .float, scalar == other else { throw i.fail("Floating-point scale operand type mismatch") }
            expected = leftKind
        case (144, .vector(let width, let scalar), .matrix(let columns, let rows, let other))
            where width == rows && scalar.kind == .float && scalar == other:
            expected = .vector(size: columns, scalar: scalar)
        case (145, .matrix(let columns, let rows, let scalar), .vector(let width, let other))
            where columns == width && scalar.kind == .float && scalar == other:
            expected = .vector(size: rows, scalar: scalar)
        case (146, .matrix(let inner, let rows, let scalar), .matrix(let columns, let rightRows, let other))
            where inner == rightRows && scalar.kind == .float && scalar == other:
            expected = .matrix(columns: columns, rows: rows, scalar: scalar)
        case (147, .vector(let rows, let scalar), .vector(let columns, let other))
            where scalar.kind == .float && scalar == other:
            expected = .matrix(columns: columns, rows: rows, scalar: scalar)
        case (148, .vector(let width, let scalar), .vector(let rightWidth, let other))
            where width == rightWidth && scalar.kind == .float && scalar == other:
            expected = .scalar(scalar)
        default: throw i.fail("Floating-point vector or matrix operand types do not match the opcode")
        }
        guard resultKind == expected else { throw i.fail("Floating-point vector or matrix result type mismatch") }

        if i.op == 148 {
            try result(i, .math(function: .dot, argument: left.expression, argument1: right.expression,
                                argument2: nil, argument3: nil), &statements)
        } else if i.op == 147, case .matrix(let columns, let rows, _) = expected {
            let scalarType = try indexedType(rightType, 0, i)
            var values: [IRHandle<IRExpression>] = []
            for column in 0..<UInt32(columns.rawValue) {
                let component = try emitValue(scalarType, .accessIndex(base: right.expression, index: column), &statements, i)
                let scale = try emitValue(leftType, .splat(size: rows, value: component), &statements, i)
                let value = try emitValue(leftType, .binary(op: .multiply, left: left.expression, right: scale), &statements, i)
                values.append(value)
            }
            try result(i, .compose(type: resultType, components: values), &statements)
        } else {
            let multiplier: IRHandle<IRExpression>
            if i.op == 142, case .vector(let size, _) = leftKind {
                multiplier = try emitValue(leftType, .splat(size: size, value: right.expression), &statements, i)
            } else {
                multiplier = right.expression
            }
            try result(i, .binary(op: .multiply, left: left.expression, right: multiplier), &statements)
        }
    }
}
