import SAGAShaderIR

extension GLSLParser {
    func negateMatrix(_ value: GLSLValue, at token: GLSLToken) throws -> GLSLValue? {
        guard case .matrix(let columns, let rows, let scalar) = kind(value.type) else { return nil }
        let columnType = try addType(.vector(size: rows, scalar: scalar))
        let components = try (0..<UInt32(columns.rawValue)).map { index in
            let column = try expressionNode(columnType, .accessIndex(base: value.expression, index: index), at: token)
            return try expressionNode(columnType, .unary(op: .negate, expression: column.expression), at: token).expression
        }
        return try expressionNode(value.type, .compose(type: value.type, components: components), at: token)
    }

    func glslMatrixBinary(_ left: GLSLValue, _ right: GLSLValue, at token: GLSLToken) throws -> GLSLValue? {
        func isMatrix(_ value: GLSLValue) -> Bool {
            if case .matrix = kind(value.type) { return true }
            return false
        }
        guard isMatrix(left) || isMatrix(right) else { return nil }
        let operation: IRBinaryOperator
        switch token.text {
        case "+": operation = .add
        case "-": operation = .subtract
        case "*": operation = .multiply
        case "/": operation = .divide
        case "==": operation = .equal
        case "!=": operation = .notEqual
        default: throw glslError(token, .unsupportedConstruct, "GLSL matrix operator not lowered")
        }
        let scalar: IRScalar = self.scalar(left.type) == .f64 || self.scalar(right.type) == .f64 ? .f64 : .f32
        let lhs = try matrixFloatOperand(left, scalar: scalar, at: token)
        let rhs = try matrixFloatOperand(right, scalar: scalar, at: token)
        let a = kind(lhs.type)
        let b = kind(rhs.type)
        let result: IRTypeKind
        switch (operation, a, b) {
        case (.add, .matrix, .matrix), (.subtract, .matrix, .matrix):
            guard a == b else { throw glslError(token, .typeMismatch, "Matrix arithmetic requires equal dimensions") }
            result = a
        case (.multiply, .matrix(let columns, let rows, _), .matrix(let width, let height, _)) where columns == height:
            result = .matrix(columns: width, rows: rows, scalar: scalar)
        case (.multiply, .matrix(let columns, let rows, _), .vector(let size, _)) where columns == size:
            result = .vector(size: rows, scalar: scalar)
        case (.multiply, .vector(let size, _), .matrix(let columns, let rows, _)) where size == rows:
            result = .vector(size: columns, scalar: scalar)
        case (.multiply, .matrix, .scalar): result = a
        case (.multiply, .scalar, .matrix): result = b
        case (.divide, .matrix, .matrix) where a == b:
            return try matrixComponentwise(operation, lhs, rhs, type: lhs.type, at: token)
        case (.equal, .matrix(let columns, let rows, _), .matrix),
             (.notEqual, .matrix(let columns, let rows, _), .matrix):
            guard a == b else { throw glslError(token, .typeMismatch, "Matrix comparison requires equal dimensions") }
            return try matrixComparison(operation, lhs, rhs, columns: columns, rows: rows, scalar: scalar, at: token)
        case (.add, .matrix, .scalar), (.subtract, .matrix, .scalar), (.divide, .matrix, .scalar):
            return try matrixComponentwise(operation, lhs, rhs, type: lhs.type, at: token)
        case (.add, .scalar, .matrix), (.subtract, .scalar, .matrix), (.divide, .scalar, .matrix):
            return try matrixComponentwise(operation, lhs, rhs, type: rhs.type, at: token)
        default:
            throw glslError(token, .typeMismatch, "Invalid GLSL matrix operand dimensions")
        }
        return try expressionNode(
            addType(result), .binary(op: operation, left: lhs.expression, right: rhs.expression), at: token)
    }

    private func matrixFloatOperand(_ value: GLSLValue, scalar: IRScalar, at token: GLSLToken) throws -> GLSLValue {
        let target: IRTypeKind
        switch kind(value.type) {
        case .scalar: target = .scalar(scalar)
        case .vector(let size, _): target = .vector(size: size, scalar: scalar)
        case .matrix(let columns, let rows, _): target = .matrix(columns: columns, rows: rows, scalar: scalar)
        default: throw glslError(token, .typeMismatch, "Matrix operands must be floating-point numeric values")
        }
        guard let converted = try implicitlyConvert(value, to: addType(target), at: token) else {
            throw glslError(token, .typeMismatch, "Matrix operand cannot be converted to float")
        }
        return converted
    }

    private func matrixComponentwise(
        _ operation: IRBinaryOperator, _ left: GLSLValue, _ right: GLSLValue,
        type: IRHandle<IRType>, at token: GLSLToken
    ) throws -> GLSLValue {
        guard case .matrix(let columns, let rows, let scalar) = kind(type) else {
            throw ShaderFailure.invalidHandle
        }
        let columnType = try addType(.vector(size: rows, scalar: scalar))
        func column(_ operand: GLSLValue, _ index: UInt32) throws -> IRHandle<IRExpression> {
            let expression: IRExpressionKind
            if case .scalar = kind(operand.type) {
                expression = .splat(size: rows, value: operand.expression)
            } else {
                expression = .accessIndex(base: operand.expression, index: index)
            }
            return try expressionNode(columnType, expression, at: token).expression
        }
        // Both operands were evaluated before the column arithmetic.
        var values: [IRHandle<IRExpression>] = []
        for index in 0..<UInt32(columns.rawValue) {
            let lhs = try column(left, index)
            let rhs = try column(right, index)
            let value = try expressionNode(columnType, .binary(op: operation, left: lhs, right: rhs), at: token)
            values.append(value.expression)
        }
        return try expressionNode(type, .compose(type: type, components: values), at: token)
    }

    private func matrixComparison(
        _ operation: IRBinaryOperator, _ left: GLSLValue, _ right: GLSLValue,
        columns: IRVectorSize, rows: IRVectorSize, scalar: IRScalar, at token: GLSLToken
    ) throws -> GLSLValue {
        let column = try addType(.vector(size: rows, scalar: scalar))
        let predicate = try addType(.vector(size: rows, scalar: .bool))
        let boolean = try addType(.scalar(.bool))
        if scalar == .f64 {
            let columns = try (0..<UInt32(columns.rawValue)).map { index in
                let lhs = try expressionNode(column, .accessIndex(base: left.expression, index: index), at: token)
                let rhs = try expressionNode(column, .accessIndex(base: right.expression, index: index), at: token)
                let equal = try expressionNode(predicate, .binary(op: .equal, left: lhs.expression, right: rhs.expression), at: token)
                return try expressionNode(boolean, .relational(function: .all, argument: equal.expression), at: token)
            }
            var equal = columns[0]
            for next in columns.dropFirst() {
                equal = try expressionNode(boolean, .binary(op: .logicalAnd, left: equal.expression, right: next.expression), at: token)
            }
            if operation == .notEqual {
                return try expressionNode(boolean, .unary(op: .logicalNot, expression: equal.expression), at: token)
            }
            return equal
        }
        let word = try addType(.scalar(.u32))
        let words = try addType(.vector(size: rows, scalar: .u32))
        // Materialize the bit operations so generated expressions do not repeat or nest entire columns.
        func materialize(_ value: GLSLValue) throws -> GLSLValue {
            try expressionNode(value.type, .runtimeValue(value.expression), at: token)
        }
        func splat(_ literal: UInt32) throws -> IRHandle<IRExpression> {
            let scalar = try expressionNode(word, .literal(.u32(literal)), at: token, emit: false)
            let vector = try expressionNode(words, .splat(size: rows, value: scalar.expression), at: token)
            return try materialize(vector).expression
        }
        func bits(_ operand: GLSLValue, _ index: UInt32) throws -> IRHandle<IRExpression> {
            let value = try expressionNode(column, .accessIndex(base: operand.expression, index: index), at: token)
            let bits = try expressionNode(words, .bitcast(expression: value.expression, type: words), at: token)
            return try materialize(bits).expression
        }
        func binary(
            _ op: IRBinaryOperator, _ lhs: IRHandle<IRExpression>, _ rhs: IRHandle<IRExpression>,
            type: IRHandle<IRType>
        ) throws -> IRHandle<IRExpression> {
            let value = try expressionNode(type, .binary(op: op, left: lhs, right: rhs), at: token)
            return try materialize(value).expression
        }
        let mask = try splat(0x7fffffff)
        let infinity = try splat(0x7f800000)
        let zero = try splat(0)
        var equalColumns: [GLSLValue] = []
        for index in 0..<UInt32(columns.rawValue) {
            let lhs = try bits(left, index)
            let rhs = try bits(right, index)
            let sameBits = try binary(.equal, lhs, rhs, type: predicate)
            let magnitude = try binary(.and, lhs, mask, type: words)
            let ordered = try binary(.lessEqual, magnitude, infinity, type: predicate)
            let combined = try binary(.inclusiveOr, lhs, rhs, type: words)
            let unsignedCombined = try binary(.and, combined, mask, type: words)
            let bothZero = try binary(.equal, unsignedCombined, zero, type: predicate)
            // Equal bit patterns must be ordered. Signed zeros compare equal with either sign.
            let sameOrderedBits = try binary(.and, sameBits, ordered, type: predicate)
            let sameValue = try binary(.inclusiveOr, sameOrderedBits, bothZero, type: predicate)
            let columnEqual = try expressionNode(boolean, .relational(function: .all, argument: sameValue), at: token)
            equalColumns.append(try materialize(columnEqual))
        }
        guard var equal = equalColumns.first else { throw ShaderFailure.invalidHandle }
        for next in equalColumns.dropFirst() {
            equal = try expressionNode(
                boolean, .binary(op: .logicalAnd, left: equal.expression, right: next.expression), at: token)
        }
        if operation == .notEqual {
            let different = try expressionNode(boolean, .unary(op: .logicalNot, expression: equal.expression), at: token)
            return try materialize(different)
        }
        return try materialize(equal)
    }
}
