import SAGAShaderIR

extension WGSLParser {
    func expressionOperator() -> WGSLToken {
        let first = token
        guard first.kind == .symbol, first.text == "<" || first.text == ">",
              cursor + 1 < tokens.count else { return first }
        let next = tokens[cursor + 1]
        guard next.kind == .symbol, first.end == next.start,
              next.text == first.text || next.text == first.text + "=" else { return first }
        return WGSLToken(kind: .symbol, text: first.text + next.text, start: first.start, end: next.end)
    }

    func integerShift(
        _ operation: IRBinaryOperator, left: WGSLValue, right: WGSLValue,
        leftTokens: Range<Int>, rightTokens: Range<Int>, at token: WGSLToken
    ) throws -> WGSLValue {
        if case .vector(let size, let scalar) = types[left.type.index].kind {
            guard [.i32, .u32, .i64, .u64].contains(scalar),
                  case .vector(let countSize, let count) = types[right.type.index].kind,
                  countSize == size, count == .u32 || count == .abstractInt else {
                throw wgslError(.typeMismatch, "Vector shift requires matching integer values and u32 counts", token)
            }
            if let amounts = try knownIntegerComponents(rightTokens, at: token) {
                guard amounts.count == Int(size.rawValue),
                      amounts.allSatisfy({ $0.value >= 0 && $0.value < Int64(scalar.width) * 8 }) else {
                    throw wgslError(.invalidType, "Constant vector shift count outside bit width", token)
                }
                if operation == .shiftLeft, let values = try knownIntegerComponents(leftTokens, at: token) {
                    guard values.count == amounts.count else {
                        throw wgslError(.typeMismatch, "Constant vector shape mismatch", token)
                    }
                    let evaluator = WGSLIntegerEvaluator(parser: self, tokens: [], fallback: token)
                    for (value, amount) in zip(values, amounts) {
                        _ = try evaluator.binary("<<", value, amount, at: token)
                    }
                }
            }
            let rhs = try coerce(right, to: intern(.vector(size: size, scalar: .u32)), at: token)
            return try expressionNode(.binary(op: operation, left: left.handle, right: rhs.handle),
                                      type: left.type, at: token)
        }
        guard case .scalar(let scalar) = types[left.type.index].kind,
              [.i32, .u32, .i64, .u64, .abstractInt].contains(scalar),
              case .scalar(let count) = types[right.type.index].kind, count == .u32 || count == .abstractInt else {
            throw wgslError(.typeMismatch, "Shift requires an integer value and u32 count", token)
        }
        if let amount = try knownIntegerConstant(rightTokens, at: token) {
            let width = Int64(scalar.width) * 8
            guard amount.value >= 0, amount.value < width else {
                throw wgslError(.invalidType, "Constant shift count outside bit width", token)
            }
            if let initial = try knownIntegerConstant(leftTokens, at: token) {
                let evaluator = WGSLIntegerEvaluator(parser: self, tokens: [], fallback: token)
                let symbol = operation == .shiftLeft ? "<<" : ">>"
                let result = try evaluator.binary(symbol, initial, amount, at: token)
                return try expressionNode(.literal(result.literal), type: left.type, at: token, emit: false)
            }
        }
        let lhs = try defaultMaterialize(left, at: token)
        let rhs = try coerce(right, to: intern(.scalar(.u32)), at: token)
        return try expressionNode(.binary(op: operation, left: lhs.handle, right: rhs.handle), type: lhs.type, at: token)
    }

    func integerVectorBinary(
        _ operation: IRBinaryOperator, left: WGSLValue, right: WGSLValue,
        leftTokens: Range<Int>, rightTokens: Range<Int>, at token: WGSLToken
    ) throws -> WGSLValue? {
        let lhs = types[left.type.index].kind
        let rhs = types[right.type.index].kind
        let size: IRVectorSize
        if case .vector(let width, _) = lhs { size = width }
        else if case .vector(let width, _) = rhs { size = width }
        else { return nil }
        func leaf(_ type: IRTypeKind) -> IRScalar? {
            switch type {
            case .scalar(let scalar), .vector(_, let scalar): return scalar
            default: return nil
            }
        }
        guard let l = leaf(lhs), let r = leaf(rhs),
              [IRScalar.i32, .u32, .i64, .u64, .abstractInt].contains(l),
              [IRScalar.i32, .u32, .i64, .u64, .abstractInt].contains(r) else { return nil }
        let scalar = l == .abstractInt ? r : l
        guard scalar != .abstractInt, l == r || l == .abstractInt || r == .abstractInt else {
            throw wgslError(.typeMismatch, "Integer vector operands need a common concrete lane type", token)
        }
        let type = try intern(.vector(size: size, scalar: scalar))
        let broadcast: Bool
        let compare: Bool
        switch operation {
        case .add, .subtract, .multiply, .divide, .modulo: (broadcast, compare) = (true, false)
        case .equal, .notEqual, .less, .lessEqual, .greater, .greaterEqual: (broadcast, compare) = (false, true)
        case .and, .inclusiveOr, .exclusiveOr: (broadcast, compare) = (false, false)
        default: throw unsupported("integer vector binary overload", at: token)
        }
        let a = try vectorOperand(left, to: type, size: size, scalar: scalar, broadcast: broadcast, at: token)
        let b = try vectorOperand(right, to: type, size: size, scalar: scalar, broadcast: broadcast, at: token)
        if (operation == .divide || operation == .modulo),
           let divisors = try knownIntegerComponents(rightTokens, at: token) {
            guard divisors.allSatisfy({ $0.value != 0 }) else {
                throw wgslError(.invalidType, "Constant vector divisor is zero", token)
            }
            if scalar.kind == .sint, divisors.contains(where: { $0.value == -1 }),
               let dividends = try knownIntegerComponents(leftTokens, at: token) {
                let width = Int(size.rawValue)
                guard (dividends.count == 1 || dividends.count == width),
                      (divisors.count == 1 || divisors.count == width) else {
                    throw wgslError(.typeMismatch, "Constant vector shape mismatch", token)
                }
                let minimum = scalar == .i64 ? Int64.min : Int64(Int32.min)
                for lane in 0..<width {
                    let lhs = dividends[dividends.count == 1 ? 0 : lane].value
                    let rhs = divisors[divisors.count == 1 ? 0 : lane].value
                    guard lhs != minimum || rhs != -1 else {
                        throw wgslError(.invalidType, "Constant vector division or remainder overflow", token)
                    }
                }
            }
        }
        let result = compare ? try intern(.vector(size: size, scalar: .bool)) : type
        return try expressionNode(.binary(op: operation, left: a.handle, right: b.handle), type: result, at: token)
    }

    func vectorOperand(
        _ operand: WGSLValue, to type: WType, size: IRVectorSize, scalar: IRScalar,
        broadcast: Bool, at token: WGSLToken
    ) throws -> WGSLValue {
        if case .scalar = types[operand.type.index].kind, broadcast {
            let value = try coerce(operand, to: intern(.scalar(scalar)), at: token)
            return try expressionNode(.splat(size: size, value: value.handle), type: type, at: token)
        }
        return try coerce(operand, to: type, at: token)
    }

    // WGSL let-bindings are runtime expressions even if their initializer is a literal.
    func knownIntegerConstant(_ range: Range<Int>, at token: WGSLToken) throws -> WGSLIntegerConstant? {
        guard let values = try knownIntegerComponents(range, at: token), values.count == 1 else { return nil }
        return values[0]
    }

    private func knownIntegerComponents(_ range: Range<Int>, at token: WGSLToken) throws -> [WGSLIntegerConstant]? {
        switch try knownConstantValue(range, at: token) {
        case .integer(let value): return [value]
        case .vector(let vector):
            var values: [WGSLIntegerConstant] = []
            for component in vector.elements {
                guard case .integer(let value) = component else { return nil }
                values.append(value)
            }
            return values
        default: return nil
        }
    }
}
