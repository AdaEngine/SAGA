import SAGAShaderIR

private struct ShiftGroup {
    var hasShift = false
    var hasBitwise = false
}

extension WGSLParser {
    func literal(_ t: WGSLToken, negativeInt64: Bool = false) throws -> (IRLiteral,IRScalar) {
        if t.text == "true" || t.text == "false" { return (.bool(t.text == "true"),.bool) }
        guard t.kind == .number else { throw wgslError(.unexpectedToken,"Expected numeric/bool literal",t) }
        try budget.consume(.work,amount:t.text.utf8.count)
        var text = t.text
        let isHex = text.hasPrefix("0x") || text.hasPrefix("0X")
        let floating = text.contains(".") || (isHex ? text.lowercased().contains("p") : text.lowercased().contains("e"))
        let explicitHalf = text.hasSuffix("h")
        if explicitHalf {
            try requireFloat16(at: t)
            guard !isHex || text.lowercased().contains("p") else {
                throw wgslError(.invalidToken, "Hex f16 literal requires a binary exponent", t)
            }
        }
        // In hexadecimal integers 'f' is a digit, not a suffix.
        let explicitFloat = text.hasSuffix("f") && (!isHex || text.lowercased().contains("p"))
        if floating || explicitFloat || explicitHalf {
            if explicitFloat || explicitHalf { text.removeLast() }
            if isHex { return try hexFloat(text, kind: explicitHalf ? .f16 : explicitFloat ? .f32 : .abstract, at: t) }
            guard !text.hasSuffix("u"), !text.hasSuffix("i"), let d = Double(text), d.isFinite else { throw wgslError(.invalidToken,"Invalid float literal",t) }
            if explicitFloat {
                guard let f = Float(text), f.isFinite else { throw wgslError(.invalidToken,"f32 literal overflow",t) }; return (.f32(f),.f32)
            }
            if explicitHalf {
                let bits = Binary16.encode(d)
                guard bits & 0x7c00 != 0x7c00 else { throw wgslError(.invalidToken, "f16 literal overflow", t) }
                return (.f16Bits(bits), .f16)
            }
            return (.abstractFloat(d),.abstractFloat)
        }
        let suffix = text.last
        let explicitU64 = text.hasSuffix("lu")
        let explicitI64 = text.hasSuffix("li")
        let explicitU = suffix == "u", explicitI = suffix == "i"
        if explicitU64 || explicitI64 { text.removeLast(2) }
        else if explicitU || explicitI { text.removeLast() }
        let digits = isHex ? String(text.dropFirst(2)) : text
        if !isHex && digits.count > 1 && digits.hasPrefix("0") { throw wgslError(.invalidToken,"Leading zero integer",t) }
        guard let number = UInt64(digits,radix:isHex ? 16 : 10) else { throw wgslError(.invalidToken,"Integer literal overflow or missing digits",t) }
        if explicitU64 { return (.u64(number), .u64) }
        if explicitI64 {
            if negativeInt64, number == UInt64(Int64.max) + 1 { return (.i64(.min), .i64) }
            guard let value = Int64(exactly: number) else { throw wgslError(.invalidToken, "i64 literal overflow", t) }
            return (.i64(negativeInt64 ? -value : value), .i64)
        }
        if explicitU { guard let n = UInt32(exactly:number) else { throw wgslError(.invalidToken,"u32 literal overflow",t) }; return (.u32(n),.u32) }
        if explicitI { guard let n = Int32(exactly:number) else { throw wgslError(.invalidToken,"i32 literal overflow",t) }; return (.i32(n),.i32) }
        guard let n = Int64(exactly:number) else { throw wgslError(.invalidToken,"Abstract integer overflow",t) }
        return (.abstractInt(n),.abstractInt)
    }
    func convertLiteral(_ literal: IRLiteral, from scalar: IRScalar, to target: WType, at t: WGSLToken) throws -> IRLiteral {
        guard case .scalar(let s) = types[target.index].kind else { throw wgslError(.typeMismatch,"Scalar literal incompatible with type",t) }
        if s == scalar { return literal }
        switch literal {
        case .abstractInt(let n):
            if s == .i16 { return .i16(Int16(truncatingIfNeeded: n)) }
            if s == .u16 { return .u16(UInt16(truncatingIfNeeded: n)) }
            if s == .i64 { return .i64(n) }
            if s == .u64, let v = UInt64(exactly: n) { return .u64(v) }
            if s == .u32, let v = UInt32(exactly:n) { return .u32(v) }
            if s == .i32, let v = Int32(exactly:n) { return .i32(v) }
            if s == .f32 { let v = Float(n); if v.isFinite { return .f32(v) } }
            if s == .f16,
               let converted = WGSLFloatingConstant.rounded(Double(n), to: .f16, nagaFloatRounding: options.nagaFloatRounding) {
                return converted.literal
            }
            if s == .abstractFloat { return .abstractFloat(Double(n)) }
        case .abstractFloat(let n):
            if s == .f32 || s == .f16,
               let converted = WGSLFloatingConstant.rounded(n, to: s == .f32 ? .f32 : .f16,
                                                            nagaFloatRounding: options.nagaFloatRounding) {
                return converted.literal
            }
        default: break
        }
        throw wgslError(.typeMismatch,"Literal cannot be materialized as requested type",t)
    }
    func statement(_ kind: IRStatementKind, at t: WGSLToken? = nil) throws {
        if !allowsPointerSubviews, case .call(_, let arguments, _) = kind {
            allowsPointerSubviews = try arguments.contains { argument in
                try budget.consume(.work)
                let expression = body.expressions[argument.index]
                guard case .pointer = types[expression.type.index].kind else { return false }
                switch expression.kind {
                case .access, .accessIndex: return true
                default: return false
                }
            }
        }
        try node(); let h = try IRHandle<IRStatement>(index:body.statements.count)
        body.statements.append(IRStatement(kind:kind,location:t?.location)); body.current.append(h)
    }
    func expressionNode(_ kind: IRExpressionKind, type: WType, at t: WGSLToken, emit: Bool = true) throws -> WGSLValue {
        try node(); let h = try WExpr(index:body.expressions.count)
        body.expressions.append(IRExpression(type:type,kind:kind,location:t.location))
        body.pipelineStages.append(kind.pipelineEvaluationStage(type: types[type.index].kind) {
            body.pipelineStages[$0.index]
        })
        if emit { try statement(.emit([h]),at:t) }
        return WGSLValue(handle:h,type:type,pointer:false,writable:false)
    }
    func loaded(_ value: WGSLValue, at t: WGSLToken) throws -> WGSLValue {
        if let view = value.swizzle { return try readSwizzleView(value, view: view, at: t) }
        if case .bindingArray = types[value.type.index].kind {
            throw wgslError(.typeMismatch, "A binding array can only be indexed", t)
        }
        if !value.pointer { return value }
        if case .atomic = types[value.type.index].kind { throw unsupported("atomic read without atomicLoad",at:t) }
        return try expressionNode(.load(pointer:value.handle),type:value.type,at:t)
    }
    func coerce(_ value: WGSLValue, to target: WType, at t: WGSLToken) throws -> WGSLValue {
        let v = try loaded(value,at:t)
        if v.type == target { return v }
        let kind = types[v.type.index].kind
        if case .array(_, .constant(let size), _) = kind,
           case .array(let element, .constant(let targetSize), _) = types[target.index].kind,
           size == targetSize, case .compose(_, let components) = body.expressions[v.handle.index].kind {
            try enter()
            defer { nesting -= 1 }
            let source = try constantKind(v.type, at: t)
            guard source.containsAbstract else { throw wgslError(.typeMismatch, "Incompatible array type", t) }
            let converted = try components.map { handle in
                try budget.consume(.work)
                let value = WGSLValue(handle: handle, type: body.expressions[handle.index].type,
                                      pointer: false, writable: false)
                return try coerce(value, to: element, at: t).handle
            }
            return try expressionNode(.compose(type: target, components: converted), type: target, at: t)
        }
        if case .matrix(let columns,let rows,let scalar)=types[target.index].kind,
           case .matrix(let c,let r,let source)=kind,columns == c,rows == r,source.kind == .abstractFloat,
           case .compose(_,let xs)=body.expressions[v.handle.index].kind {
          let columnType=try intern(.vector(size:rows,scalar:scalar))
          var converted:[WExpr]=[]
          for h in xs {
            try budget.consume(.work)
            let e=body.expressions[h.index]
            converted.append(try coerce(WGSLValue(handle:h,type:e.type,pointer:false,writable:false),to:columnType,at:t).handle)
          }
          return try expressionNode(.compose(type:target,components:converted),type:target,at:t)
        }
        if case .vector(let n,let scalar)=types[target.index].kind,case .vector(let width,let source)=kind,n == width,
           source.kind == .abstractInt || source.kind == .abstractFloat,
           let literals=try abstractVectorLeaves(v.handle,at:t) {
            var converted:[WExpr]=[]
            let element=try intern(.scalar(scalar))
            for literal in literals {try budget.consume(.work);converted.append(try expressionNode(.literal(convertLiteral(literal,from:source,to:element,at:t)),type:element,at:t,emit:false).handle)}
            return try expressionNode(.compose(type:target,components:converted),type:target,at:t)
        }
        if case .vector(let n,let scalar)=types[target.index].kind,case .vector(let count,let source)=kind,n == count,
           source.kind == .abstractInt || source.kind == .abstractFloat,
           case .compose(_,let xs)=body.expressions[v.handle.index].kind {
            var converted:[WExpr]=[]
            for h in xs {
                try budget.consume(.work)
                let e=body.expressions[h.index]
                converted.append(try coerce(WGSLValue(handle:h,type:e.type,pointer:false,writable:false),to:intern(.scalar(scalar)),at:t).handle)
            }
            return try expressionNode(.compose(type:target,components:converted),type:target,at:t)
        }
        if case .scalar(let scalar) = kind, scalar == .abstractInt || scalar == .abstractFloat {
            var expressionKind = body.expressions[v.handle.index].kind
            if case .constant(let h) = expressionKind { expressionKind = globalExpressions[constants[h.index].initializer.index].kind }
            if let folded=try abstractScalarLeaf(v.handle,at:t) {expressionKind = .literal(folded)}
            if case .literal(let literal) = expressionKind {
                return try expressionNode(.literal(convertLiteral(literal,from:scalar,to:target,at:t)),type:target,at:t,emit:false)
            }
            throw unsupported("materialization of nonliteral abstract expression",at:t)
        }
        throw wgslError(.typeMismatch,"Incompatible expression type",t)
    }
    func defaultMaterialize(_ value: WGSLValue, at t: WGSLToken) throws -> WGSLValue {
        let v = try loaded(value,at:t)
        switch types[v.type.index].kind {
        case .matrix(let c,let r,.abstractFloat):return try coerce(v,to:intern(.matrix(columns:c,rows:r,scalar:.f32)),at:t)
        case .vector(let n,let scalar) where scalar.kind == .abstractInt || scalar.kind == .abstractFloat:
            return try coerce(v,to:intern(.vector(size:n,scalar:scalar.kind == .abstractInt ? .i32:.f32)),at:t)
        case .scalar(.abstractInt): return try coerce(v,to:intern(.scalar(.i32)),at:t)
        case .scalar(.abstractFloat): return try coerce(v,to:intern(.scalar(.f32)),at:t)
        case .array(_, .constant, _):
            let kind = try constantKind(v.type, at: t)
            return try kind.containsAbstract ? coerce(v, to: constantType(kind.concretized, at: t), at: t) : v
        case .structure: return try materializeDecomposition(v, at: t) ?? v
        default: return v
        }
    }
    func lookup(_ name: WGSLToken) throws -> WGSLValue {
        try budget.consume(.work,amount:name.text.utf8.count)
        let key = wgslKey(name.text)
        for scope in body.scopes.indices.reversed() {
            try budget.consume(.work)
            if let v = body.scopes[scope][key] {
                try requireVisibleToContinuing(key, scope: scope, at: name)
                return v
            }
        }
        try resolveValueDeclaration(name)
        if let handle = overrideNames[key] {
            return try expressionNode(.override(handle), type: overrides[handle.index].type, at: name, emit: false)
        }
        if let composite = compositeConstants[key] { return try compositeRuntimeValue(composite, at: name) }
        if let parts = decompositionConstants[key] { return try decompositionRuntimeValue(parts, at: name) }
        if let matrix=matrixConstants[wgslKey(name.text)] {return try matrixRuntimeValue(matrix,to:nil,at:name)}
        if let vector=vectorConstants[wgslKey(name.text)] {return try vectorRuntimeValue(vector,to:nil,at:name)}
        if let value=abstractFloatConstants[wgslKey(name.text)] {
            return try expressionNode(.literal(.abstractFloat(value)),type:intern(.scalar(.abstractFloat)),at:name,emit:false)
        }
        if let v = integerConstants[wgslKey(name.text)], v.kind == .abstract {
            return try expressionNode(.literal(v.literal),type:intern(.scalar(.abstractInt)),at:name,emit:false)
        }
        if let i = constantNames[wgslKey(name.text)] {
            return try expressionNode(.constant(try IRHandle(index:i)),type:constants[i].type,at:name,emit:false)
        }
        if let i = globalNames[wgslKey(name.text)] {
            let g = globals[i]
            if g.space == .handle {
                return try expressionNode(.globalVariable(try IRHandle(index: i)), type: g.type, at: name, emit: false)
            }
            let pointerType = try intern(.pointer(base: g.type, space: g.space))
            let reference = try expressionNode(.globalVariable(try IRHandle(index:i)),type:pointerType,at:name,emit:false)
            let writable: Bool
            switch g.space { case .uniform, .immediate: writable = false; case .storage(let access): writable = access.contains(.store); default:writable = true }
            return WGSLValue(handle:reference.handle,type:g.type,pointer:true,writable:writable)
        }
        throw wgslError(.unresolvedName,"Unknown name '\(name.text)'",name)
    }
    func expression(_ minimum: Int = 0) throws -> WGSLValue {
        var group = ShiftGroup()
        return try expression(minimum, group: &group)
    }

    private func expression(_ minimum: Int, group: inout ShiftGroup) throws -> WGSLValue {
        try enter(); defer { nesting -= 1 }
        let leftStart = cursor
        var left = try prefix()
        var numericOperator: String?
        while cursor != pipelineExpressionEnd {
            try budget.consume(.work)
            let operatorStart = cursor
            let op = expressionOperator()
            guard op.kind == .symbol, let (precedence, binary) = Self.binary[op.text], precedence >= minimum else { break }
            let shift = binary == .shiftLeft || binary == .shiftRight
            let bitwise = binary == .and || binary == .exclusiveOr || binary == .inclusiveOr
            if (shift && group.hasBitwise) || (bitwise && group.hasShift) {
                throw wgslError(.unexpectedToken,"Shift and bitwise operators require parentheses",op)
            }
            group.hasShift = group.hasShift || shift
            group.hasBitwise = group.hasBitwise || bitwise
            if precedence >= 8 {
                if let previous = numericOperator, shift || previous == "<<" || previous == ">>" {
                    throw wgslError(.unexpectedToken,"Shift group requires parentheses",op)
                }
                numericOperator = op.text
            }
            if op.text == "&&" || op.text == "||" {
                _ = try take(); left = try shortCircuit(left, op:op, precedence:precedence); continue
            }
            _ = try take()
            if shift { _ = try take() }
            left = try loaded(left,at:op)
            let rightStart = cursor
            if shift {
                let operand = try loaded(prefix(), at: op)
                left = try integerShift(binary, left: left, right: operand,
                                        leftTokens: leftStart..<operatorStart, rightTokens: rightStart..<cursor, at: op)
                continue
            }
            var right = try loaded(expression(precedence+1, group: &group),at:op)
            if let matrix = try matrixBinary(binary, left:left, right:right, at:op) {
                left = matrix; continue
            }
            if let vector = try integerVectorBinary(binary, left: left, right: right,
                                                    leftTokens: leftStart..<operatorStart,
                                                    rightTokens: rightStart..<cursor, at: op) {
                left = vector
                continue
            }
            if let vector = try floatingVectorBinary(binary, left: left, right: right, constantTokens: leftStart..<cursor, at: op) {
                left = vector
                continue
            }
            if case .vector(_, .bool) = types[left.type.index].kind {
                guard left.type == right.type, [.and, .inclusiveOr, .equal, .notEqual].contains(binary) else {
                    throw wgslError(.typeMismatch, "Boolean vector operands require a matching shape and &, |, == or !=", op)
                }
                left = try expressionNode(.binary(op: binary, left: left.handle, right: right.handle), type: left.type, at: op)
                continue
            }
            if left.type != right.type {
                if isAbstract(left.type), types[right.type.index].kind != .scalar(.abstractInt) {
                    left = try coerce(left, to: right.type, at: op)
                } else if isAbstract(right.type) {
                    right = try coerce(right, to: left.type, at: op)
                }
            }
            guard left.type == right.type else { throw wgslError(.typeMismatch,"Binary operands must match",op) }
            let comparison = ["==","!=","<",">","<=",">="].contains(op.text)
            let operandKind = types[left.type.index].kind
            let s: IRScalar
            if case .scalar(let scalar) = operandKind { s = scalar }
            else { throw unsupported("vector/matrix binary overload",at:op) }
            if s == .bool {
                guard ["==","!=","&","|"].contains(op.text) else { throw wgslError(.typeMismatch,"Invalid bool operator",op) }
            } else if ["&","|","^"].contains(op.text) && (s.kind == .float || s.kind == .abstractFloat) { throw wgslError(.typeMismatch,"Integer operand required",op) }
            if (binary == .divide || binary == .modulo) && [.sint, .uint, .abstractInt].contains(s.kind),
               let divisor = try knownIntegerConstant(rightStart..<cursor, at: op) {
                guard divisor.value != 0 else { throw wgslError(.invalidType,"Constant division or remainder by zero",op) }
                let minimum = Int64.min >> (64 - Int(s.width) * 8)
                if divisor.value == -1, s.kind == .sint,
                   try knownIntegerConstant(leftStart..<operatorStart, at: op)?.value == minimum {
                    throw wgslError(.invalidType,"Constant signed division or remainder overflow",op)
                }
            }
            let result = comparison ? try intern(.scalar(.bool)) : left.type
            let folded: IRLiteral?
            if s == .f32 || s == .f16 {
                folded = try knownConstantValue(leftStart..<cursor, at: op)?.literal
            } else {
                folded = try fold(binary, left: left, right: right, at: op)
            }
            if let folded {
                left = try expressionNode(.literal(folded),type:result,at:op,emit:false)
            } else {
                left = try expressionNode(.binary(op:binary,left:left.handle,right:right.handle),type:result,at:op)
            }
        }
        return left
    }
    static let binary: [String:(Int,IRBinaryOperator)] = [
        "||":(1,.logicalOr),"&&":(2,.logicalAnd),"|":(3,.inclusiveOr),"^":(4,.exclusiveOr),"&":(5,.and),
        "==":(6,.equal),"!=":(6,.notEqual),"<":(7,.less),"<=":(7,.lessEqual),">":(7,.greater),">=":(7,.greaterEqual),
        "<<":(8,.shiftLeft),">>":(8,.shiftRight),
        "+":(9,.add),"-":(9,.subtract),"*":(10,.multiply),"/":(10,.divide),"%":(10,.modulo)]
    func isAbstract(_ t: WType) -> Bool {
        switch types[t.index].kind { case .scalar(.abstractInt),.scalar(.abstractFloat):true; default:false }
    }
    func prefix() throws -> WGSLValue {
        try enter(); defer { nesting -= 1 }
        let t = token
        var value: WGSLValue
        if t.text == "&" || t.text == "*" {
            _ = try take()
            let operand = try prefix()
            return t.text == "&" ? try addressOf(operand, at: t) : try dereference(operand, at: t)
        }
        if ["-","!","~"].contains(t.text) {
            _ = try take()
            if t.text == "-", token.kind == .number, token.text.hasSuffix("li") {
                let (literal, scalar) = try literal(take(), negativeInt64: true)
                return try expressionNode(.literal(literal), type: intern(.scalar(scalar)), at: t, emit: false)
            }
            let v = try loaded(prefix(),at:t)
            if case .vector(_, let scalar) = types[v.type.index].kind {
                if t.text == "!", scalar == .bool {
                    return try expressionNode(.unary(op: .logicalNot, expression: v.handle), type: v.type, at: t)
                }
                guard (t.text == "-" && [.i16, .i32, .i64, .f16, .f32].contains(scalar)) ||
                      (t.text == "~" && [.i16, .u16, .i32, .u32, .i64, .u64].contains(scalar)) else {
                    throw unsupported("vector unary overload",at:t)
                }
                return try expressionNode(.unary(op: t.text == "-" ? .negate : .bitwiseNot, expression: v.handle),
                                          type: v.type, at: t)
            }
            guard case .scalar(let s) = types[v.type.index].kind else { throw unsupported("vector unary overload",at:t) }
            if t.text == "!" {
                guard s == .bool else { throw wgslError(.typeMismatch,"Logical not needs bool",t) }
            } else if t.text == "~" {
                guard s.kind == .sint || s.kind == .uint || s.kind == .abstractInt else { throw wgslError(.typeMismatch,"Bitwise not needs integer",t) }
            } else if s.kind == .uint || s == .bool { throw wgslError(.typeMismatch,"Cannot negate unsigned/bool",t) }
            if case .literal(.abstractInt(let n)) = body.expressions[v.handle.index].kind, t.text == "-", n != Int64.min {
                value = try expressionNode(.literal(.abstractInt(-n)),type:v.type,at:t,emit:false)
            } else if case .literal(.abstractInt(let n)) = body.expressions[v.handle.index].kind, t.text == "~" {
                value = try expressionNode(.literal(.abstractInt(~n)),type:v.type,at:t,emit:false)
            } else if case .literal(.abstractFloat(let n)) = body.expressions[v.handle.index].kind, t.text == "-" {
                value = try expressionNode(.literal(.abstractFloat(-n)),type:v.type,at:t,emit:false)
            } else {
                value = try expressionNode(.unary(op:t.text == "-" ? .negate : t.text == "!" ? .logicalNot : .bitwiseNot,expression:v.handle),type:v.type,at:t)
            }
        } else if t.kind == .number || t.text == "true" || t.text == "false" {
            _ = try take(); let (literal,s) = try literal(t)
            value = try expressionNode(.literal(literal),type:intern(.scalar(s)),at:t,emit:false)
        } else if try eat("(") {
            value = try expression(); try expect(")")
        } else if t.kind == .word {
            if isTypeName(t.text), body.scopes.contains(where: { $0[wgslKey(t.text)] != nil }) {
                let name = try ident()
                guard !isNext("(") else { throw wgslError(.typeMismatch,"Local value shadows predeclared type",name) }
                value = try lookup(name)
            } else if t.text == "array", cursor + 1 < tokens.count, tokens[cursor + 1].text == "(" {
                _ = try take(); try expect("("); value = try inferredArray(t)
            } else if (["vec2", "vec3", "vec4"].contains(t.text) || WGSLScalarEvaluator.matrixNames[t.text] != nil),
               declarationNames[wgslKey(t.text)] == nil, cursor + 1 < tokens.count, tokens[cursor + 1].text == "(" {
                _ = try take(); try expect("("); value = try inferredComposite(t)
            } else if isTypeName(t.text) {
                value = try typedConstructor(at: t)
            } else {
                let start = cursor
                let n = try ident()
                if n.text == "bitcast", isNext("<"), declarationNames[wgslKey(n.text)] == nil,
                   !(try localConstantNameExists(n)) {
                    value = try bitcastExpression(at: n, start: start)
                } else if try eat("(") { value = try call(n) }
                else { value = try lookup(n) }
            }
        } else { throw wgslError(.unexpectedToken,"Expected expression",t) }
        while true {
            if isNext("[") || isNext("."), !value.pointer, case .pointer = types[value.type.index].kind {
                value = try dereference(value, at: t)
            }
            if try eat("[") {
                let indexAt = token
                let indexStart = cursor
                let index = try widenedSubscript(try defaultMaterialize(expression(), at: indexAt), at: indexAt)
                let indexTokens = indexStart..<cursor
                try expect("]")
                let indexType = types[index.type.index].kind
                guard indexType == .scalar(.u32) || indexType == .scalar(.i32) else { throw wgslError(.typeMismatch,"Index must be i32/u32",t) }
                if let view = value.swizzle {
                    value = try indexSwizzleView(value, view: view, index: index, tokens: indexTokens, at: indexAt)
                    continue
                }
                let isConstantIndex = body.pipelineStages[index.handle.index] == .constant
                var fixedOffset: UInt32?
                if let extent = try abstractIndexExtent(value, at: t) {
                    if isConstantIndex {
                        let offset = try constantSubscript(index, tokens: indexTokens, at: indexAt)
                        guard offset >= 0, offset < Int64(extent) else {
                            throw wgslError(.invalidType, "Constant index out of bounds", indexAt)
                        }
                        if case .array = types[value.type.index].kind,
                           case .compose(_, let elements) = body.expressions[value.handle.index].kind {
                            let handle = elements[Int(offset)]
                            value = WGSLValue(handle: handle, type: body.expressions[handle.index].type,
                                              pointer: false, writable: false)
                            continue
                        }
                        fixedOffset = UInt32(offset)
                    } else {
                        value = try defaultMaterialize(value, at: t)
                    }
                } else if isConstantIndex, case .vector(let width, _) = types[value.type.index].kind,
                          let literal = try abstractScalarLeaf(index.handle, at: indexAt) {
                    let offset: Int64
                    switch literal {
                    case .i32(let n): offset = Int64(n)
                    case .u32(let n): offset = Int64(n)
                    default: throw wgslError(.typeMismatch, "Index must be integer", indexAt)
                    }
                    guard offset >= 0, offset < Int64(width.rawValue) else {
                        throw wgslError(.invalidType, "Constant vector index out of bounds", indexAt)
                    }
                }
                let element: WType
                switch types[value.type.index].kind {
                case .array(let base,_,_), .bindingArray(let base,_):element = base
                case .vector(_,let scalar):element = try intern(.scalar(scalar))
                case .matrix(_,let rows,let scalar):element = try intern(.vector(size:rows,scalar:scalar))
                default:throw wgslError(.typeMismatch,"Value not indexable",t)
                }
                if value.pointer {
                    let space = try pointerSpace(value)
                    let p = try expressionNode(.access(base:value.handle,index:index.handle),type:intern(.pointer(base:element,space:space)),at:t)
                    value = WGSLValue(handle:p.handle,type:element,pointer:true,writable:value.writable)
                } else {
                    let access = fixedOffset.map { IRExpressionKind.accessIndex(base: value.handle, index: $0) }
                        ?? .access(base: value.handle, index: index.handle)
                    value = try expressionNode(access, type: element, at: t)
                }
            } else if try eat(".") {
                let field = try ident()
                switch types[value.type.index].kind {
                case .structure(let members,_):
                    var match: Int?
                    for (i,m) in members.enumerated() { try budget.consume(.work,amount:field.text.utf8.count+1); if m.name?.utf8.elementsEqual(field.text.utf8) == true { match = i; break } }
                    guard let i = match else { throw wgslError(.unresolvedName,"No such struct member",field) }
                    let ty = members[i].type
                    if !value.pointer, case .compose(_, let components) = body.expressions[value.handle.index].kind,
                       predeclaredTypes.contains(where: { $0.type == value.type && $0.kind.decomposition != nil }) {
                        value = WGSLValue(handle: components[i], type: ty, pointer: false, writable: false)
                        continue
                    }
                    let resultType = value.pointer ? try intern(.pointer(base:ty,space:pointerSpace(value))) : ty
                    let p = try expressionNode(.accessIndex(base:value.handle,index:UInt32(i)),type:resultType,at:field)
                    value = WGSLValue(handle:p.handle,type:ty,pointer:value.pointer,writable:value.writable)
                case .vector(let size,let scalar):
                    let chars = Array(field.text)
                    guard chars.count >= 1 && chars.count <= 4 else { throw wgslError(.typeMismatch,"Invalid swizzle length",field) }
                    let xy = Array("xyzw"), rgba = Array("rgba")
                    var indices: [Int] = []
                    let alphabet = xy.contains(chars[0]) ? xy : rgba
                    for c in chars {
                        guard let n = alphabet.firstIndex(of:c), n < Int(size.rawValue) else { throw wgslError(.typeMismatch,"Invalid swizzle",field) }; indices.append(n)
                    }
                    value = try vectorSelection(value, lanes: indices, scalar: scalar, at: field)
                default: throw wgslError(.typeMismatch,"Member access on noncomposite",field)
                }
            } else { break }
        }
        return value
    }

    private func abstractIndexExtent(_ value: WGSLValue, at token: WGSLToken) throws -> UInt32? {
        guard !value.pointer else { return nil }
        switch types[value.type.index].kind {
        case .vector(let width, .abstractInt), .vector(let width, .abstractFloat),
             .matrix(let width, _, .abstractFloat):
            return UInt32(width.rawValue)
        case .array(_, .constant(let count), _):
            return try constantKind(value.type, at: token).containsAbstract ? count : nil
        default:
            return nil
        }
    }

    private func constantSubscript(_ value: WGSLValue, tokens range: Range<Int>, at token: WGSLToken) throws -> Int64 {
        switch body.expressions[value.handle.index].kind {
        case .literal(.i32(let index)): return Int64(index)
        case .literal(.u32(let index)): return Int64(index)
        default:
            guard case .integer(let index) = try scalarConstant(Array(tokens[range]), at: token) else {
                throw wgslError(.typeMismatch, "Index must be integer", token)
            }
            return index.value
        }
    }
    private func widenedSubscript(_ index: WGSLValue, at token: WGSLToken) throws -> WGSLValue {
        let scalar: IRScalar
        switch types[index.type.index].kind {
        case .scalar(.i16): scalar = .i32
        case .scalar(.u16): scalar = .u32
        default: return index
        }
        let type = try intern(.scalar(scalar))
        switch body.expressions[index.handle.index].kind {
        case .literal(.i16(let value)):
            return try expressionNode(.literal(.i32(Int32(value))), type: type, at: token, emit: false)
        case .literal(.u16(let value)):
            return try expressionNode(.literal(.u32(UInt32(value))), type: type, at: token, emit: false)
        default:
            return try expressionNode(.cast(expression: index.handle, target: scalar), type: type, at: token)
        }
    }
    private func addressOf(_ value: WGSLValue, at t: WGSLToken) throws -> WGSLValue {
        guard value.pointer else { throw wgslError(.typeMismatch, "Address-of requires a reference", t) }
        if case .bindingArray = types[value.type.index].kind {
            throw wgslError(.invalidReference, "Cannot take the address of an entire binding array", t)
        }
        guard value.swizzle == nil else { throw wgslError(.invalidReference, "Cannot take the address of a swizzle view", t) }
        let expression = body.expressions[value.handle.index]
        let base: WExpr?
        switch expression.kind {
        case .access(let parent, _), .accessIndex(let parent, _): base = parent
        default: base = nil
        }
        if let base, case .pointer(let pointee, _) = types[body.expressions[base.index].type.index].kind,
           case .vector = types[pointee.index].kind {
            throw wgslError(.invalidReference, "Cannot take the address of a vector component", t)
        }
        // The IR stores both WGSL references and pointer values as typed addresses.
        return WGSLValue(handle: value.handle, type: expression.type, pointer: false, writable: false)
    }

    private func dereference(_ value: WGSLValue, at t: WGSLToken) throws -> WGSLValue {
        guard !value.pointer, case .pointer(let base, let space) = types[value.type.index].kind else {
            throw wgslError(.typeMismatch, "Indirection requires a pointer", t)
        }
        let writable: Bool
        switch space {
        case .storage(let access): writable = access.contains(.store)
        case .uniform: writable = false
        default: writable = true
        }
        return WGSLValue(handle: value.handle, type: base, pointer: true, writable: writable)
    }

    func pointerSpace(_ value: WGSLValue) throws -> IRAddressSpace {
        guard case .pointer(_,let space) = types[body.expressions[value.handle.index].type.index].kind else { throw wgslError(.invalidType,"Expected pointer",token) }; return space
    }
    func isTypeName(_ name: String) -> Bool {
        predeclaredType(name) != nil || typeNames[wgslKey(name)] != nil || declarationNames[wgslKey(name)].map { ["alias","struct"].contains(declarations[$0].kind) } == true || ["u16","u32","u64","i16","i32","i64","f32","f16","bool","vec2","vec3","vec4","array","mat2x2","mat2x3","mat2x4","mat3x2","mat3x3","mat3x4","mat4x2","mat4x3","mat4x4"].contains(name)
    }
    func arguments() throws -> [WGSLValue] {
        var args: [WGSLValue] = []
        while !isNext(")") {
            try node(); args.append(try loaded(expression(),at:token))
            if !(try eat(",")) { break }
        }
        try expect(")"); return args
    }
    private func typedConstructor(at name: WGSLToken) throws -> WGSLValue {
        let start = cursor
        let target = try type()
        try expect("(")
        let result = try construct(target, at: name)
        guard types[target.index].kind.usesFloat16,
              let constant = try knownConstantValue(start..<cursor, at: name) else { return result }
        if case .matrix(let matrix) = constant {
            return try matrixRuntimeValue(matrix, to: target, at: name)
        }
        if case .vector(let vector) = constant {
            return try vectorRuntimeValue(vector, to: target, at: name)
        }
        return try expressionNode(.literal(constant.literal), type: target, at: name, emit: false)
    }
    func construct(_ target: WType, at t: WGSLToken, supplied: [WGSLValue]? = nil) throws -> WGSLValue {
        var args = try supplied ?? arguments()
        if args.isEmpty {
            switch types[target.index].kind {
            case .scalar, .vector, .matrix, .array(_, .constant, _), .structure: break
            default:throw unsupported("zero composite constructor",at:t)
            }
            return try expressionNode(.zeroValue(target),type:target,at:t,emit:false)
        }
        switch types[target.index].kind {
        case .scalar(let scalar):
            guard args.count == 1 else { throw wgslError(.typeMismatch,"Scalar constructor arity",t) }
            if args[0].type == target { return args[0] }
            guard case .scalar(let source) = types[args[0].type.index].kind, scalar.isWGSLConversionScalar else {
                throw wgslError(.typeMismatch, "Scalar constructor requires a scalar", t)
            }
            if source.isWGSLConversionScalar {
                return try expressionNode(.cast(expression: args[0].handle, target: scalar), type: target, at: t)
            }
            guard let value = try pureCompositeConstant(args[0], at: t) else {
                throw wgslError(.typeMismatch, "Abstract scalar conversion requires a constant", t)
            }
            let evaluator = WGSLScalarEvaluator(parser: self, tokens: [], fallback: t)
            let result = try evaluator.scalarConvert(value, to: .scalar(scalar, at: t), at: t, explicit: true)
            return try expressionNode(.literal(result.literal), type: target, at: t, emit: false)
        case .vector(let size,let scalar):
            let elem = try intern(.scalar(scalar))
            if args.count == 1, case .vector(let sourceSize, let source) = types[args[0].type.index].kind,
               sourceSize == size, source != scalar, scalar.isWGSLConversionScalar {
                if source.isWGSLConversionScalar {
                    return try expressionNode(.cast(expression: args[0].handle, target: scalar), type: target, at: t)
                }
                guard case .vector(let vector)? = try pureCompositeConstant(args[0], at: t) else {
                    throw wgslError(.typeMismatch, "Abstract vector conversion requires a constant", t)
                }
                let leaf = try WGSLConstKind.scalar(scalar, at: t)
                let evaluator = WGSLScalarEvaluator(parser: self, tokens: [], fallback: t)
                let elements = try vector.elements.map { value in
                    try budget.consume(.work)
                    return try evaluator.scalarConvert(value, to: leaf, at: t, explicit: true)
                }
                return try vectorRuntimeValue(.init(elements: elements, element: leaf), to: target, at: t)
            }
            if args.count == 1, case .scalar = types[args[0].type.index].kind {
                let v = try coerce(args[0],to:elem,at:t)
                return try expressionNode(.splat(size:size,value:v.handle),type:target,at:t)
            }
            var components: [WGSLValue] = []
            for arg in args {
                try budget.consume(.work)
                switch types[arg.type.index].kind {
                case .scalar:
                    components.append(try coerce(arg,to:elem,at:t))
                case .vector(let n,_):
                    let vector=try coerce(arg,to:intern(.vector(size:n,scalar:scalar)),at:t)
                    for i in 0..<Int(n.rawValue) {
                        try budget.consume(.work)
                        components.append(try expressionNode(.accessIndex(base:vector.handle,index:UInt32(i)),type:elem,at:t))
                    }
                default:throw wgslError(.typeMismatch,"Vector components must be scalar/vector",t)
                }
                guard components.count <= Int(size.rawValue) else {throw wgslError(.typeMismatch,"Too many vector components",t)}
            }
            guard components.count == Int(size.rawValue) else {throw wgslError(.typeMismatch,"Too few vector components",t)}
            args=components
        case .matrix(let columns,let rows,let scalar):
            if args.count == 1, case .matrix(let c, let r, let input) = types[args[0].type.index].kind {
                guard c == columns, r == rows else { throw wgslError(.typeMismatch, "Matrix conversion shape mismatch", t) }
                if input != scalar, [.f16, .f32].contains(input), [.f16, .f32].contains(scalar) {
                    return try expressionNode(.cast(expression: args[0].handle, target: scalar), type: target, at: t)
                }
                return try coerce(args[0], to: target, at: t)
            }
            let column=try intern(.vector(size:rows,scalar:scalar))
            if args.count == Int(columns.rawValue),args.allSatisfy({if case .vector=types[$0.type.index].kind {true} else {false}}) {
                for i in args.indices {args[i]=try coerce(args[i],to:column,at:t)}
            } else {
                guard args.count == Int(columns.rawValue)*Int(rows.rawValue) else {throw wgslError(.typeMismatch,"Matrix constructor component count",t)}
                let element=try intern(.scalar(scalar));var converted:[WGSLValue]=[]
                for c in 0..<Int(columns.rawValue) {
                    var values:[WExpr]=[]
                    for r in 0..<Int(rows.rawValue) {try budget.consume(.work);values.append(try coerce(args[c*Int(rows.rawValue)+r],to:element,at:t).handle)}
                    converted.append(try expressionNode(.compose(type:column,components:values),type:column,at:t))
                }
                args=converted
            }
        case .structure(let members,_):
            guard members.count == args.count else { throw wgslError(.typeMismatch,"Struct constructor arity",t) }
            for i in args.indices { args[i] = try coerce(args[i],to:members[i].type,at:t) }
        case .array(let elem,.constant(let n),_):
            guard UInt64(args.count) == UInt64(n) else { throw wgslError(.typeMismatch,"Array constructor arity",t) }
            for i in args.indices { args[i] = try coerce(args[i],to:elem,at:t) }
        default: throw unsupported("constructor overload",at:t)
        }
        return try expressionNode(.compose(type:target,components:args.map(\.handle)),type:target,at:t)
    }
    func call(_ name: WGSLToken) throws -> WGSLValue {
        try resolveFunctionDeclaration(name)
        if declarationNames[wgslKey(name.text)] == nil, Self.subgroupGatherBuiltins.contains(name.text) {
            return try subgroupGather(at: name)
        }
        if declarationNames[wgslKey(name.text)] == nil {
            if name.text == "subgroupBallot" { return try subgroupBallot(at: name) }
            if name.text == "subgroupElect" { return try subgroupElect(at: name) }
            if let (operation, collective) = IRSubgroupOperation.wgslBuiltins[name.text] {
                return try subgroupCollective(operation, collective: collective, at: name)
            }
        }
        if name.text == "workgroupUniformLoad", declarationNames[wgslKey(name.text)] == nil {
            let pointer = try dereference(expression(), at: name)
            guard try pointerSpace(pointer) == .workgroup else {
                throw wgslError(.typeMismatch, "workgroupUniformLoad requires the address of a workgroup value", name)
            }
            let type: WType
            if case .atomic(let scalar) = types[pointer.type.index].kind { type = try intern(.scalar(scalar)) }
            else { type = pointer.type }
            _ = try eat(",")
            try expect(")")
            let result = try expressionNode(.workgroupUniformLoadResult(type: type), type: type, at: name, emit: false)
            try statement(.workgroupUniformLoad(pointer: pointer.handle, result: result.handle), at: name)
            return result
        }
        let atomic = IRAtomicFunction.integerRMWBuiltins[name.text]
        let compares = name.text == "atomicCompareExchangeWeak"
        if declarationNames[wgslKey(name.text)] == nil, name.text == "atomicLoad" || atomic != nil || compares {
            let (pointer, type) = try atomicPointer(at: name)
            if name.text == "atomicLoad" {
                _ = try eat(",")
                try expect(")")
                return try expressionNode(.load(pointer: pointer.handle), type: type, at: name)
            }
            try expect(",")
            let first = try coerce(expression(), to: type, at: name)
            let value: WGSLValue
            let operation: IRAtomicFunction
            let resultType: WType
            if compares {
                guard types[type.index].kind != .scalar(.f32) else {
                    throw wgslError(.typeMismatch, "Float atomics support add, subtract and exchange", name)
                }
                try expect(",")
                value = try coerce(expression(), to: type, at: name)
                operation = .exchange(compare: first.handle)
                resultType = try atomicCompareResult(for: type, at: name)
            } else if let atomic {
                value = first
                operation = atomic
                resultType = type
            } else { throw ShaderFailure.invalidHandle }
            _ = try eat(",")
            try expect(")")
            let result = try expressionNode(.atomicResult(type: resultType, comparison: compares),
                                            type: resultType, at: name, emit: false)
            try statement(.atomic(pointer: pointer.handle, function: operation, value: value.handle, result: result.handle), at: name)
            return result
        }
        if name.text == "arrayLength", functionNames[wgslKey(name.text)] == nil {
            let array = try dereference(expression(), at: name)
            guard case .storage = try pointerSpace(array),
                  case .array(_, .dynamic, _) = types[array.type.index].kind else {
                throw wgslError(.typeMismatch, "arrayLength requires the address of a runtime storage array", name)
            }
            _ = try eat(",")
            try expect(")")
            return try expressionNode(.arrayLength(array.handle), type: intern(.scalar(.u32)), at: name)
        }
        if ["textureSample", "textureSampleLevel", "textureSampleBias", "textureSampleGrad", "textureGather",
            "textureSampleCompare", "textureSampleCompareLevel", "textureGatherCompare",
            "textureSampleBaseClampToEdge"].contains(name.text),
           declarationNames[wgslKey(name.text)] == nil {
            return try textureSampleBuiltin(at: name)
        }
        if declarationNames[wgslKey(name.text)] == nil,
           let function = IRMathFunction(wgslName: name.text) {
            if let arity = function.integerBitArity { return try integerBitBuiltin(function, arity: arity, at: name) }
            if let packing = function.packingBuiltin { return try packingCall(packing, function: function, at: name) }
            if function.geometricArity != nil { return try geometricBuiltin(function, at: name) }
            if function.decomposesFloat { return try decompositionBuiltin(function, at: name) }
            if function == .ldexp { return try ldexpBuiltin(at: name) }
            if let numeric = function.numericBuiltin { return try numericBuiltin(function, builtin: numeric, at: name) }
        }
        var args = try arguments()
        if let i = functionNames[wgslKey(name.text)] {
            guard !entries.contains(where: { $0.function.index == i }) else { throw wgslError(.invalidControlFlow,"Entry point cannot be called",name) }
            let fn = functions[i]
            guard args.count == fn.arguments.count else { throw wgslError(.typeMismatch,"Function argument count",name) }
            for j in args.indices { args[j] = try coerce(args[j],to:fn.arguments[j].type,at:name) }
            guard let result = fn.result else { throw unsupported("void function used as expression",at:name) }
            let v = try expressionNode(.callResult(try IRHandle(index:i)),type:result.type,at:name,emit:false)
            try statement(.call(function:try IRHandle(index:i),arguments:args.map(\.handle),result:v.handle),at:name); return v
        }
        if declarationNames[wgslKey(name.text)] == nil,
           let reduction = IRRelationalFunction(rawValue: name.text), reduction.isBooleanReduction {
            guard !(try localConstantNameExists(name)), args.count == 1, let operand = args.first,
                  reduction.accepts(types[operand.type.index].kind) else {
                throw wgslError(.typeMismatch, "Boolean reduction requires one bool scalar or vector", name)
            }
            return try expressionNode(.relational(function: reduction, argument: operand.handle),
                                      type: intern(.scalar(.bool)), at: name)
        }
        if name.text == "select", declarationNames[wgslKey(name.text)] == nil {
            return try selectValue(args, at: name)
        }
        if ["textureLoad", "textureDimensions", "textureNumLevels", "textureNumSamples", "textureNumLayers"].contains(name.text),
           declarationNames[wgslKey(name.text)] == nil {
            return try textureBuiltin(args, at: name)
        }
        if declarationNames[wgslKey(name.text)] == nil, let derivative = IRDerivativeAxis.wgslBuiltins[name.text] {
            guard !(try localConstantNameExists(name)), args.count == 1 else {
                throw wgslError(.typeMismatch, "Derivative builtin requires one operand", name)
            }
            let shape: IRTypeKind
            switch types[args[0].type.index].kind {
            case .scalar: shape = .scalar(.f32)
            case .vector(let size, _): shape = .vector(size: size, scalar: .f32)
            default: throw wgslError(.typeMismatch, "Derivative operand must be f32 or an f32 vector", name)
            }
            let type = try intern(shape)
            let value = try coerce(args[0], to: type, at: name)
            return try expressionNode(.derivative(axis: derivative.axis, control: derivative.control, expression: value.handle),
                                      type: type, at: name)
        }
        throw unsupported("builtin or forward function '\(name.text)'",at:name)
    }

    private func textureSampleBuiltin(at name: WGSLToken) throws -> WGSLValue {
        let clampToEdge = name.text == "textureSampleBaseClampToEdge"
        let arity: Int
        switch name.text {
        case "textureSample", "textureGather", "textureSampleBaseClampToEdge": arity = 3
        case "textureSampleGrad": arity = 5
        default: arity = 4
        }
        let depthGather = name.text == "textureGather" ? try gatherStartsWithDepthTexture() : false
        let gather: IRSwizzleComponent?
        if name.text == "textureGather" && !depthGather {
            let at = token
            guard case .integer(let index) = try textureConstantArgument() else {
                throw wgslError(.typeMismatch, "Texture gather component must be an integer constant", at)
            }
            guard (0...3).contains(index.value) else {
                throw wgslError(.invalidType, "Texture gather component must be between 0 and 3", at)
            }
            gather = IRSwizzleComponent.allCases[Int(index.value)]
            try expect(",")
        } else {
            gather = depthGather || name.text == "textureGatherCompare" ? .x : nil
        }
        try node()
        let image = try loaded(expression(), at: token)
        let imageType = types[image.type.index].kind
        let arrayed = imageType.isArrayedTexture
        let volume = imageType.isVolumeTexture
        let line = imageType.isLineTexture
        if line && name.text != "textureSample" && name.text != "textureSampleLevel" {
            throw wgslError(.typeMismatch, "Unsupported sampling operation on texture_1d", name)
        }
        if volume && gather != nil {
            throw wgslError(.typeMismatch, "Volume textures do not support texture gathers", name)
        }
        if imageType.isExternalTexture && !clampToEdge {
            throw wgslError(.typeMismatch, "External texture sampling requires textureSampleBaseClampToEdge", name)
        }
        if clampToEdge && imageType != .sampledFloat2D && !imageType.isExternalTexture {
            throw wgslError(.typeMismatch, "Base clamp sampling requires texture_2d<f32> or texture_external", name)
        }
        var args = [image]
        for _ in 1..<(arity + (arrayed ? 1 : 0)) {
            try expect(",")
            try node()
            args.append(try loaded(expression(), at: token))
        }
        let offset: WExpr?
        if try eat(","), !isNext(")") {
            guard !clampToEdge else {
                throw wgslError(.typeMismatch, "textureSampleBaseClampToEdge takes three arguments", token)
            }
            let nativeLineOffset = options.allow1DTextureSampleLevelOffset && name.text == "textureSampleLevel"
            guard !imageType.isCubeTexture && (!line || nativeLineOffset) else {
                throw wgslError(.typeMismatch, "Texture dimension does not support texel offsets", token)
            }
            offset = try textureSampleOffset(width: line ? nil : (volume ? .three : .two))
            _ = try eat(",")
        } else {
            offset = nil
        }
        try expect(")")
        let comparison = ["textureSampleCompare", "textureSampleCompareLevel", "textureGatherCompare"].contains(name.text)
        let depth = imageType.isDepthTexture
        let validDepth = comparison || depthGather || name.text == "textureSample" || name.text == "textureSampleLevel"
        let sampledScalar = imageType.isExternalTexture ? IRScalar.f32 : imageType.sampledTextureScalar
        guard let scalar = sampledScalar, !imageType.isMultisampled2D,
              depth ? validDepth : (!comparison && (scalar == .f32 || gather != nil)),
              types[args[1].type.index].kind == .sampler(comparison: comparison) else {
            throw wgslError(.typeMismatch, "Texture and sampler types do not match the sampling operation", name)
        }
        let arrayIndex: WExpr?
        if arrayed {
            arrayIndex = try textureArrayIndex(args[3], at: name)
        } else {
            arrayIndex = nil
        }
        let levelIndex = arrayed ? 4 : 3
        let reference = comparison ? try coerce(args[levelIndex], to: intern(.scalar(.f32)), at: name).handle : nil
        let coordinateType = imageType.isCubeTexture ? IRTypeKind.vector(size: .three, scalar: .f32) :
            imageType.texelCoordinateType(.f32)
        let coordinate = try intern(coordinateType)
        let uv = try coerce(args[2], to: coordinate, at: name)
        let level: IRSampleLevel
        switch name.text {
        case "textureSample", "textureSampleCompare": level = .auto
        case "textureGather", "textureSampleCompareLevel", "textureGatherCompare", "textureSampleBaseClampToEdge":
            level = .zero
        case "textureSampleLevel" where depth:
            let value = try defaultMaterialize(args[levelIndex], at: name)
            guard [.scalar(.i32), .scalar(.u32)].contains(types[value.type.index].kind) else {
                throw wgslError(.typeMismatch, "Depth sampling level must be i32 or u32", name)
            }
            level = .exact(value.handle)
        case "textureSampleGrad":
            let x = try coerce(args[levelIndex], to: coordinate, at: name)
            let y = try coerce(args[levelIndex + 1], to: coordinate, at: name)
            level = .gradient(x: x.handle, y: y.handle)
        default:
            let value = try coerce(args[levelIndex], to: intern(.scalar(.f32)), at: name)
            level = name.text == "textureSampleBias" ? .bias(value.handle) : .exact(value.handle)
        }
        let sample = IRImageSample(image: args[0].handle, sampler: args[1].handle, gather: gather,
                                   coordinate: uv.handle, arrayIndex: arrayIndex, offset: offset, level: level, depthReference: reference,
                                   clampToEdge: clampToEdge)
        let result: IRTypeKind = depth && gather == nil ? .scalar(scalar) : .vector(size: .four, scalar: scalar)
        return try expressionNode(.imageSample(sample), type: intern(result), at: name)
    }

    private func gatherStartsWithDepthTexture() throws -> Bool {
        // Texture handles in this frontend are names, possibly wrapped in parentheses.
        var index = cursor
        while tokens[index].text == "(" {
            try budget.consume(.work)
            index += 1
        }
        let name = tokens[index]
        guard name.kind == .word else { return false }
        for _ in cursor..<index {
            index += 1
            guard tokens[index].text == ")" else { return false }
        }
        guard tokens[index + 1].text == "," else { return false }

        try budget.consume(.work, amount: name.text.utf8.count)
        let key = wgslKey(name.text)
        for scope in body.scopes.reversed() {
            try budget.consume(.work)
            if let value = scope[key] { return types[value.type.index].kind.isDepthTexture }
        }
        guard let declaration = declarationNames[key], declarations[declaration].kind == "var" else { return false }
        try resolveValueDeclaration(name)
        return globalNames[key].map { types[globals[$0].type.index].kind.isDepthTexture } ?? false
    }

    private func textureConstantArgument() throws -> WGSLScalarConstant {
        let first = token
        let start = cursor
        var depth = 0
        // Keep nested constructor arguments together while finding the call argument boundary.
        while token.kind != .end {
            if depth == 0 && (isNext(")") || isNext(",")) { break }
            let next = try take()
            if next.opensGroup { depth += 1 }
            else if next.closesGroup { depth -= 1 }
        }
        return try scalarConstant(Array(tokens[start..<cursor]), at: first)
    }

    private func textureSampleOffset(width: IRVectorSize?) throws -> WExpr {
        let first = token
        let argument = try textureConstantArgument()
        guard let width else {
            guard case .integer(let value) = argument, value.kind == .sint || value.kind == .abstract else {
                throw wgslError(.typeMismatch, "1D texture sample offset must be a constant i32", first)
            }
            guard (-8...7).contains(value.value) else {
                throw wgslError(.invalidType, "Texture sample offset must be between -8 and 7", first)
            }
            return try expressionNode(.literal(.i32(Int32(value.value))), type: intern(.scalar(.i32)),
                                      at: first, emit: false).handle
        }
        guard case .vector(let value) = argument, value.elements.count == Int(width.rawValue),
              [.integer(.abstract), .integer(.sint)].contains(value.element) else {
            throw wgslError(.typeMismatch, "Texture sample offset must be a constant vec\(width.rawValue)i", first)
        }
        for component in value.elements {
            guard case .integer(let lane) = component, (-8...7).contains(lane.value) else {
                throw wgslError(.invalidType, "Texture sample offset components must be between -8 and 7", first)
            }
        }
        let type = try intern(.vector(size: width, scalar: .i32))
        return try vectorRuntimeValue(value, to: type, at: first).handle
    }

    private func textureBuiltin(_ args: [WGSLValue], at name: WGSLToken) throws -> WGSLValue {
        if let first = args.first, types[first.type.index].kind.isExternalTexture {
            let kind = types[first.type.index].kind
            switch (name.text, args.count) {
            case ("textureDimensions", 1):
                return try expressionNode(.imageQuery(image: first.handle, query: .size(level: nil)),
                                          type: intern(kind.texelCoordinateType(.u32)), at: name)
            case ("textureLoad", 2):
                let coordinate = try texelCoordinates(args[1], image: kind, at: name)
                return try expressionNode(.imageLoad(.init(image: first.handle, coordinate: coordinate.handle)),
                                          type: intern(.vector(size: .four, scalar: .f32)), at: name)
            default:
                throw wgslError(.typeMismatch, "Invalid operation on an external texture", name)
            }
        }
        let arity: ClosedRange<Int>
        switch name.text {
        case "textureLoad": arity = 3...4
        case "textureDimensions": arity = 1...2
        default: arity = 1...1
        }
        if let first = args.first, let storage = types[first.type.index].kind.storageTexture {
            let kind = types[first.type.index].kind
            let arrayed = kind.isArrayedTexture
            if name.text == "textureDimensions", args.count == 1 {
                return try expressionNode(.imageQuery(image: first.handle, query: .size(level: nil)),
                                          type: intern(kind.texelCoordinateType(.u32)), at: name)
            }
            if name.text == "textureNumLayers", args.count == 1, arrayed {
                return try expressionNode(.imageQuery(image: first.handle, query: .numLayers),
                                          type: intern(.scalar(.u32)), at: name)
            }
            if name.text == "textureLoad", args.count == (arrayed ? 3 : 2), storage.access.contains(.load) {
                let coordinates = try texelCoordinates(args[1], image: kind, at: name)
                let layer = arrayed ? try textureArrayIndex(args[2], at: name) : nil
                return try expressionNode(.imageLoad(.init(image: first.handle, coordinate: coordinates.handle, arrayIndex: layer)),
                                          type: intern(.vector(size: .four, scalar: storage.scalar)), at: name)
            }
            throw wgslError(.typeMismatch, "Invalid operation on storage texture", name)
        }
        guard arity.contains(args.count), let first = args.first,
              let scalar = types[first.type.index].kind.sampledTextureScalar else {
            throw wgslError(.typeMismatch, "Expected a sampled texture and matching builtin arguments", name)
        }
        let image = first.handle
        let kind = types[first.type.index].kind
        let multisampled = kind.isMultisampled2D
        let level: WGSLValue?
        if args.count > 1 {
            let value = try defaultMaterialize(args[args.count - 1], at: name)
            guard [.scalar(.i32), .scalar(.u32)].contains(types[value.type.index].kind) else {
                throw wgslError(.typeMismatch, multisampled ? "Texture sample index must be i32 or u32" :
                    "Texture level must be i32 or u32", name)
            }
            level = value
        } else {
            level = nil
        }
        switch name.text {
        case "textureLoad":
            guard !kind.isCubeTexture, args.count == (kind.isArrayedTexture ? 4 : 3) else {
                throw wgslError(.typeMismatch, "Texture load arguments do not match the texture dimension", name)
            }
            let layer = kind.isArrayedTexture ? try textureArrayIndex(args[2], at: name) : nil
            let coordinates = try texelCoordinates(args[1], image: kind, at: name)
            let result: IRTypeKind = kind.isDepthTexture ? .scalar(scalar) : .vector(size: .four, scalar: scalar)
            return try expressionNode(.imageLoad(.init(image: image, coordinate: coordinates.handle, arrayIndex: layer,
                sample: multisampled ? level?.handle : nil, level: multisampled ? nil : level?.handle)),
                                      type: intern(result), at: name)
        case "textureDimensions":
            guard !multisampled || level == nil else {
                throw wgslError(.typeMismatch, "Multisampled texture dimensions do not take a mip level", name)
            }
            return try expressionNode(.imageQuery(image: image, query: .size(level: level?.handle)),
                                      type: intern(kind.texelCoordinateType(.u32)), at: name)
        case "textureNumLayers":
            guard kind.isArrayedTexture else {
                throw wgslError(.typeMismatch, "Expected an array texture for textureNumLayers", name)
            }
            return try expressionNode(.imageQuery(image: image, query: .numLayers),
                                      type: intern(.scalar(.u32)), at: name)
        default:
            let samples = name.text == "textureNumSamples"
            guard samples == multisampled else {
                throw wgslError(.typeMismatch, "Texture query does not match the texture sample count", name)
            }
            return try expressionNode(.imageQuery(image: image, query: samples ? .numSamples : .numLevels),
                                      type: intern(.scalar(.u32)), at: name)
        }
    }

    func textureArrayIndex(_ value: WGSLValue, at name: WGSLToken) throws -> WExpr {
        let layer = try defaultMaterialize(value, at: name)
        guard [.scalar(.i32), .scalar(.u32)].contains(types[layer.type.index].kind) else {
            throw wgslError(.typeMismatch, "Texture array index must be i32 or u32", name)
        }
        return layer.handle
    }

    func texelCoordinates(_ value: WGSLValue, image: IRTypeKind, at name: WGSLToken) throws -> WGSLValue {
        let coordinates = try defaultMaterialize(value, at: name)
        guard [image.texelCoordinateType(.i32), image.texelCoordinateType(.u32)]
            .contains(types[coordinates.type.index].kind) else {
            throw wgslError(.typeMismatch, "Texture coordinates must match the texture dimension and use i32 or u32", name)
        }
        return coordinates
    }

    private func selectValue(_ args: [WGSLValue], at name: WGSLToken) throws -> WGSLValue {
        guard args.count == 3 else { throw wgslError(.typeMismatch, "select requires three arguments", name) }
        let first = args[0]
        let second = args[1]
        let type: WType
        if !isAbstract(first.type) {
            type = first.type
        } else if !isAbstract(second.type) {
            type = second.type
        } else {
            switch types[second.type.index].kind {
            case .scalar(.abstractFloat), .vector(_, .abstractFloat):
                type = try defaultMaterialize(second, at: name).type
            default:
                type = try defaultMaterialize(first, at: name).type
            }
        }
        let rejected = try coerce(first, to: type, at: name)
        let accepted = try coerce(second, to: type, at: name)
        return try expressionNode(.select(condition: args[2].handle, accept: accepted.handle, reject: rejected.handle),
                                  type: type, at: name)
    }

    func atomicPointer(at name: WGSLToken) throws -> (WGSLValue, WType) {
        let value = try dereference(expression(), at: name)
        guard value.writable, case .atomic(let scalar) = types[value.type.index].kind else {
            throw wgslError(.typeMismatch, "Atomic operation requires the address of a writable atomic value", name)
        }
        return (value, try intern(.scalar(scalar)))
    }

    private func atomicCompareResult(for type: WType, at name: WGSLToken) throws -> WType {
        guard case .scalar(let scalar) = types[type.index].kind else { throw ShaderFailure.invalidHandle }
        let kind = IRPredeclaredType.atomicCompareExchangeWeakResult(scalar)
        if let existing = predeclaredTypes.first(where: { $0.kind == kind }) { return existing.type }
        let boolean = try intern(.scalar(.bool))
        let width = UInt32(scalar.width)
        let members: [IRStructMember] = [
            .init(name: "old_value", type: type, offset: 0),
            .init(name: "exchanged", type: boolean, offset: width),
        ]
        let lane = "\(scalar.kind == .sint ? "i" : "u")\(width * 8)"
        let result = try intern(.structure(members: members, span: width * 2),
                                name: "__atomic_compare_result_\(lane)", at: name)
        predeclaredTypes.append(.init(kind: kind, type: result))
        return result
    }
}
