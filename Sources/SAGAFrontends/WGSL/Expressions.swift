import SAGAShaderIR

private struct ShiftGroup {
    var hasShift = false
    var hasBitwise = false
}

extension WGSLParser {
    func literal(_ t: WGSLToken) throws -> (IRLiteral,IRScalar) {
        if t.text == "true" || t.text == "false" { return (.bool(t.text == "true"),.bool) }
        guard t.kind == .number else { throw wgslError(.unexpectedToken,"Expected numeric/bool literal",t) }
        try budget.consume(.work,amount:t.text.utf8.count)
        var text = t.text
        let isHex = text.hasPrefix("0x") || text.hasPrefix("0X")
        let floating = text.contains(".") || (isHex ? text.lowercased().contains("p") : text.lowercased().contains("e"))
        if text.hasSuffix("h") { throw unsupported("f16 literal (enable f16 also required)",at:t) }
        // In hexadecimal integers 'f' is a digit, not a suffix.
        let explicitFloat = text.hasSuffix("f") && (!isHex || text.lowercased().contains("p"))
        if floating || explicitFloat {
            if explicitFloat { text.removeLast() }
            if isHex {return try hexFloat(text,explicitF32:explicitFloat,at:t)}
            guard !text.hasSuffix("u"), !text.hasSuffix("i"), let d = Double(text), d.isFinite else { throw wgslError(.invalidToken,"Invalid float literal",t) }
            if explicitFloat {
                guard let f = Float(text), f.isFinite else { throw wgslError(.invalidToken,"f32 literal overflow",t) }; return (.f32(f),.f32)
            }
            return (.abstractFloat(d),.abstractFloat)
        }
        let suffix = text.last
        let explicitU = suffix == "u", explicitI = suffix == "i"
        if explicitU || explicitI { text.removeLast() }
        let digits = isHex ? String(text.dropFirst(2)) : text
        if !isHex && digits.count > 1 && digits.hasPrefix("0") { throw wgslError(.invalidToken,"Leading zero integer",t) }
        guard let number = UInt64(digits,radix:isHex ? 16 : 10) else { throw wgslError(.invalidToken,"Integer literal overflow or missing digits",t) }
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
            if s == .u32, let v = UInt32(exactly:n) { return .u32(v) }
            if s == .i32, let v = Int32(exactly:n) { return .i32(v) }
            if s == .f32 { let v = Float(n); if v.isFinite { return .f32(v) } }
            if s == .abstractFloat { return .abstractFloat(Double(n)) }
        case .abstractFloat(let n):
            if s == .f32, abs(n) <= Double(Float.greatestFiniteMagnitude) { let v = Float(n); if v.isFinite { return .f32(v) } }
        default: break
        }
        throw wgslError(.typeMismatch,"Literal cannot be materialized as requested type",t)
    }
    func statement(_ kind: IRStatementKind, at t: WGSLToken? = nil) throws {
        try node(); let h = try IRHandle<IRStatement>(index:body.statements.count)
        body.statements.append(IRStatement(kind:kind,location:t?.location)); body.current.append(h)
    }
    func expressionNode(_ kind: IRExpressionKind, type: WType, at t: WGSLToken, emit: Bool = true) throws -> WGSLValue {
        try node(); let h = try WExpr(index:body.expressions.count)
        body.expressions.append(IRExpression(type:type,kind:kind,location:t.location))
        if emit { try statement(.emit([h]),at:t) }
        return WGSLValue(handle:h,type:type,pointer:false,writable:false)
    }
    func loaded(_ value: WGSLValue, at t: WGSLToken) throws -> WGSLValue {
        if !value.pointer { return value }
        if case .atomic = types[value.type.index].kind { throw unsupported("atomic read without atomicLoad",at:t) }
        return try expressionNode(.load(pointer:value.handle),type:value.type,at:t)
    }
    func coerce(_ value: WGSLValue, to target: WType, at t: WGSLToken) throws -> WGSLValue {
        let v = try loaded(value,at:t)
        if v.type == target { return v }
        let kind = types[v.type.index].kind
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
            switch g.space { case .uniform: writable = false; case .storage(let access): writable = access.contains(.store); default:writable = true }
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
        while true {
            try budget.consume(.work)
            let operatorStart = cursor
            let op = expressionOperator()
            guard let (precedence, binary) = Self.binary[op.text], precedence >= minimum else { break }
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
            if left.type != right.type {
                if isAbstract(left.type) { left = try coerce(left,to:right.type,at:op) }
                else if isAbstract(right.type) { right = try coerce(right,to:left.type,at:op) }
            }
            guard left.type == right.type else { throw wgslError(.typeMismatch,"Binary operands must match",op) }
            let comparison = ["==","!=","<",">","<=",">="].contains(op.text)
            let operandKind = types[left.type.index].kind
            let s: IRScalar
            if case .scalar(let scalar) = operandKind { s = scalar }
            else { throw unsupported("vector/matrix binary overload",at:op) }
            if s == .bool {
                guard ["==","!=","&","|"].contains(op.text) else { throw wgslError(.typeMismatch,"Invalid bool operator",op) }
            } else if ["&","|","^","%"].contains(op.text) && (s.kind == .float || s.kind == .abstractFloat) { throw wgslError(.typeMismatch,"Integer operand required",op) }
            if (binary == .divide || binary == .modulo) && [.sint, .uint, .abstractInt].contains(s.kind),
               let divisor = try knownIntegerConstant(rightStart..<cursor, at: op) {
                guard divisor.value != 0 else { throw wgslError(.invalidType,"Constant division or remainder by zero",op) }
                if divisor.value == -1, s == .i32,
                   try knownIntegerConstant(leftStart..<operatorStart, at: op)?.value == Int64(Int32.min) {
                    throw wgslError(.invalidType,"Constant signed division or remainder overflow",op)
                }
            }
            let result = comparison ? try intern(.scalar(.bool)) : left.type
            if let folded = try fold(binary,left:left,right:right,at:op) {
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
        if ["-","!","~"].contains(t.text) {
            _ = try take(); let v = try loaded(prefix(),at:t)
            if case .vector(_, let scalar) = types[v.type.index].kind {
                guard (t.text == "-" && scalar == .i32) ||
                      (t.text == "~" && (scalar == .i32 || scalar == .u32)) else {
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
            } else if (["vec2", "vec3", "vec4"].contains(t.text) || WGSLScalarEvaluator.matrixNames[t.text] != nil),
               declarationNames[wgslKey(t.text)] == nil, cursor + 1 < tokens.count, tokens[cursor + 1].text == "(" {
                _ = try take(); try expect("("); value = try inferredComposite(t)
            } else if isTypeName(t.text) {
                let target = try type(); try expect("("); value = try construct(target,at:t)
            } else {
                let n = try ident()
                if try eat("(") { value = try call(n) }
                else { value = try lookup(n) }
            }
        } else { throw wgslError(.unexpectedToken,"Expected expression",t) }
        while true {
            if try eat("[") {
                let indexAt=token
                let index = try defaultMaterialize(expression(),at:indexAt); try expect("]")
                if case .vector(let width,_)=types[value.type.index].kind,
                   let literal=try abstractScalarLeaf(index.handle,at:indexAt) {
                    let number:Int64
                    switch literal {case .u32(let x):number=Int64(x);case .i32(let x):number=Int64(x);default:throw wgslError(.typeMismatch,"Index must be integer",indexAt)}
                    guard number >= 0,number < Int64(width.rawValue) else {throw wgslError(.invalidType,"Constant vector index out of bounds",indexAt)}
                }
                let indexType = types[index.type.index].kind
                guard indexType == .scalar(.u32) || indexType == .scalar(.i32) else { throw wgslError(.typeMismatch,"Index must be i32/u32",t) }
                let element: WType
                switch types[value.type.index].kind {
                case .array(let base,_,_):element = base
                case .vector(_,let scalar):element = try intern(.scalar(scalar))
                case .matrix(_,let rows,let scalar):element = try intern(.vector(size:rows,scalar:scalar))
                default:throw wgslError(.typeMismatch,"Value not indexable",t)
                }
                if value.pointer {
                    let space = try pointerSpace(value)
                    let p = try expressionNode(.access(base:value.handle,index:index.handle),type:intern(.pointer(base:element,space:space)),at:t)
                    value = WGSLValue(handle:p.handle,type:element,pointer:true,writable:value.writable)
                } else {
                    value = try expressionNode(.access(base:value.handle,index:index.handle),type:element,at:t)
                }
            } else if try eat(".") {
                let field = try ident()
                switch types[value.type.index].kind {
                case .structure(let members,_):
                    var match: Int?
                    for (i,m) in members.enumerated() { try budget.consume(.work,amount:field.text.utf8.count+1); if m.name?.utf8.elementsEqual(field.text.utf8) == true { match = i; break } }
                    guard let i = match else { throw wgslError(.unresolvedName,"No such struct member",field) }
                    let ty = members[i].type
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
                    if indices.count == 1 {
                        let ty = try intern(.scalar(scalar))
                        let resultType = value.pointer ? try intern(.pointer(base:ty,space:pointerSpace(value))) : ty
                        let p = try expressionNode(.accessIndex(base:value.handle,index:UInt32(indices[0])),type:resultType,at:field)
                        value = WGSLValue(handle:p.handle,type:ty,pointer:value.pointer,writable:value.writable)
                    } else {
                        let v = try loaded(value,at:field)
                        let components: [IRSwizzleComponent] = [.x,.y,.z,.w]
                        while indices.count < 4 { indices.append(0) }
                        guard let outSize = IRVectorSize(rawValue:UInt8(chars.count)) else { throw wgslError(.invalidType,"Bad swizzle",field) }
                        value = try expressionNode(.swizzle(size:outSize,vector:v.handle,pattern:IRSwizzle(x:components[indices[0]],y:components[indices[1]],z:components[indices[2]],w:components[indices[3]])),type:intern(.vector(size:outSize,scalar:scalar)),at:field)
                    }
                default: throw wgslError(.typeMismatch,"Member access on noncomposite",field)
                }
            } else { break }
        }
        return value
    }
    func pointerSpace(_ value: WGSLValue) throws -> IRAddressSpace {
        guard case .pointer(_,let space) = types[body.expressions[value.handle.index].type.index].kind else { throw wgslError(.invalidType,"Expected pointer",token) }; return space
    }
    func isTypeName(_ name: String) -> Bool {
        predeclaredType(name) != nil || typeNames[wgslKey(name)] != nil || declarationNames[wgslKey(name)].map { ["alias","struct"].contains(declarations[$0].kind) } == true || ["u32","i32","f32","bool","vec2","vec3","vec4","array","mat2x2","mat2x3","mat2x4","mat3x2","mat3x3","mat3x4","mat4x2","mat4x3","mat4x4"].contains(name)
    }
    func arguments() throws -> [WGSLValue] {
        var args: [WGSLValue] = []
        while !isNext(")") {
            try node(); args.append(try loaded(expression(),at:token))
            if !(try eat(",")) { break }
        }
        try expect(")"); return args
    }
    func construct(_ target: WType, at t: WGSLToken, supplied: [WGSLValue]? = nil) throws -> WGSLValue {
        var args = try supplied ?? arguments()
        if args.isEmpty {
            switch types[target.index].kind {
            case .scalar,.vector,.matrix:break
            default:throw unsupported("zero composite constructor",at:t)
            }
            return try expressionNode(.zeroValue(target),type:target,at:t,emit:false)
        }
        switch types[target.index].kind {
        case .scalar(let scalar):
            guard args.count == 1 else { throw wgslError(.typeMismatch,"Scalar constructor arity",t) }
            if args[0].type == target { return args[0] }
            if isAbstract(args[0].type) { return try coerce(args[0],to:target,at:t) }
            guard case .scalar(let from) = types[args[0].type.index].kind, scalar != .bool, from != .bool else { throw unsupported("bool conversion",at:t) }
            return try expressionNode(.cast(expression:args[0].handle,target:scalar),type:target,at:t)
        case .vector(let size,let scalar):
            let elem = try intern(.scalar(scalar))
            if args.count == 1, case .vector(let sourceSize, let source) = types[args[0].type.index].kind,
               sourceSize == size, source != scalar,
               [.i32, .u32].contains(source), [.i32, .u32, .f32].contains(scalar) {
                return try expressionNode(.cast(expression: args[0].handle, target: scalar), type: target, at: t)
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
            if args.count == 1,case .matrix = types[args[0].type.index].kind {
                // The currently enabled source scalar is f32; another scalar's explicit conversion
                // must be added with its capability/type validation, not shape-coerced here.
                return try coerce(args[0],to:target,at:t)
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
        if name.text == "workgroupUniformLoad", declarationNames[wgslKey(name.text)] == nil {
            try expect("&")
            let pointer = try expression()
            guard pointer.pointer, try pointerSpace(pointer) == .workgroup else {
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
            try expect("&")
            let array = try expression()
            guard array.pointer, case .storage = try pointerSpace(array),
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
        if name.text == "select", declarationNames[wgslKey(name.text)] == nil {
            return try selectValue(args, at: name)
        }
        if ["textureLoad", "textureDimensions", "textureNumLevels", "textureNumSamples", "textureNumLayers"].contains(name.text),
           declarationNames[wgslKey(name.text)] == nil {
            return try textureBuiltin(args, at: name)
        }
        // Deliberately bounded overload table, not unknown-name passthrough.
        let math: [String:(IRMathFunction,Int)] = ["abs":(.abs,1),"min":(.min,2),"max":(.max,2),"clamp":(.clamp,3),"sqrt":(.sqrt,1),"sin":(.sin,1),"cos":(.cos,1),"floor":(.floor,1),"ceil":(.ceil,1)]
        guard let (op,arity) = math[name.text] else { throw unsupported("builtin or forward function '\(name.text)'",at:name) }
        guard args.count == arity, let first = args.first else { throw wgslError(.typeMismatch,"Builtin arity",name) }
        // Select an explicit concrete consensus before default-materializing abstract arguments.
        // A large abstract value can fit u32 even when default i32 would be unrepresentable.
        var consensus: WType?
        for argument in args {
            try budget.consume(.work)
            if !isAbstract(argument.type) {
                if let consensus, consensus != argument.type {throw wgslError(.typeMismatch,"Builtin concrete argument types disagree",name)}
                consensus = argument.type
            }
        }
        let concrete: WGSLValue
        if let consensus { concrete = try coerce(first,to:consensus,at:name) }
        else if args.contains(where: { types[$0.type.index].kind == .scalar(.abstractFloat) }) {
            concrete = try defaultMaterialize(coerce(first,to:intern(.scalar(.abstractFloat)),at:name),at:name)
        } else { concrete = try defaultMaterialize(first,at:name) }
        args[0] = concrete
        guard case .scalar(let scalar) = types[concrete.type.index].kind, scalar != .bool else { throw unsupported("vector/matrix builtin overload",at:name) }
        if !["abs","min","max","clamp"].contains(name.text) && scalar.kind != .float { throw wgslError(.typeMismatch,"Float builtin operand required",name) }
        if name.text == "abs" && scalar.kind == .uint { throw wgslError(.typeMismatch,"abs requires signed/float",name) }
        for i in args.indices.dropFirst() { args[i] = try coerce(args[i],to:concrete.type,at:name) }
        return try expressionNode(.math(function:op,argument:args[0].handle,argument1:args.count>1 ? args[1].handle:nil,argument2:args.count>2 ? args[2].handle:nil,argument3:nil),type:concrete.type,at:name)
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
        if volume && gather != nil {
            throw wgslError(.typeMismatch, "Volume textures do not support texture gathers", name)
        }
        if clampToEdge && imageType != .sampledFloat2D {
            throw wgslError(.typeMismatch, "Base clamp sampling requires texture_2d<f32>", name)
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
            guard !imageType.isCubeTexture else {
                throw wgslError(.typeMismatch, "Cube textures do not take texel offsets", token)
            }
            offset = try textureSampleOffset(width: volume ? .three : .two)
            _ = try eat(",")
        } else {
            offset = nil
        }
        try expect(")")
        let comparison = ["textureSampleCompare", "textureSampleCompareLevel", "textureGatherCompare"].contains(name.text)
        let depth = imageType.isDepthTexture
        let validDepth = comparison || depthGather || name.text == "textureSample" || name.text == "textureSampleLevel"
        guard let scalar = imageType.sampledTextureScalar, !imageType.isMultisampled2D,
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
        let vector = try intern(.vector(size: imageType.isCubeTexture || volume ? .three : .two, scalar: .f32))
        let uv = try coerce(args[2], to: vector, at: name)
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
            let x = try coerce(args[levelIndex], to: vector, at: name)
            let y = try coerce(args[levelIndex + 1], to: vector, at: name)
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
            switch try take().text {
            case "(": depth += 1
            case ")": depth -= 1
            default: break
            }
        }
        return try scalarConstant(Array(tokens[start..<cursor]), at: first)
    }

    private func textureSampleOffset(width: IRVectorSize) throws -> WExpr {
        let first = token
        guard case .vector(let value) = try textureConstantArgument(), value.elements.count == Int(width.rawValue),
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
        let arity: ClosedRange<Int>
        switch name.text {
        case "textureLoad": arity = 3...4
        case "textureDimensions": arity = 1...2
        default: arity = 1...1
        }
        if let first = args.first, let storage = types[first.type.index].kind.storageTexture2D {
            let arrayed = types[first.type.index].kind.isArrayedTexture
            if name.text == "textureDimensions", args.count == 1 {
                return try expressionNode(.imageQuery(image: first.handle, query: .size(level: nil)),
                                          type: intern(.vector(size: .two, scalar: .u32)), at: name)
            }
            if name.text == "textureNumLayers", args.count == 1, arrayed {
                return try expressionNode(.imageQuery(image: first.handle, query: .numLayers),
                                          type: intern(.scalar(.u32)), at: name)
            }
            if name.text == "textureLoad", args.count == (arrayed ? 3 : 2), storage.access.contains(.load) {
                let coordinates = try texelCoordinates(args[1], at: name)
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
        let width: IRVectorSize = kind.isVolumeTexture ? .three : .two
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
            let coordinates = try texelCoordinates(args[1], width: width, at: name)
            let result: IRTypeKind = kind.isDepthTexture ? .scalar(scalar) : .vector(size: .four, scalar: scalar)
            return try expressionNode(.imageLoad(.init(image: image, coordinate: coordinates.handle, arrayIndex: layer,
                sample: multisampled ? level?.handle : nil, level: multisampled ? nil : level?.handle)),
                                      type: intern(result), at: name)
        case "textureDimensions":
            guard !multisampled || level == nil else {
                throw wgslError(.typeMismatch, "Multisampled texture dimensions do not take a mip level", name)
            }
            return try expressionNode(.imageQuery(image: image, query: .size(level: level?.handle)),
                                      type: intern(.vector(size: width, scalar: .u32)), at: name)
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

    func texelCoordinates(_ value: WGSLValue, width: IRVectorSize = .two, at name: WGSLToken) throws -> WGSLValue {
        let coordinates = try defaultMaterialize(value, at: name)
        guard [.vector(size: width, scalar: .i32), .vector(size: width, scalar: .u32)]
            .contains(types[coordinates.type.index].kind) else {
            throw wgslError(.typeMismatch, "Texture coordinates must be vec\(width.rawValue)i or vec\(width.rawValue)u", name)
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
        try expect("&")
        let value = try expression()
        guard value.pointer, value.writable, case .atomic(let scalar) = types[value.type.index].kind else {
            throw wgslError(.typeMismatch, "Atomic operation requires the address of a writable atomic value", name)
        }
        return (value, try intern(.scalar(scalar)))
    }

    private func atomicCompareResult(for type: WType, at name: WGSLToken) throws -> WType {
        guard case .scalar(let scalar) = types[type.index].kind else { throw ShaderFailure.invalidHandle }
        let kind = IRPredeclaredType.atomicCompareExchangeWeakResult(scalar)
        if let existing = predeclaredTypes.first(where: { $0.kind == kind }) { return existing.type }
        let boolean = try intern(.scalar(.bool))
        let members: [IRStructMember] = [
            .init(name: "old_value", type: type, offset: 0),
            .init(name: "exchanged", type: boolean, offset: 4),
        ]
        let result = try intern(.structure(members: members, span: 8),
                                name: scalar == .i32 ? "__atomic_compare_result_i32" : "__atomic_compare_result_u32", at: name)
        predeclaredTypes.append(.init(kind: kind, type: result))
        return result
    }
}
