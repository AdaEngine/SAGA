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
            let g = globals[i], pointerType = try intern(.pointer(base:globals[i].type,space:globals[i].space))
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
               (source == .i32 || source == .u32) && (scalar == .i32 || scalar == .u32) {
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
}
