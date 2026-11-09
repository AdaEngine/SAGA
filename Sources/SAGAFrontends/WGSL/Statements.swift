import SAGAShaderIR

extension WGSLParser {
    func define(_ n: WGSLToken, _ v: WGSLValue) throws {
        try budget.consume(.work,amount:n.text.utf8.count+1)
        guard let last = body.scopes.indices.last else { throw ShaderFailure.invalidHandle }
        guard body.scopes[last][wgslKey(n.text)] == nil else { throw wgslError(.duplicateName,"Duplicate local '\(n.text)'",n) }
        body.scopes[last][wgslKey(n.text)] = v
    }
    func function(_ attrs: [String:Attribute]) throws {
        try checkAttributes(attrs,allowed:["compute","fragment","vertex","workgroup_size","subgroup_size","diagnostic","early_depth_test","must_use"])
        let name = try ident(); try reserve(name)
        let diagnostic = try functionDiagnosticFilters(attrs)
        body = WGSLFunctionBody()
        try expect("(")
        while !isNext(")") {
            let a = try attributes(), n = try ident(); try expect(":"); let ty = try type()
            if case .pointer(_, let space) = types[ty.index].kind, space != .function, space != .private {
                allowsPointerSubviews = true
            }
            try node()
            let index = UInt32(body.arguments.count)
            body.arguments.append(IRFunctionArgument(name:n.text,type:ty,binding:try binding(a),location:n.location))
            let v = try expressionNode(.functionArgument(index),type:ty,at:n,emit:false)
            try define(n,v)
            if !(try eat(",")) { break }
        }
        try expect(")")
        if try eat("->") {
            let a = try attributes(); body.result = IRFunctionResult(type:try type(),binding:try binding(a))
        }
        if let attribute = attrs["must_use"], body.result == nil {
            throw wgslError(.invalidType, "@must_use requires a function with a return type", attribute.name)
        }
        let root = try block(newScope:false)
        if try body.result == nil && !terminates(root) {
            let saved = body.current; body.current = body.blocks[root.index].statements
            try statement(.returnValue(nil),at:name)
            body.blocks[root.index] = IRBlock(statements:body.current,location:body.blocks[root.index].location); body.current = saved
        }
        let stageNames = ["vertex","fragment","compute"].filter { attrs[$0] != nil }
        guard stageNames.count <= 1 else { throw wgslError(.invalidBinding,"Conflicting stages",name) }
        if stageNames.isEmpty && attrs["workgroup_size"] != nil { throw wgslError(.invalidBinding,"workgroup_size requires compute stage",name) }
        for s in stageNames {
            if let a = attrs[s], !a.arguments.isEmpty { throw wgslError(.unexpectedToken,"Stage attribute has no arguments",a.name) }
        }
        let subgroupSize: IRSubgroupSize?
        if let attribute = attrs["subgroup_size"] {
            guard stageNames == ["compute"] else {
                throw wgslError(.invalidBinding, "subgroup_size requires a compute entry point", attribute.name)
            }
            guard subgroupSizeControlEnabled else {
                throw wgslError(.unsupportedCapability, "subgroup_size requires subgroup_size_control", attribute.name)
            }
            subgroupSize = try subgroupSizeConfiguration(attribute)
        } else {
            subgroupSize = nil
        }
        var earlyDepthTest: IREarlyDepthTest?
        if let attribute = attrs["early_depth_test"], let mode = attribute.arguments.first {
            guard stageNames == ["fragment"] else {
                throw wgslError(.invalidControlFlow, "@early_depth_test requires a fragment entry point", attribute.name)
            }
            switch mode.text {
            case "force": earlyDepthTest = .force
            case "greater_equal": earlyDepthTest = .allow(.greaterEqual)
            case "less_equal": earlyDepthTest = .allow(.lessEqual)
            case "unchanged": earlyDepthTest = .allow(.unchanged)
            default: throw wgslError(.unexpectedToken, "Unknown early depth test mode", mode)
            }
        }
        let fh = try IRHandle<IRFunction>(index:functions.count)
        if let s = stageNames.first {
            let stage: IRStage = s == "compute" ? .compute : s == "vertex" ? .vertex : .fragment
            var workgroup = IRWorkgroupSize()
            var workgroupOverrides: IRWorkgroupOverrides?
            if s == "compute" {
                guard let a = attrs["workgroup_size"], (1...3).contains(a.expressions.count) else {
                    throw wgslError(.invalidBinding, "Compute needs workgroup_size with 1...3 dimensions", name)
                }
                (workgroup, workgroupOverrides) = try workgroupConfiguration(a)
                guard body.result == nil else { throw wgslError(.invalidBinding,"Compute cannot return a value",name) }
            } else if attrs["workgroup_size"] != nil { throw wgslError(.invalidBinding,"Non-compute workgroup_size",name) }
            try node()
            entries.append(.init(name: name.text, stage: stage, function: fh, earlyDepthTest: earlyDepthTest,
                                 workgroupSize: workgroup, workgroupOverrides: workgroupOverrides,
                                 subgroupSize: subgroupSize, location: name.location))
        } else {
            guard body.arguments.allSatisfy({ $0.binding == nil }), body.result?.binding == nil else { throw wgslError(.invalidBinding,"Non-entry function has IO attributes",name) }
        }
        try node()
        functions.append(IRFunction(name:name.text,arguments:body.arguments,result:body.result,localVariables:IRTable(body.locals),expressions:IRTable(body.expressions),statements:IRTable(body.statements),blocks:IRTable(body.blocks),body:root,diagnosticFilter:diagnostic,location:name.location))
        functionNames[wgslKey(name.text)] = fh.index
        if attrs["must_use"] != nil { mustUseFunctions.insert(fh.index) }
        body = WGSLFunctionBody()
    }
    func block(newScope: Bool = true) throws -> IRHandle<IRBlock> {
        try enter(); defer { nesting -= 1 }
        let start = try expect("{")
        let previous = body.current; body.current = []
        if newScope { body.scopes.append([:]);body.constScopes.append([:]) }
        defer { body.current = previous; if newScope { body.scopes.removeLast();body.constScopes.removeLast() } }
        while !isNext("}") {
            guard token.kind != .end else { throw wgslError(.unexpectedToken,"Unterminated block",token) }
            try parseStatement()
        }
        try expect("}")
        return try saveBlock(body.current,at:start)
    }
    func saveBlock(_ statements: [IRHandle<IRStatement>], at t: WGSLToken) throws -> IRHandle<IRBlock> {
        try node(); let h = try IRHandle<IRBlock>(index:body.blocks.count)
        body.blocks.append(IRBlock(statements:statements,location:t.location)); return h
    }
    func terminates(_ block: IRHandle<IRBlock>) throws -> Bool {
        try enter(); defer { nesting -= 1 }
        return try body.blocks[block.index].statements.contains(where: statementTerminates)
    }
    func statementTerminates(_ h: IRHandle<IRStatement>) throws -> Bool {
        try budget.consume(.work)
        switch body.statements[h.index].kind {
        case .returnValue,.break,.continue: return true
        case .block(let b):return try terminates(b)
        case .ifCondition(_,let a,let b):return try terminates(a) && terminates(b)
        case .switchValue(_, let cases):
            var visited = Set<IRHandle<IRBlock>>()
            for entry in cases where visited.insert(entry.body).inserted {
                if try !terminates(entry.body) || breaksSwitch(entry.body) { return false }
            }
            return true
        default:return false
        }
    }

    private func breaksSwitch(_ block: IRHandle<IRBlock>) throws -> Bool {
        try enter(); defer { nesting -= 1 }
        for statement in body.blocks[block.index].statements {
            try budget.consume(.work)
            switch body.statements[statement.index].kind {
            case .break: return true
            case .block(let child):
                if try breaksSwitch(child) { return true }
            case .ifCondition(_, let yes, let no):
                if try breaksSwitch(yes) || breaksSwitch(no) { return true }
            default: break
            }
            if try statementTerminates(statement) { break }
        }
        return false
    }
    func parseStatement() throws {
        try budget.consume(.work)
        let start = token
        if try eat(";") { return }
        if isNext("{") { let b = try block(); try statement(.block(b),at:start); return }
        if isNext("const_assert") {try constAssertion();return}
        if try eat("const") {try localConst(at:start);return}
        if try eat("let") { try local(mutable:false,at:start); return }
        if try eat("var") { try local(mutable:true,at:start); return }
        if try eat("return") {
            if try eat(";") {
                guard body.result == nil else { throw wgslError(.typeMismatch,"Missing return value",start) }
                try statement(.returnValue(nil),at:start)
            } else {
                guard let result = body.result else { throw wgslError(.typeMismatch,"Void function cannot return value",start) }
                let v = try coerce(expression(),to:result.type,at:start); try expect(";"); try statement(.returnValue(v.handle),at:start)
            }; return
        }
        if try eat("if") { try ifStatement(at:start); return }
        if try eat("while") { try whileStatement(at:start); return }
        if try eat("loop") { try loopStatement(at:start); return }
        if try eat("for") { try forStatement(at:start); return }
        if try eat("switch") { try switchStatement(at:start); return }
        if try eat("break") {
            guard (body.loopDepth > 0 || body.switchDepth > 0) && !body.continuing else { throw wgslError(.invalidControlFlow,"Invalid break context",start) }
            try expect(";"); try statement(.break,at:start); return
        }
        if try eat("continue") {
            guard body.loopDepth > 0 && !body.continuing else { throw wgslError(.invalidControlFlow,"Invalid continue context",start) }
            try noteContinue()
            try expect(";"); try statement(.continue,at:start); return
        }
        if try eat("discard") { try expect(";"); try statement(.demoteToHelperInvocation,at:start); return }
        if ["continuing"].contains(start.text) { throw unsupported("statement '\(start.text)'",at:start) }
        try simpleStatement(end: ";")
    }

    func simpleStatement(end: String) throws {
        let start = token
        let key = wgslKey(start.text)
        if try eat("_") {
            try expect("=")
            _ = try loaded(expression(), at: start)
            try expect(end)
            return
        }
        if start.kind == .word, cursor + 1 < tokens.count, tokens[cursor + 1].text == "(",
           !isTypeName(start.text) {
            try resolveFunctionDeclaration(start)
            if let index = functionNames[key], functions[index].result == nil {
                guard !entries.contains(where: { $0.function.index == index }) else {
                    throw wgslError(.invalidControlFlow, "Cannot call entry point", start)
                }
                _ = try take()
                try expect("(")
                var args = try arguments()
                let parameters = functions[index].arguments
                guard args.count == parameters.count else { throw wgslError(.typeMismatch, "Function arity", start) }
                for i in args.indices { args[i] = try coerce(args[i], to: parameters[i].type, at: start) }
                try expect(end)
                try statement(.call(function: try IRHandle(index: index), arguments: args.map(\.handle), result: nil), at: start)
                return
            }
            if declarationNames[key] == nil {
                if start.text == "debugPrintf" { try debugPrintfStatement(at: start, end: end); return }
                if start.text == "atomicMin" || start.text == "atomicMax" {
                    _ = try take()
                    try expect("(")
                    let (pointer, type) = try atomicPointer(at: start)
                    try expect(",")
                    let value = try coerce(expression(), to: type, at: start)
                    _ = try eat(",")
                    try expect(")")
                    try expect(end)
                    let result: WExpr?
                    if [.scalar(.u64), .scalar(.i64)].contains(types[type.index].kind) {
                        result = nil
                    } else {
                        result = try expressionNode(.atomicResult(type: type, comparison: false),
                                                    type: type, at: start, emit: false).handle
                    }
                    try statement(.atomic(pointer: pointer.handle, function: start.text == "atomicMin" ? .min : .max,
                                          value: value.handle, result: result), at: start)
                    return
                }
                if start.text == "atomicStore" {
                    _ = try take()
                    try expect("(")
                    let (pointer, type) = try atomicPointer(at: start)
                    try expect(",")
                    let value = try coerce(expression(), to: type, at: start)
                    _ = try eat(",")
                    try expect(")")
                    try expect(end)
                    try statement(.store(pointer: pointer.handle, value: value.handle), at: start)
                    return
                }
                if ["workgroupBarrier", "storageBarrier", "textureBarrier", "subgroupBarrier"].contains(start.text) {
                    _ = try take()
                    try expect("(")
                    try expect(")")
                    try expect(end)
                    let flag: IRBarrier = switch start.text {
                    case "workgroupBarrier": .workgroup
                    case "storageBarrier": .storage
                    case "textureBarrier": .texture
                    default: .subgroup
                    }
                    try statement(.controlBarrier(flag), at: start)
                    return
                }
            }
        }
        let textureAtomic: IRAtomicFunction? = switch start.text {
        case "textureAtomicAdd": .add
        case "textureAtomicMin": .min
        case "textureAtomicMax": .max
        case "textureAtomicAnd": .and
        case "textureAtomicOr": .inclusiveOr
        case "textureAtomicXor": .exclusiveOr
        default: nil
        }
        if start.text == "textureStore" || textureAtomic != nil, declarationNames[key] == nil,
           cursor + 1 < tokens.count, tokens[cursor + 1].text == "(" {
            _ = try take()
            try expect("(")
            let args = try arguments()
            guard let image = args.first else { throw wgslError(.typeMismatch, "Missing storage texture", start) }
            let kind = types[image.type.index].kind
            let arrayed = kind.isArrayedTexture
            guard let storage = kind.storageTexture, storage.access.contains(.store),
                  args.count == (arrayed ? 4 : 3) else {
                throw wgslError(.typeMismatch, "textureStore requires a writable storage texture and matching arguments", start)
            }
            let coordinate = try texelCoordinates(args[1], image: kind, at: start)
            let layer = arrayed ? try textureArrayIndex(args[2], at: start) : nil
            if textureAtomic != nil, !storage.access.contains(.atomic) {
                throw wgslError(.typeMismatch, "Texture atomic operation requires atomic access", start)
            }
            let valueType: IRTypeKind = textureAtomic == nil ? .vector(size: .four, scalar: storage.scalar) : .scalar(storage.scalar)
            let value = try coerce(args[args.count - 1], to: intern(valueType), at: start)
            try expect(end)
            if let textureAtomic {
                let root: IRExpressionKind
                switch body.expressions[image.handle.index].kind {
                case .access(let base, _), .accessIndex(let base, _): root = body.expressions[base.index].kind
                case let kind: root = kind
                }
                guard case .globalVariable = root else {
                    throw wgslError(.invalidReference, "Texture atomics require a global texture binding", start)
                }
                try statement(.imageAtomic(image: image.handle, coordinate: coordinate.handle,
                                           arrayIndex: layer, function: textureAtomic, value: value.handle), at: start)
            } else {
                try statement(.imageStore(image: image.handle, coordinate: coordinate.handle,
                                          arrayIndex: layer, value: value.handle), at: start)
            }
            return
        }
        let leftStart = cursor
        let lhs = try expression()
        let leftTokens = leftStart..<cursor
        if try eat(end) {
            guard try isStandaloneCall(leftTokens) else {
                throw wgslError(.unexpectedToken, "Only a call may be an expression statement", start)
            }
            switch body.expressions[lhs.handle.index].kind {
            case .callResult(let callee):
                if functionNames[key] == callee.index && !mustUseFunctions.contains(callee.index) { return }
            case .atomicResult where declarationNames[key] == nil:
                return
            case .load where start.text == "atomicLoad" && declarationNames[key] == nil:
                return
            default:
                break
            }
            throw wgslError(.unusedResult, "The result of '\(start.text)' must be used", start)
        }
        guard lhs.pointer && lhs.writable else { throw wgslError(.invalidAddressSpace,"Assignment needs writable reference",start) }
        let op = expressionOperator()
        _ = try take()
        if op.text == "<<=" || op.text == ">>=" { _ = try take() }
        let rhs: WGSLValue
        if op.text == "=" {
            rhs = try coerce(expression(), to: lhs.type, at: op)
        } else if ["+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="].contains(op.text),
                  let (_, binary) = Self.binary[String(op.text.dropLast())] {
            let left = try loaded(lhs, at: op)
            let rightStart = cursor
            let operand = try loaded(expression(), at: op)
            rhs = try compoundValue(binary, left: left, right: operand, leftTokens: leftTokens,
                                    rightTokens: rightStart..<cursor, at: op)
        } else if op.text == "++" || op.text == "--" {
            let l = try loaded(lhs,at:op)
            let lit: IRLiteral
            if types[lhs.type.index].kind == .scalar(.u32) { lit = .u32(1) }
            else if types[lhs.type.index].kind == .scalar(.i32) { lit = .i32(1) }
            else { throw wgslError(.typeMismatch,"Increment requires integer",op) }
            let one = try expressionNode(.literal(lit),type:lhs.type,at:op,emit:false)
            rhs = try expressionNode(.binary(op:op.text == "++" ? .add : .subtract,left:l.handle,right:one.handle),type:lhs.type,at:op)
        } else { throw wgslError(.unexpectedToken,"Expected assignment",op) }
        try expect(end)
        if let view = lhs.swizzle {
            try writeSwizzleView(lhs, view: view, update: rhs, at: start)
        } else {
            try statement(.store(pointer: lhs.handle, value: rhs.handle), at: start)
        }
    }

    private func isStandaloneCall(_ range: Range<Int>) throws -> Bool {
        let start = range.lowerBound
        guard range.count >= 3, tokens[start].kind == .word,
              tokens[range.upperBound - 1].text == ")" else { return false }
        try budget.consume(.work, amount: range.count)
        // Find the opening parenthesis of the final argument list without mistaking a
        // parenthesized expression inside a template argument or a trailing operand for the call.
        var depth = 0
        for index in range.reversed() {
            switch tokens[index].text {
            case ")": depth += 1
            case "(":
                depth -= 1
                if depth == 0 {
                    if index == start + 1 { return true }
                    return tokens[start + 1].kind == .templateStart && tokens[index - 1].kind == .templateEnd &&
                        (isTypeName(tokens[start].text) || tokens[start].text == "bitcast")
                }
            default: break
            }
        }
        return false
    }

    private func compoundValue(
        _ op: IRBinaryOperator, left: WGSLValue, right: WGSLValue, leftTokens: Range<Int>,
        rightTokens: Range<Int>, at token: WGSLToken
    ) throws -> WGSLValue {
        if op == .shiftLeft || op == .shiftRight {
            return try integerShift(op, left: left, right: right, leftTokens: leftTokens, rightTokens: rightTokens, at: token)
        }
        if let value = try matrixBinary(op, left: left, right: right, at: token) {
            return try coerce(value, to: left.type, at: token)
        }
        if let value = try integerVectorBinary(op, left: left, right: right,
                                              leftTokens: leftTokens, rightTokens: rightTokens, at: token) {
            return try coerce(value, to: left.type, at: token)
        }
        if let value = try floatingVectorBinary(op, left: left, right: right, at: token) {
            return try coerce(value, to: left.type, at: token)
        }
        let operand = try coerce(right, to: left.type, at: token)
        switch types[left.type.index].kind {
        case .scalar(.bool), .vector(_, .bool):
            guard op == .and || op == .inclusiveOr else {
                throw wgslError(.typeMismatch, "Boolean compound assignment requires &= or |=", token)
            }
            return try expressionNode(.binary(op: op, left: left.handle, right: operand.handle), type: left.type, at: token)
        case .scalar(let scalar):
            if (op == .divide || op == .modulo), scalar.kind == .sint || scalar.kind == .uint,
               let divisor = try knownIntegerConstant(rightTokens, at: token), divisor.value == 0 {
                throw wgslError(.invalidType, "Constant divisor is zero", token)
            }
            return try expressionNode(.binary(op: op, left: left.handle, right: operand.handle), type: left.type, at: token)
        default:
            throw unsupported("compound assignment overload", at: token)
        }
    }

    func local(mutable: Bool, at t: WGSLToken) throws {
        if try eat("<") {
            guard mutable else { throw wgslError(.unexpectedToken,"let cannot have address space",t) }
            try expect("function"); try expect(">")
        }
        let n = try ident()
        let annotation = try eat(":") ? try type() : nil
        let hasInit = try eat("=")
        let initial = hasInit ? try expression() : nil
        guard mutable || initial != nil else { throw wgslError(.unexpectedToken,"let requires initializer",n) }
        guard annotation != nil || initial != nil else { throw wgslError(.invalidType,"var requires type or initializer",n) }
        let v: WGSLValue?
        if let initial { v = try annotation.map { try coerce(initial,to:$0,at:n) } ?? defaultMaterialize(initial,at:n) } else { v = nil }
        guard let ty = annotation ?? v?.type else { throw ShaderFailure.invalidHandle }
        if mutable {
            switch types[ty.index].kind {
            case .atomic, .array(_, .dynamic, _): throw wgslError(.invalidType,"Invalid function variable type",n)
            default: break
            }
            try node(); let local = try IRHandle<IRLocalVariable>(index:body.locals.count)
            body.locals.append(IRLocalVariable(name:n.text,type:ty,location:n.location))
            let ptr = try expressionNode(.localVariable(local),type:intern(.pointer(base:ty,space:.function)),at:n,emit:false)
            try define(n,WGSLValue(handle:ptr.handle,type:ty,pointer:true,writable:true))
            let value = try v ?? expressionNode(.zeroValue(ty),type:ty,at:n,emit:false)
            try statement(.store(pointer:ptr.handle,value:value.handle),at:n)
        } else if let v {
            let scalar: Bool
            if case .scalar = types[ty.index].kind { scalar = true } else { scalar = false }
            if !types[ty.index].kind.isTextureOrSampler, body.pipelineStages[v.handle.index] != .runtime,
               scalar || hasOverrideDeclarations {
                let snapshot = try expressionNode(.runtimeValue(v.handle), type: ty, at: n)
                try define(n, snapshot)
            } else {
                try define(n, v)
            }
        }
        try expect(";")
    }
    func ifStatement(at t: WGSLToken) throws {
        try enter(); defer { nesting -= 1 }
        let cond = try loaded(expression(),at:t)
        guard types[cond.type.index].kind == .scalar(.bool) else { throw wgslError(.typeMismatch,"if condition requires bool",t) }
        let accept = try block()
        let reject: IRHandle<IRBlock>
        if try eat("else") {
            if try eat("if") {
                let previous = body.current; body.current = []
                try ifStatement(at:t); reject = try saveBlock(body.current,at:t); body.current = previous
            } else { reject = try block() }
        } else { reject = try saveBlock([],at:t) }
        try statement(.ifCondition(condition:cond.handle,accept:accept,reject:reject),at:t)
    }
    func whileStatement(at t: WGSLToken) throws {
        let previous = body.current; body.current = []
        let cond = try loaded(expression(),at:t)
        guard types[cond.type.index].kind == .scalar(.bool) else { throw wgslError(.typeMismatch,"while needs bool",t) }
        let empty = try saveBlock([],at:t)
        let beforeBreak = body.current; body.current = []
        try statement(.break,at:t); let breakBlock = try saveBlock(body.current,at:t); body.current = beforeBreak
        try statement(.ifCondition(condition:cond.handle,accept:empty,reject:breakBlock),at:t)
        body.loopDepth += 1
        let loopBody = try block()
        body.loopDepth -= 1
        try statement(.block(loopBody),at:t)
        let whole = try saveBlock(body.current,at:t); body.current = previous
        let continuing = try saveBlock([],at:t)
        try statement(.loop(body:whole,continuing:continuing,breakIf:nil),at:t)
    }
}
