import SAGAShaderIR

/// Independent Swift GLSL frontend. Returns raw IR, never a semantic-validation witness.
/// Unsupported language features fail explicitly; see glsl-frontend-coverage.json.
public struct GLSLFrontend: BoundedShaderFrontend {
    public typealias Module = IRModule
    public let options: GLSLFrontendOptions
    public let limits: ShaderLimits
    public var inputFormat: ShaderInputFormat { .glsl }
    public init(options: GLSLFrontendOptions, limits: ShaderLimits = .default) { self.options=options; self.limits=limits }
    public func parse(_ input: ShaderInput, budget: inout ShaderBudget) throws -> IRModule {
        guard case .glsl(let source, let stage, let defines)=input else { throw ShaderFailure.unsupported("Expected GLSL input") }
        guard IRStage(stage)==options.stage else { throw ShaderFailure.unsupported("GLSL input stage and options disagree") }
        var merged=options.defines
        for (key,value) in defines {
            try budget.consume(.work,amount:key.utf8.count+value.utf8.count)
            if let old=merged[key], old != value { throw ShaderFailure.unsupported("Conflicting GLSL define: \(key)") }
            merged[key]=value
        }
        // Options-only definitions were not included in the generic wrapper's input quota.
        for (key,value) in options.defines where defines[key]==nil {
            try budget.consume(.sourceBytes,amount:key.utf8.count+value.utf8.count)
        }
        return try parseSource(source, defines:merged, budget:&budget)
    }
    private func parseSource(_ source:String, defines:[String:String], budget:inout ShaderBudget) throws -> IRModule {
        let tokens=try GLSLScanner.scan(source,budget:&budget)
        var preprocessor=GLSLPreprocessor()
        let expanded=try preprocessor.process(tokens,defines:defines,budget:&budget)
        let parser=GLSLParser(tokens:expanded,sourceLength:source.utf8.count,stage:options.stage,
                              version:preprocessor.version,budget:budget)
        defer { budget=parser.budget }
        return try parser.parse()
    }
}

struct GLSLValue {
    let expression: IRHandle<IRExpression>
    let type: IRHandle<IRType>
    let pointer: Bool
    let writable: Bool
    let swizzle: GLSLWritableSwizzle?
    let combinedSampler: IRHandle<IRExpression>?

    init(expression: IRHandle<IRExpression>, type: IRHandle<IRType>, pointer: Bool, writable: Bool,
         swizzle: GLSLWritableSwizzle? = nil, combinedSampler: IRHandle<IRExpression>? = nil) {
        self.expression = expression
        self.type = type
        self.pointer = pointer
        self.writable = writable
        self.swizzle = swizzle
        self.combinedSampler = combinedSampler
    }
}
struct GLSLWritableSwizzle {
    let pointerType: IRHandle<IRType>
    let indices: [UInt32]
}
struct GLSLSymbol { let value:GLSLValue }
struct GLSLInterface {
    let global:IRHandle<IRGlobalVariable>
    let type:IRHandle<IRType>
    var binding:IRBinding
    let input:Bool
    let name:String
}
struct GLSLScope {
    var symbols: [String: GLSLSymbol] = [:]
    var types: [String: IRHandle<IRType>] = [:]
}

final class GLSLFunctionContext {
    var expressions:[IRExpression]=[]
    var locals:[IRLocalVariable]=[]
    var statements:[IRStatement]=[]
    var blocks:[IRBlock]=[]
    var current:[IRHandle<IRStatement>]=[]
    var scopes = [GLSLScope()]
    var loopDepth=0
    var switches: [(loopDepth: Int, hasBreak: Bool)] = []
    let result:IRHandle<IRType>?
    init(result:IRHandle<IRType>?) { self.result=result }
}

/// Recursive descent statements + precedence-climbing expressions. All recursive entries charge depth.
final class GLSLParser {
    let tokens:[GLSLToken]
    let stage:IRStage
    let version:Int
    var budget:ShaderBudget
    var i=0
    var depth=0
    var types:[IRType]=[]
    var namedTypes:[String:IRHandle<IRType>]=[:]
    var globals:[IRGlobalVariable]=[]
    var interfaces:[GLSLInterface]=[]
    var functions:[IRFunction]=[]
    var functionNames:[String:[IRHandle<IRFunction>]]=[:]
    var globalNames:[String:IRHandle<IRGlobalVariable>]=[:]
    var uninstancedBlockMembers: [String: (block: IRHandle<IRGlobalVariable>, index: UInt32, type: IRHandle<IRType>)] = [:]
    var textureRoles: [GLSLTextureOrigin: GLSLTextureRole] = [:]
    var constantNames: [String: IRHandle<IRConstant>] = [:]
    var constants: [IRConstant] = []
    var globalExpressions: [IRExpression] = []
    var workgroup=IRWorkgroupSize()
    var hasLocalSize=false
    var context:GLSLFunctionContext?
    init(tokens:[GLSLToken],sourceLength:Int,stage:IRStage,version:Int,budget:ShaderBudget) {
        self.tokens=tokens + [.init(text:"",start:sourceLength,end:sourceLength)]
        self.stage=stage; self.version=version; self.budget=budget
    }
    var token:GLSLToken { tokens[i] }
    var text:String { token.text }
    func advance() throws -> GLSLToken {
        try budget.consume(.work)
        let t=token; if i<tokens.count-1 { i+=1 }; return t
    }
    @discardableResult func expect(_ s:String) throws -> GLSLToken {
        guard text==s else { throw glslError(token,.unexpectedToken,"Expected '\(s)', got '\(text)'") }; return try advance()
    }
    func take(_ s:String) throws -> Bool { if text==s { _ = try advance(); return true }; return false }
    func identifier(allowNamedType: Bool = false) throws -> GLSLToken {
        guard token.barycentricPerspective == nil else {
            throw glslError(token, .unexpectedToken, "Cannot redeclare a barycentric builtin")
        }
        guard !token.isViewIndex else {
            throw glslError(token, .unexpectedToken, "Cannot redeclare gl_ViewIndex")
        }
        guard glslIdentifier(text), !["void","if","else","while","for","return","break","continue","discard","true","false","struct","switch","case","default","do"].contains(text) else { throw glslError(token,.unexpectedToken,"Expected identifier") }
        let type = try allowNamedType ? builtinType(text, float16Enabled: token.float16Enabled) : resolveType(token)
        guard type == nil else { throw glslError(token,.unexpectedToken,"Type name cannot be identifier") }
        return try advance()
    }
    func enter() throws { try budget.consume(.work); guard depth<budget.limits.depth else { throw ShaderFailure.budgetExceeded(.depth) }; depth+=1 }
    func addType(_ kind:IRTypeKind) throws -> IRHandle<IRType> {
        for (n,t) in types.enumerated() { try budget.consume(.work); if t.kind==kind { return try .init(index:n) } }
        try budget.consume(.nodes); let h=try IRHandle<IRType>(index:types.count); types.append(.init(kind:kind)); return h
    }
    func kind(_ type:IRHandle<IRType>) -> IRTypeKind { types[type.index].kind }
    func scalar(_ type:IRHandle<IRType>) -> IRScalar? {
        switch kind(type) { case .scalar(let s), .vector(_,let s), .matrix(_,_,let s): return s; default:return nil }
    }
    func namedType(_ name: String) throws -> IRHandle<IRType>? {
        if let context {
            for scope in context.scopes.reversed() {
                try budget.consume(.work, amount: name.utf8.count)
                if let type = scope.types[name] { return type }
            }
        }
        return namedTypes[name]
    }
    func resolveType(_ token: GLSLToken) throws -> IRHandle<IRType>? {
        if let type = try namedType(token.text) { return type }
        return try builtinType(token.text, float16Enabled: token.float16Enabled)
    }
    private func builtinType(_ name:String, float16Enabled: Bool = false) throws -> IRHandle<IRType>? {
        if let resource = glslOpaqueResourceKind(name) { return try addType(resource) }
        if float16Enabled, name == "float16_t" { return try addType(.scalar(.f16)) }
        let scalars:[String:IRScalar]=["bool":.bool,"int":.i32,"uint":.u32,"float":.f32,"double":.init(kind:.float,width:8)]
        if let s=scalars[name] { return try addType(.scalar(s)) }
        let prefixes:[(String,IRScalar)]=[("vec",.f32),("ivec",.i32),("uvec",.u32),("bvec",.bool),
                                        ("dvec",.init(kind:.float,width:8)),("f16vec",.f16)]
        for (prefix,s) in prefixes where name.hasPrefix(prefix) {
            guard s != .f16 || float16Enabled else { return nil }
            if let n=UInt8(name.dropFirst(prefix.count)), let size=IRVectorSize(rawValue:n) { return try addType(.vector(size:size,scalar:s)) }
        }
        let isHalfMatrix = float16Enabled && name.hasPrefix("f16mat")
        if name.hasPrefix("mat") || isHalfMatrix {
            let component: IRScalar = isHalfMatrix ? .f16 : .f32
            let dimensions = Array(name.dropFirst(isHalfMatrix ? 6 : 3).prefix(4))
            guard let first = dimensions.first, let width = UInt8(String(first)),
                  let columns = IRVectorSize(rawValue: width) else { return nil }
            if dimensions.count == 1 {
                return try addType(.matrix(columns: columns, rows: columns, scalar: component))
            }
            if dimensions.count == 3, dimensions[1] == "x", let height = UInt8(String(dimensions[2])),
               let rows = IRVectorSize(rawValue: height) {
                return try addType(.matrix(columns: columns, rows: rows, scalar: component))
            }
        }
        return nil
    }
    func parseType() throws -> IRHandle<IRType>? {
        if try take("void") { return nil }
        let t=try advance(); guard let type=try resolveType(t) else { throw glslError(t,.unsupportedConstruct,"Unsupported GLSL type \(t.text)") }; return type
    }
    func parse() throws -> IRModule {
        guard [.compute,.vertex,.fragment].contains(stage) else { throw glslError(token,.unsupportedConstruct,"GLSL stage not implemented") }
        while !text.isEmpty { try externalDeclaration() }
        guard let main=functionNames["main"]?.first, functions[main.index].result==nil, functions[main.index].arguments.isEmpty else { throw glslError(token,.invalidControlFlow,"Expected void main()") }
        try finalizeGLSLColorBindings()
        try finalizeGLSLComparisonTextures()
        let wrapper=try wrap(main)
        return IRModule(types:.init(types),constants:.init(constants),globalVariables:.init(globals),
                        globalExpressions:.init(globalExpressions),functions:.init(functions),
                        entryPoints:[.init(name:"main",stage:stage,function:wrapper,workgroupSize:stage == .compute ? workgroup : .init(x:0,y:0,z:0))])
    }
    func registerInterface(_ name:String,type:IRHandle<IRType>,binding:IRBinding,input:Bool) throws {
        try budget.consume(.nodes)
        let h=try IRHandle<IRGlobalVariable>(index:globals.count)
        globals.append(.init(name:name,space:.private,type:type)); globalNames[name]=h
        interfaces.append(.init(global:h,type:type,binding:binding,input:input,name:name))
    }
    func function(_ name:GLSLToken,result:IRHandle<IRType>?) throws {
        let overloads = functionNames[name.text] ?? []
        guard (!hasGlobalName(name.text) || !overloads.isEmpty), !name.text.hasPrefix("gl_"),
              name.text != "main" || overloads.isEmpty else {
            throw glslError(name,.duplicateName,"Duplicate/reserved function")
        }
        let c=GLSLFunctionContext(result:result); context=c
        var arguments:[IRFunctionArgument]=[]
        try expect("(")
        if text=="void" { _ = try advance(); try expect(")") }
        else if !(try take(")")) {
            repeat {
                _ = try take("in")
                guard let type=try parseType() else { throw glslError(token,.invalidType,"Void parameter") }; let arg=try identifier()
                guard c.scopes[0].symbols[arg.text]==nil else { throw glslError(arg,.duplicateName,"Duplicate parameter") }
                let value=try expressionNode(type,.functionArgument(UInt32(arguments.count)),at:arg,emit:false)
                if kind(type).isTextureOrSampler {
                    c.scopes[0].symbols[arg.text] = .init(value: value)
                    arguments.append(.init(name: arg.text, type: type))
                    continue
                }
                try budget.consume(.nodes); let local=try IRHandle<IRLocalVariable>(index:c.locals.count)
                c.locals.append(.init(name:arg.text,type:type,location:try arg.location))
                let pointer=try expressionNode(addType(.pointer(base:type,space:.function)),.localVariable(local),at:arg,emit:false,pointer:true,writable:true)
                try append(.store(pointer:pointer.expression,value:value.expression),at:arg)
                c.scopes[0].symbols[arg.text] = .init(value:pointer); arguments.append(.init(name:arg.text,type:type))
            } while try take(",")
            try expect(")")
        }
        let signature = arguments.map(\.type)
        for overload in overloads {
            try budget.consume(.work, amount: signature.count + 1)
            if functions[overload.index].arguments.map(\.type) == signature {
                throw glslError(name, .duplicateName, "Duplicate function signature")
            }
        }
        let parameterSetup=c.current;c.current=[]
        let (body,terminal)=try compound(pushScope:false)
        if result != nil && !terminal { throw glslError(name,.invalidControlFlow,"Function may exit without return") }
        // A void fallthrough is explicit. Do not place a statement after a terminator.
        var blocks=c.blocks
        blocks[body.index] = .init(statements:parameterSetup+blocks[body.index].statements)
        if !terminal { let s=try statementNode(.returnValue(nil),at:name); blocks[body.index] = .init(statements:blocks[body.index].statements+[s]) }
        try budget.consume(.nodes)
        let h=try IRHandle<IRFunction>(index:functions.count)
        functions.append(.init(name:name.text,arguments:arguments,result:result.map { .init(type:$0) },localVariables:.init(c.locals),expressions:.init(c.expressions),statements:.init(c.statements),blocks:.init(blocks),body:body,location:try name.location))
        functionNames[name.text, default: []].append(h); context=nil
    }
    func requireContext() throws -> GLSLFunctionContext {
        guard let c=context else { throw ShaderFailure.unsupported("Function context required") }; return c
    }
    func expressionNode(_ type:IRHandle<IRType>,_ kind:IRExpressionKind,at t:GLSLToken,emit:Bool=true,pointer:Bool=false,writable:Bool=false) throws -> GLSLValue {
        let c=try requireContext(); try budget.consume(.nodes)
        let h=try IRHandle<IRExpression>(index:c.expressions.count)
        c.expressions.append(.init(type:type,kind:kind,location:try t.location))
        if emit { try append(.emit([h]),at:t) }
        return .init(expression:h,type:type,pointer:pointer,writable:writable)
    }
    func statementNode(_ kind:IRStatementKind,at t:GLSLToken) throws -> IRHandle<IRStatement> {
        let c=try requireContext(); try budget.consume(.nodes); let h=try IRHandle<IRStatement>(index:c.statements.count)
        c.statements.append(.init(kind:kind,location:try t.location)); return h
    }
    func append(_ kind:IRStatementKind,at t:GLSLToken) throws { let h=try statementNode(kind,at:t); try requireContext().current.append(h) }
    func block(_ statements:[IRHandle<IRStatement>]) throws -> IRHandle<IRBlock> {
        let c=try requireContext(); try budget.consume(.nodes); let h=try IRHandle<IRBlock>(index:c.blocks.count); c.blocks.append(.init(statements:statements)); return h
    }
    func compound(pushScope:Bool=true) throws -> (IRHandle<IRBlock>,Bool) {
        try enter(); defer { depth-=1 }; try expect("{"); let c=try requireContext()
        let saved=c.current; c.current=[]; if pushScope { c.scopes.append(GLSLScope()) }
        defer { c.current=saved; if pushScope { c.scopes.removeLast() } }
        var terminal=false
        while text != "}" {
            if text.isEmpty { throw glslError(token,.unexpectedToken,"Unterminated block") }
            if terminal { throw glslError(token,.unsupportedConstruct,"Unreachable statements not lowered") }
            terminal=try statement()
        }
        try expect("}"); return (try block(c.current),terminal)
    }
    func childStatement() throws -> (IRHandle<IRBlock>,Bool) {
        if text=="{" { return try compound() }
        let c=try requireContext(); let saved=c.current; c.current=[]; c.scopes.append(GLSLScope())
        defer { c.current=saved; c.scopes.removeLast() }
        let terminal=try statement(); return (try block(c.current),terminal)
    }
    func statement() throws -> Bool {
        try enter(); defer { depth-=1 }; let t=token; let c=try requireContext()
        if text=="{" { let (b,end)=try compound(); try append(.block(b),at:t); return end }
        if try take(";") { return false }
        if try take("if") {
            try expect("("); let condition=try rvalue(expression(),at:t); try requireBool(condition,at:t); try expect(")")
            let (a,at)=try childStatement(); let (r,rt):(IRHandle<IRBlock>,Bool)
            if try take("else") { (r,rt)=try childStatement() } else { r=try block([]); rt=false }
            try append(.ifCondition(condition:condition.expression,accept:a,reject:r),at:t); return at && rt
        }
        if try take("for") { try forLoop(at:t); return false }
        if try take("switch") { return try switchStatement(at: t) }
        if try take("while") {
            try expect("(")
            let saved=c.current; c.current=[]
            let condition=try rvalue(expression(),at:t); try requireBool(condition,at:t); try expect(")")
            let check=c.current; c.current=saved
            c.loopDepth+=1; let (body,_)=try childStatement(); c.loopDepth-=1
            let stop=try block([try statementNode(.break,at:t)])
            let empty=try block([])
            let branch=try statementNode(.ifCondition(condition:condition.expression,accept:empty,reject:stop),at:t)
            let child=try statementNode(.block(body),at:t)
            let loopBody=try block(check+[branch,child])
            let continuing=try block([])
            try append(.loop(body:loopBody,continuing:continuing,breakIf:nil),at:t); return false
        }
        if try take("return") {
            if try take(";") { guard c.result==nil else { throw glslError(t,.typeMismatch,"Return requires value") }; try append(.returnValue(nil),at:t) }
            else {
                let source = try rvalue(expression(), at: t)
                guard let result = c.result, let value = try implicitlyConvert(source, to: result, at: t) else {
                    throw glslError(t, .typeMismatch, "Return type mismatch")
                }
                try expect(";")
                try append(.returnValue(value.expression), at: t)
            }; return true
        }
        if text=="break" || text=="continue" {
            let isBreak = text == "break"
            _ = try advance()
            if isBreak, let selection = c.switches.indices.last, c.switches[selection].loopDepth == c.loopDepth {
                c.switches[selection].hasBreak = true
            } else if c.loopDepth == 0 {
                throw glslError(t, .invalidControlFlow, "Jump outside loop or switch")
            }
            try expect(";")
            try append(isBreak ? .break : .continue, at: t)
            return true
        }
        if try take("discard") { guard stage == .fragment else { throw glslError(t,.invalidControlFlow,"Discard outside fragment") }; try expect(";"); try append(.kill,at:t); return true }
        if ["barrier", "memoryBarrierShared", "memoryBarrierBuffer"].contains(text) {
            guard stage == .compute else {
                throw glslError(t,.invalidControlFlow,"Workgroup barrier outside compute")
            }
            _ = try advance()
            try expect("(")
            try expect(")")
            try expect(";")
            switch t.text {
            case "barrier": try append(.controlBarrier(.workgroup), at: t)
            case "memoryBarrierShared": try append(.memoryBarrier(.workgroup), at: t)
            default: try append(.memoryBarrier(.storage), at: t)
            }
            return false
        }
        if try take("precision") { try defaultPrecisionDeclaration(); return false }
        if text == "do" { throw glslError(t,.unsupportedConstruct,"Unsupported GLSL statement: \(text)") }
        var immutable=false
        if try take("const") { immutable=true }
        let base: IRHandle<IRType>?
        if try take("struct") {
            let name = try identifier(allowNamedType: true)
            let scope = c.scopes.count - 1
            guard c.scopes[scope].types[name.text] == nil, c.scopes[scope].symbols[name.text] == nil,
                  !name.text.hasPrefix("gl_") else {
                throw glslError(name, .duplicateName, "Duplicate/reserved local type")
            }
            let type = try members(runtime: false)
            c.scopes[scope].types[name.text] = type
            if !immutable, try take(";") { return false }
            base = type
        } else {
            base = try resolveType(token)
            if base != nil { _ = try advance() }
        }
        if let base {
            let name=try identifier();let type=try arraySuffix(base)
            guard c.scopes[c.scopes.count-1].symbols[name.text]==nil, !name.text.hasPrefix("gl_") else { throw glslError(name,.duplicateName,"Duplicate/reserved local") }
            if immutable, kind(type) == .scalar(.i32) || kind(type) == .scalar(.u32) {
                try expect("=")
                let constant = try integerConstant(at: name, convertingTo: type)
                let value = try expressionNode(type, .literal(constant.literal), at: name, emit: false)
                c.scopes[c.scopes.count - 1].symbols[name.text] = .init(value: value)
                try expect(";")
                return false
            }
            try budget.consume(.nodes); let local=try IRHandle<IRLocalVariable>(index:c.locals.count)
            c.locals.append(.init(name:name.text,type:type,location:try name.location))
            let pointerType=try addType(.pointer(base:type,space:.function))
            let pointer=try expressionNode(pointerType,.localVariable(local),at:name,emit:false,pointer:true,writable:!immutable)
            // Initializer resolves in enclosing scope, not the new declaration itself.
            if try take("=") {
                let source = try rvalue(expression(), at: name)
                guard let value = try implicitlyConvert(source, to: type, at: name) else {
                    throw glslError(name, .typeMismatch, "Initializer type mismatch")
                }
                if immutable, case .literal = c.expressions[source.expression.index].kind {} else if immutable { throw glslError(name,.unsupportedConstruct,"Nonliteral const evaluation not implemented") }
                try append(.store(pointer:pointer.expression,value:value.expression),at:name)
            } else if immutable { throw glslError(name,.unexpectedToken,"Const requires initializer") }
            c.scopes[c.scopes.count-1].symbols[name.text] = .init(value:pointer); try expect(";"); return false
        }
        if immutable { throw glslError(t,.unsupportedConstruct,"Unsupported const type") }
        try simpleStatement(end:";"); return false
    }
    func simpleStatement(end:String) throws {
        try simpleExpression()
        try expect(end)
    }
    func simpleExpression() throws {
        let t=token
        let left: GLSLValue
        if let overloads = functionNames[t.text], t.text != "main",
           tokens.indices.contains(i + 1), tokens[i + 1].text == "(" {
            try enter()
            defer { depth -= 1 }
            _ = try advance()
            try expect("(")
            let args = try callArguments(at: t)
            guard let value = try userCall(overloads, arguments: args, allowingVoid: true, at: t) else { return }
            left = try finishExpression(value, minimum: 0)
        } else {
            left = try expression()
        }
        if ["=","+=","-=","*=","/=","%=","&=","|=","^=","<<=",">>="].contains(text) {
            let op=try advance()
            try requireWritable(left, operation: "Assignment", at: t)
            // Capture the previous value before evaluation of RHS side effects.
            let previous=op.text != "=" ? try rvalue(left,at:t):nil
            var right=try rvalue(expression(),at:op)
            let target = left.pointer ? try pointee(left.type, at: t) : left.type
            if let previous {
                let operation = GLSLToken(text: String(op.text.dropLast()), start: op.start, end: op.end)
                right = try binary(previous, right, op: operation)
            }
            guard let value = try implicitlyConvert(right, to: target, at: op) else {
                throw glslError(op, .typeMismatch, "Assignment type mismatch")
            }
            try storeValue(value, in: left, at: op)
        } else { _ = try rvalue(left,at:t) }
    }

    func requireWritable(_ target: GLSLValue, operation: String, at token: GLSLToken) throws {
        guard target.writable, target.pointer || target.swizzle != nil else {
            throw glslError(token, .invalidAddressSpace, "\(operation) requires writable lvalue")
        }
        if let swizzle = target.swizzle, Set(swizzle.indices).count != swizzle.indices.count {
            throw glslError(token, .invalidAddressSpace, "Writable swizzle must select distinct components")
        }
    }

    func storeValue(_ value: GLSLValue, in target: GLSLValue, at token: GLSLToken) throws {
        if let swizzle = target.swizzle {
            try storeSwizzle(target.expression, swizzle, value: value, at: token)
        } else {
            try append(.store(pointer: target.expression, value: value.expression), at: token)
        }
    }

    func pointee(_ type:IRHandle<IRType>,at t:GLSLToken) throws -> IRHandle<IRType> {
        guard case .pointer(let base,_)=kind(type) else { throw glslError(t,.invalidType,"Expected pointer") }; return base
    }
    func rvalue(_ value:GLSLValue,at t:GLSLToken, allowCombinedSampler: Bool = false) throws -> GLSLValue {
        if value.combinedSampler != nil, !allowCombinedSampler {
            throw glslError(t, .unsupportedConstruct, "Combined sampler expressions are only valid as texture operation arguments")
        }
        if let swizzle = value.swizzle {
            let pointer = GLSLValue(expression: value.expression, type: swizzle.pointerType, pointer: true, writable: true)
            return try swizzleValue(rvalue(pointer, at: t), type: value.type, indices: swizzle.indices, at: t)
        }
        if !value.pointer { return value }
        if case .pointer(_, .storage(let access))=kind(value.type),!access.contains(.load) { throw glslError(t,.invalidAddressSpace,"Read from writeonly storage") }
        return try expressionNode(pointee(value.type,at:t),.load(pointer:value.expression),at:t)
    }
    private func storeSwizzle(_ base: IRHandle<IRExpression>, _ swizzle: GLSLWritableSwizzle,
                              value: GLSLValue, at token: GLSLToken) throws {
        guard case .pointer(_, let space) = kind(swizzle.pointerType),
              case .vector(_, let scalar) = kind(value.type) else {
            throw glslError(token, .invalidType, "Invalid writable swizzle type")
        }
        let component = try addType(.scalar(scalar))
        let pointerType = try addType(.pointer(base: component, space: space))
        // All RHS components come from the same value, even when the views overlap.
        let snapshot = try expressionNode(value.type, .runtimeValue(value.expression), at: token)
        for (lane, index) in swizzle.indices.enumerated() {
            let pointer = try expressionNode(pointerType, .accessIndex(base: base, index: index), at: token,
                                             pointer: true, writable: true)
            let laneValue = try expressionNode(component, .accessIndex(base: snapshot.expression, index: UInt32(lane)), at: token)
            try append(.store(pointer: pointer.expression, value: laneValue.expression), at: token)
        }
    }
    func requireBool(_ value:GLSLValue,at t:GLSLToken) throws {
        guard kind(value.type) == .scalar(.bool) else { throw glslError(t,.typeMismatch,"Expected scalar bool") }
    }
    func symbol(_ name:GLSLToken) throws -> GLSLValue {
        if name.barycentricPerspective != nil, !name.barycentricsEnabled {
            throw glslError(name, .unsupportedCapability, "Barycentric coordinates require GL_EXT_fragment_shader_barycentric")
        }
        if name.isViewIndex, !name.multiviewEnabled {
            throw glslError(name, .unsupportedCapability, "gl_ViewIndex requires GL_EXT_multiview")
        }
        let c=try requireContext()
        for scope in c.scopes.reversed() { try budget.consume(.work,amount:name.text.utf8.count); if let symbol=scope.symbols[name.text] { return symbol.value } }
        if let constant = constantNames[name.text] {
            return try expressionNode(constants[constant.index].type, .constant(constant), at: name, emit: false)
        }
        if let member = try uninstancedBlockMember(name) { return member }
        if globalNames[name.text]==nil { try builtin(name) }
        if let h=globalNames[name.text] {
            let global=globals[h.index]
            if global.space == .handle { return try expressionNode(global.type, .globalVariable(h), at: name, emit: false) }
            let pointer=try addType(.pointer(base:global.type,space:global.space))
            return try expressionNode(pointer,.globalVariable(h),at:name,emit:false,pointer:true,writable:isWritable(global) && !(interfaces.first(where:{$0.global==h})?.input ?? false))
        }
        throw glslError(name,.unresolvedName,"Unresolved GLSL name: \(name.text)")
    }

    func hasGlobalName(_ name: String) -> Bool {
        globalNames[name] != nil || functionNames[name] != nil || constantNames[name] != nil ||
            uninstancedBlockMembers[name] != nil
    }
    func builtin(_ name:GLSLToken) throws {
        let descriptor:(IRBuiltIn,String,Bool,IRStage)
        switch name.text {
        case "gl_Position":descriptor=(.position(invariant:false),"vec4",false,.vertex)
        case "gl_VertexID","gl_VertexIndex":descriptor=(.vertexIndex,"int",true,.vertex)
        case "gl_InstanceID","gl_InstanceIndex":descriptor=(.instanceIndex,"int",true,.vertex)
        case "gl_ViewIndex":
            guard stage == .vertex || stage == .fragment else {
                throw glslError(name, .invalidControlFlow, "gl_ViewIndex requires a vertex or fragment stage")
            }
            descriptor = (.viewIndex, "int", true, stage)
        case "gl_BaseInstance":
            guard version >= 460 else { throw glslError(name,.unsupportedConstruct,"gl_BaseInstance requires GLSL 4.60") }
            descriptor=(.baseInstance,"int",true,.vertex)
        case "gl_GlobalInvocationID":descriptor=(.globalInvocationId,"uvec3",true,.compute)
        case "gl_LocalInvocationID":descriptor=(.localInvocationId,"uvec3",true,.compute)
        case "gl_LocalInvocationIndex":descriptor=(.localInvocationIndex,"uint",true,.compute)
        case "gl_WorkGroupID":descriptor=(.workgroupId,"uvec3",true,.compute)
        case "gl_NumWorkGroups":descriptor=(.numWorkgroups,"uvec3",true,.compute)
        case "gl_FragCoord":descriptor=(.position(invariant:false),"vec4",true,.fragment)
        case "gl_FrontFacing":descriptor=(.frontFacing,"bool",true,.fragment)
        case "gl_FragDepth":descriptor=(.fragDepth,"float",false,.fragment)
        default:
            guard let perspective = name.barycentricPerspective else { return }
            descriptor = (.barycentric(perspective: perspective), "vec3", true, .fragment)
        }
        guard descriptor.3==stage else { throw glslError(name,.invalidControlFlow,"Builtin not valid for stage") }
        guard let type=try builtinType(descriptor.1) else { throw glslError(name,.invalidType,"Builtin type unavailable") }
        try registerInterface(name.text,type:type,binding:.builtIn(descriptor.0),input:descriptor.2)
    }
    func expression(_ min:Int=0) throws -> GLSLValue {
        try enter(); defer { depth-=1 }; let t=try advance(); var left:GLSLValue
        if t.text=="(" { left=try expression(); try expect(")") }
        else if t.text == "++" || t.text == "--" {
            let target = try expression(12)
            left = try increment(target, postfix: false, at: t)
        }
        else if ["-","+","!","~"].contains(t.text) {
            let v=try rvalue(expression(12),at:t)
            if t.text=="+" { guard scalar(v.type)?.kind != .bool else { throw glslError(t,.typeMismatch,"Invalid unary plus") }; left=v }
            else {
                let s=scalar(v.type); guard let s else { throw glslError(t,.typeMismatch,"Invalid unary operand") }
                if t.text=="!" { guard s.kind == .bool else { throw glslError(t,.typeMismatch,"Logical not requires bool") } }
                else if t.text=="~" { guard [.sint,.uint].contains(s.kind) else { throw glslError(t,.typeMismatch,"Bitwise not requires integer") } }
                else { guard [.sint,.float].contains(s.kind) else { throw glslError(t,.typeMismatch,"Negation requires signed number") } }
                left=try expressionNode(v.type,.unary(op:t.text=="!" ? .logicalNot:t.text=="~" ? .bitwiseNot:.negate,expression:v.expression),at:t)
            }
        } else if t.text=="true" || t.text=="false" { left=try expressionNode(addType(.scalar(.bool)),.literal(.bool(t.text=="true")),at:t,emit:false) }
        else if t.text.first?.isNumber==true || t.text.first=="." { left=try number(t) }
        else if glslIdentifier(t.text) {
            if try take("(") {
                left=try call(t,args:callArguments(at:t))
            } else { left=try symbol(t) }
        } else { throw glslError(t,.unexpectedToken,"Expected GLSL expression") }
        return try finishExpression(left, minimum: min)
    }

    private func finishExpression(_ initial: GLSLValue, minimum: Int) throws -> GLSLValue {
        var left = initial
        while text == "." || text == "[" || text == "++" || text == "--" {
            if text == "++" || text == "--" {
                let op = try advance()
                left = try increment(left, postfix: true, at: op)
            } else {
                left = try access(left)
            }
        }
        let precedence=["||":1,"&&":2,"|":3,"^":4,"&":5,"==":6,"!=":6,"<":7,">":7,"<=":7,">=":7,"<<":8,">>":8,"+":9,"-":9,"*":10,"/":10,"%":10]
        while let p=precedence[text],p>=minimum {
            let op=try advance()
            if op.text=="&&" || op.text=="||" { left=try shortCircuit(left,op:op,precedence:p);continue }
            let l=try rvalue(left,at:op); let r=try rvalue(expression(p+1),at:op); left=try binary(l,r,op:op)
        }
        return left
    }

    private func callArguments(at token: GLSLToken) throws -> [GLSLValue] {
        var arguments: [GLSLValue] = []
        if !(try take(")")) {
            let allowCombined = try acceptsGLSLCombinedSamplerArgument(token.text) &&
                functionNames[token.text] == nil && namedType(token.text) == nil
            repeat { arguments.append(try rvalue(expression(), at: token, allowCombinedSampler: allowCombined)) } while try take(",")
            try expect(")")
        }
        return arguments
    }
    func number(_ t:GLSLToken) throws -> GLSLValue {
        var raw=t.text; let unsigned=raw.hasSuffix("u") || raw.hasSuffix("U")
        if unsigned { raw.removeLast() }
        if !raw.hasPrefix("0x") && !raw.hasPrefix("0X") && (raw.contains(".") || raw.contains("e") || raw.contains("E")) {
            guard !unsigned else { throw glslError(t,.invalidToken,"Unsigned floating literal") }
            if raw.hasSuffix("hf") || raw.hasSuffix("HF") {
                guard t.float16Enabled else {
                    throw glslError(t, .unsupportedConstruct, "Half literal requires a GLSL float16 extension")
                }
                guard let value = Double(raw.dropLast(2)), value.isFinite else {
                    throw glslError(t, .invalidToken, "Invalid/out-of-range half literal")
                }
                let bits = Binary16.encode(value)
                guard bits & 0x7c00 != 0x7c00 else {
                    throw glslError(t, .invalidToken, "Invalid/out-of-range half literal")
                }
                return try expressionNode(addType(.scalar(.f16)), .literal(.f16Bits(bits)), at: t, emit: false)
            }
            if raw.hasSuffix("f") || raw.hasSuffix("F") { raw.removeLast() }
            guard let value=Float(raw),value.isFinite else { throw glslError(t,.invalidToken,"Invalid/out-of-range floating literal") }
            return try expressionNode(addType(.scalar(.f32)),.literal(.f32(value)),at:t,emit:false)
        }
        let radix:Int
        if raw.hasPrefix("0x") || raw.hasPrefix("0X") { radix=16; raw=String(raw.dropFirst(2)) }
        else if raw.count>1 && raw.hasPrefix("0") { radix=8 } else { radix=10 }
        if unsigned {
            guard let value=UInt32(raw,radix:radix) else { throw glslError(t,.invalidToken,"Invalid uint literal") }
            return try expressionNode(addType(.scalar(.u32)),.literal(.u32(value)),at:t,emit:false)
        }
        guard let value=Int32(raw,radix:radix) else { throw glslError(t,.invalidToken,"Invalid int literal") }
        return try expressionNode(addType(.scalar(.i32)),.literal(.i32(value)),at:t,emit:false)
    }
    func binary(_ lhs:GLSLValue,_ rhs:GLSLValue,op:GLSLToken) throws -> GLSLValue {
        if let matrix = try glslMatrixBinary(lhs, rhs, at: op) { return matrix }
        let map:[String:IRBinaryOperator] = ["+":.add,"-":.subtract,"*":.multiply,"/":.divide,"%":.modulo,"==":.equal,"!=":.notEqual,"<":.less,"<=":.lessEqual,">":.greater,">=":.greaterEqual,"&":.and,"|":.inclusiveOr,"^":.exclusiveOr,"<<":.shiftLeft,">>":.shiftRight]
        guard let operation=map[op.text] else { throw glslError(op,.unsupportedConstruct,"Binary operator unsupported") }
        guard let left = IRBitcastShape(kind(lhs.type)), let right = IRBitcastShape(kind(rhs.type)),
              left.size == nil || right.size == nil || left.size == right.size else {
            throw glslError(op, .typeMismatch, "Binary operand shapes differ")
        }
        if operation == .shiftLeft || operation == .shiftRight {
            guard [.i32, .u32].contains(left.scalar), [.i32, .u32].contains(right.scalar),
                  right.size == nil || right.size == left.size else {
                throw glslError(op, .typeMismatch, "Shift requires integer operands and a compatible count shape")
            }
            let count = try binaryOperand(rhs, scalar: .u32, size: left.size, at: op)
            return try expressionNode(lhs.type, .binary(op: operation, left: lhs.expression, right: count.expression), at: op)
        }
        let equality = operation == .equal || operation == .notEqual
        let ordering = [IRBinaryOperator.less, .lessEqual, .greater, .greaterEqual].contains(operation)
        let comparison = equality || ordering
        guard (!comparison || left.size == right.size), (!ordering || left.size == nil),
              let component = commonImplicitScalar(left.scalar, right.scalar),
              component != .bool || equality else {
            throw glslError(op, .typeMismatch, "Incompatible binary operands")
        }
        if [.and, .inclusiveOr, .exclusiveOr, .modulo].contains(operation),
           ![.i32, .u32].contains(component) {
            throw glslError(op, .typeMismatch, "Integer operands required")
        }
        let size = left.size ?? right.size
        let a = try binaryOperand(lhs, scalar: component, size: size, at: op)
        let b = try binaryOperand(rhs, scalar: component, size: size, at: op)
        let result = try comparison ? addType(IRBitcastShape(size: size, scalar: .bool).type) : a.type
        let value = try expressionNode(result, .binary(op: operation, left: a.expression, right: b.expression), at: op)
        if equality, size != nil {
            let reduction: IRRelationalFunction = operation == .equal ? .all : .any
            return try expressionNode(addType(.scalar(.bool)),
                                      .relational(function: reduction, argument: value.expression), at: op)
        }
        return value
    }

    private func binaryOperand(_ value: GLSLValue, scalar: IRScalar, size: IRVectorSize?,
                               at token: GLSLToken) throws -> GLSLValue {
        guard let sourceShape = IRBitcastShape(kind(value.type)),
              let converted = try implicitlyConvert(
                value, to: addType(IRBitcastShape(size: sourceShape.size, scalar: scalar).type), at: token)
        else {
            throw glslError(token, .typeMismatch, "Incompatible binary operands")
        }
        guard let size, sourceShape.size == nil else { return converted }
        return try expressionNode(addType(.vector(size: size, scalar: scalar)),
                                  .splat(size: size, value: converted.expression), at: token)
    }
    func call(_ name:GLSLToken,args:[GLSLValue]) throws -> GLSLValue {
        if let target=try resolveType(name) {
            guard !args.isEmpty else { throw glslError(name,.typeMismatch,"Constructor requires arguments") }
            if case .scalar(let s)=kind(target) {
                guard args.count==1,case .scalar=kind(args[0].type) else { throw glslError(name,.typeMismatch,"Scalar constructor arguments") }
                if target==args[0].type { return args[0] }
                return try expressionNode(target,.cast(expression:args[0].expression,target:s),at:name)
            }
            if case .vector(let size,let component)=kind(target) {
                if args.count == 1, case .vector(let width, let source) = kind(args[0].type),
                   width == size, source != component,
                   (source == .i32 || source == .u32) && (component == .i32 || component == .u32) {
                    return try expressionNode(target,.cast(expression:args[0].expression,target:component),at:name)
                }
                if args.count==1,case .scalar=kind(args[0].type) {
                    var value=args[0]
                    if scalar(value.type) != component { value=try expressionNode(addType(.scalar(component)),.cast(expression:value.expression,target:component),at:name) }
                    return try expressionNode(target,.splat(size:size,value:value.expression),at:name)
                }
                var count=0
                let components = component == .f32 || component == .f16
                    ? try args.map { try floatConstructorValue($0, to: component, at: name) } : args
                for a in components {
                    guard scalar(a.type)==component else { throw glslError(name,.typeMismatch,"Vector constructor scalar type mismatch") }
                    switch kind(a.type) { case .scalar:count+=1; case .vector(let n,_):count+=Int(n.rawValue); default:throw glslError(name,.typeMismatch,"Invalid vector component") }
                }
                guard count==Int(size.rawValue) else { throw glslError(name,.typeMismatch,"Vector component count") }
                return try expressionNode(target,.compose(type:target,components:components.map(\.expression)),at:name)
            }
            if case .matrix(let columns, let rows, let component) = kind(target) {
                return try matrixConstructor(target, columns: columns, rows: rows, component: component,
                                             arguments: args, at: name)
            }
            if case .structure(let members, _) = kind(target) {
                guard args.count == members.count, zip(args, members).allSatisfy({ $0.type == $1.type }) else {
                    throw glslError(name,.typeMismatch,"Structure constructor argument types")
                }
                return try expressionNode(target,.compose(type:target,components:args.map(\.expression)),at:name)
            }
        }
        if name.text=="main" { throw glslError(name,.invalidControlFlow,"main cannot be called from shader code") }
        if let overloads = functionNames[name.text] {
            guard let result = try userCall(overloads, arguments: args, allowingVoid: false, at: name) else {
                throw ShaderFailure.invalidHandle
            }
            return result
        }
        if let texture = try glslTextureCall(name, arguments: args) { return texture }
        let comparison: IRBinaryOperator? = switch name.text {
        case "lessThan": .less
        case "lessThanEqual": .lessEqual
        case "greaterThan": .greater
        case "greaterThanEqual": .greaterEqual
        case "equal": .equal
        case "notEqual": .notEqual
        default: nil
        }
        if let comparison { return try vectorComparison(comparison, arguments: args, at: name) }
        if ["all", "any", "not"].contains(name.text) {
            guard args.count == 1, case .vector(_, .bool) = kind(args[0].type) else {
                throw glslError(name, .typeMismatch, "Expected a single Boolean vector argument")
            }
            let value = args[0]
            if name.text == "not" {
                return try expressionNode(value.type, .unary(op: .logicalNot, expression: value.expression), at: name)
            }
            return try expressionNode(addType(.scalar(.bool)), .relational(function: name.text == "all" ? .all : .any,
                                                                           argument: value.expression), at: name)
        }
        if let math = try glslMathBuiltin(name, arguments: args) { return math }
        throw glslError(name,.unsupportedConstruct,"Unimplemented function/builtin: \(name.text)")
    }

    private func vectorComparison(
        _ operation: IRBinaryOperator, arguments: [GLSLValue], at token: GLSLToken
    ) throws -> GLSLValue {
        guard arguments.count == 2, case .vector(let size, let left) = kind(arguments[0].type),
              case .vector(let rightSize, let right) = kind(arguments[1].type), size == rightSize else {
            throw glslError(token, .typeMismatch, "Expected two vectors of the same width")
        }
        guard let scalar = commonImplicitScalar(left, right),
              scalar != .bool || operation == .equal || operation == .notEqual else {
            throw glslError(token, .typeMismatch, "Incompatible vector comparison component types")
        }
        let operands = try arguments.map { try binaryOperand($0, scalar: scalar, size: size, at: token).expression }
        let result = try addType(.vector(size: size, scalar: .bool))
        return try expressionNode(result, .binary(op: operation, left: operands[0], right: operands[1]), at: token)
    }

    private func matrixConstructor(
        _ target: IRHandle<IRType>, columns: IRVectorSize, rows: IRVectorSize, component: IRScalar,
        arguments: [GLSLValue], at token: GLSLToken
    ) throws -> GLSLValue {
        if arguments.count == 1, arguments[0].type == target { return arguments[0] }
        let scalar = try addType(.scalar(component))
        let columnType = try addType(.vector(size: rows, scalar: component))
        let zeroLiteral: IRLiteral = component == .f16 ? .f16Bits(0) : .f32(0)
        let oneLiteral: IRLiteral = component == .f16 ? .f16Bits(0x3c00) : .f32(1)
        let width = Int(columns.rawValue)
        let height = Int(rows.rawValue)
        let required = width * height
        var components: [IRHandle<IRExpression>] = []
        components.reserveCapacity(required)
        if arguments.count == 1, case .scalar = kind(arguments[0].type) {
            let diagonal = try floatConstructorValue(arguments[0], to: component, at: token).expression
            let zero = try expressionNode(scalar, .literal(zeroLiteral), at: token, emit: false).expression
            for column in 0..<width {
                for row in 0..<height { components.append(column == row ? diagonal : zero) }
            }
        } else if arguments.count == 1,
                  case .matrix(let sourceWidth, let sourceHeight, let sourceScalar) = kind(arguments[0].type) {
            let source = arguments[0].expression
            let sourceColumnType = try addType(.vector(size: sourceHeight, scalar: sourceScalar))
            let sourceScalarType = try addType(.scalar(sourceScalar))
            let zero = try expressionNode(scalar, .literal(zeroLiteral), at: token, emit: false).expression
            let one = try expressionNode(scalar, .literal(oneLiteral), at: token, emit: false).expression
            for column in 0..<width {
                let existing = column < Int(sourceWidth.rawValue)
                    ? try expressionNode(sourceColumnType, .accessIndex(base: source, index: UInt32(column)), at: token)
                    : nil
                for row in 0..<height {
                    if let existing, row < Int(sourceHeight.rawValue) {
                        let value = try expressionNode(sourceScalarType,
                                                       .accessIndex(base: existing.expression, index: UInt32(row)), at: token)
                        components.append(try floatConstructorValue(value, to: component, at: token).expression)
                    } else {
                        components.append(column == row ? one : zero)
                    }
                }
            }
        } else {
            for argument in arguments {
                try budget.consume(.work)
                guard components.count < required else {
                    throw glslError(token, .typeMismatch, "Too many matrix constructor arguments")
                }
                let value = try floatConstructorValue(argument, to: component, at: token)
                if case .vector(let size, _) = kind(value.type) {
                    let count = min(Int(size.rawValue), required - components.count)
                    for lane in 0..<count {
                        let component = try expressionNode(
                            scalar, .accessIndex(base: value.expression, index: UInt32(lane)), at: token)
                        components.append(component.expression)
                    }
                } else {
                    components.append(value.expression)
                }
            }
            guard components.count == required else {
                throw glslError(token, .typeMismatch, "Not enough matrix constructor components")
            }
        }
        // GLSL consumes components in column-major order even when an argument
        // vector crosses a column boundary.
        let values = try stride(from: 0, to: required, by: height).map { start in
            try expressionNode(columnType, .compose(type: columnType, components: Array(components[start..<(start + height)])),
                               at: token).expression
        }
        return try expressionNode(target, .compose(type: target, components: values), at: token)
    }

    private func floatConstructorValue(_ value: GLSLValue, to scalar: IRScalar, at token: GLSLToken) throws -> GLSLValue {
        guard let shape = IRBitcastShape(kind(value.type)) else {
            throw glslError(token, .typeMismatch, "Expected a scalar or vector constructor component")
        }
        if shape.scalar == scalar { return value }
        let converted = IRBitcastShape(size: shape.size, scalar: scalar).type
        return try expressionNode(addType(converted), .cast(expression: value.expression, target: scalar), at: token)
    }

    private func userCall(
        _ overloads: [IRHandle<IRFunction>], arguments: [GLSLValue], allowingVoid: Bool, at name: GLSLToken
    ) throws -> GLSLValue? {
        let handle = try resolveFunction(overloads, argumentTypes: arguments.map(\.type), at: name)
        let function = functions[handle.index]
        if function.result == nil && !allowingVoid { throw glslError(name, .typeMismatch, "Void function used in an expression") }
        let operands = try zip(arguments, function.arguments).map { value, parameter in
            try budget.consume(.work)
            guard let converted = try implicitlyConvert(value, to: parameter.type, at: name) else {
                throw ShaderFailure.invalidHandle
            }
            return converted.expression
        }
        let result = try function.result.map { try expressionNode($0.type, .callResult(handle), at: name, emit: false) }
        try append(.call(function: handle, arguments: operands, result: result?.expression), at: name)
        return result
    }

    private enum ImplicitConversion: Equatable {
        case exact, floatToDouble, intToFloat, intToDouble, other

        func better(than other: Self) -> Bool {
            guard self != other else { return false }
            switch (self, other) {
            case (.exact, _): return true
            case (_, .exact): return false
            case (.floatToDouble, _), (.intToFloat, .intToDouble): return true
            default: return false
            }
        }
    }

    private func implicitConversion(from source: IRHandle<IRType>, to target: IRHandle<IRType>) -> ImplicitConversion? {
        if source == target { return .exact }
        guard let input = IRBitcastShape(kind(source)), let output = IRBitcastShape(kind(target)),
              input.size == output.size else { return nil }
        return implicitScalarConversion(from: input.scalar, to: output.scalar)
    }

    private func implicitScalarConversion(from source: IRScalar, to target: IRScalar) -> ImplicitConversion? {
        if source == target { return .exact }
        let double = IRScalar(kind: .float, width: 8)
        switch (source, target) {
        case (.i32, .u32): return .other
        case (.i32, .f32), (.u32, .f32): return .intToFloat
        case (.i32, double), (.u32, double): return .intToDouble
        case (.f32, double): return .floatToDouble
        default: return nil
        }
    }

    func implicitlyConvert(_ value: GLSLValue, to target: IRHandle<IRType>, at token: GLSLToken) throws -> GLSLValue? {
        guard let conversion = implicitConversion(from: value.type, to: target) else { return nil }
        if conversion == .exact { return value }
        guard let component = scalar(target) else { throw ShaderFailure.invalidHandle }
        return try expressionNode(target, .cast(expression: value.expression, target: component), at: token)
    }

    func commonImplicitScalar(_ lhs: IRScalar, _ rhs: IRScalar) -> IRScalar? {
        if implicitScalarConversion(from: lhs, to: rhs) != nil { return rhs }
        if implicitScalarConversion(from: rhs, to: lhs) != nil { return lhs }
        return nil
    }

    private func resolveFunction(
        _ overloads: [IRHandle<IRFunction>], argumentTypes: [IRHandle<IRType>], at name: GLSLToken
    ) throws -> IRHandle<IRFunction> {
        var viable: [(function: IRHandle<IRFunction>, conversions: [ImplicitConversion])] = []
        overload: for handle in overloads {
            try budget.consume(.work)
            let parameters = functions[handle.index].arguments
            guard parameters.count == argumentTypes.count else { continue }
            var conversions: [ImplicitConversion] = []
            for (argument, parameter) in zip(argumentTypes, parameters) {
                try budget.consume(.work)
                guard let conversion = implicitConversion(from: argument, to: parameter.type) else { continue overload }
                conversions.append(conversion)
            }
            if conversions.allSatisfy({ $0 == .exact }) { return handle }
            viable.append((handle, conversions))
        }
        guard var best = viable.first else {
            throw glslError(name, .typeMismatch, "No matching overload for '\(name.text)'")
        }
        func better(_ left: [ImplicitConversion], than right: [ImplicitConversion]) throws -> Bool {
            var wins = false
            for (a, b) in zip(left, right) {
                try budget.consume(.work)
                if b.better(than: a) { return false }
                if a.better(than: b) { wins = true }
            }
            return wins
        }
        for candidate in viable.dropFirst() where try better(candidate.conversions, than: best.conversions) {
            best = candidate
        }
        // A best match must beat every alternative without a worse conversion on another argument.
        for candidate in viable where candidate.function != best.function {
            guard try better(best.conversions, than: candidate.conversions) else {
                throw glslError(name, .typeMismatch, "Ambiguous overload for '\(name.text)'")
            }
        }
        return best.function
    }

    func wrap(_ main:IRHandle<IRFunction>) throws -> IRHandle<IRFunction> {
        let c=GLSLFunctionContext(result:nil); context=c; defer { context=nil }
        let t=GLSLToken(text:"main",start:0,end:0); var args:[IRFunctionArgument]=[]
        var outputs:[GLSLInterface]=[]
        for io in interfaces {
            let pt=try addType(.pointer(base:io.type,space:.private))
            let pointer=try expressionNode(pt,.globalVariable(io.global),at:t,emit:false,pointer:true,writable:true)
            if io.input {
                let signedIndex = io.binding == .builtIn(.vertexIndex) || io.binding == .builtIn(.instanceIndex)
                    || io.binding == .builtIn(.baseInstance) || io.binding == .builtIn(.viewIndex)
                let argumentType = signedIndex ? try addType(.scalar(.u32)) : io.type
                let argument = try expressionNode(argumentType,.functionArgument(UInt32(args.count)),at:t,emit:false)
                let value = signedIndex ? try expressionNode(
                    io.type,.bitcast(expression:argument.expression,type:io.type),at:t) : argument
                args.append(.init(name:io.name,type:argumentType,binding:io.binding))
                try append(.store(pointer:pointer.expression,value:value.expression),at:t)
            } else { outputs.append(io) }
        }
        try append(.call(function:main,arguments:[],result:nil),at:t)
        let result = try wrapOutputs(outputs, at: t)
        let body=try block(c.current); try budget.consume(.nodes); let h=try IRHandle<IRFunction>(index:functions.count)
        functions.append(.init(name:"_saga_glsl_entry",arguments:args,result:result,expressions:.init(c.expressions),statements:.init(c.statements),blocks:.init(c.blocks),body:body)); return h
    }

    private func wrapOutputs(_ outputs: [GLSLInterface], at token: GLSLToken) throws -> IRFunctionResult? {
        let values = try outputs.map { io in
            let pointerType = try addType(.pointer(base: io.type, space: .private))
            let pointer = try expressionNode(pointerType, .globalVariable(io.global), at: token, emit: false, pointer: true)
            return try rvalue(pointer, at: token).expression
        }
        guard let first = outputs.first else {
            try append(.returnValue(nil), at: token)
            return nil
        }
        if values.count == 1 {
            try append(.returnValue(values[0]), at: token)
            return .init(type: first.type, binding: first.binding)
        }
        var members: [IRStructMember] = []
        var offset: UInt32 = 0
        var alignment: UInt32 = 1
        for output in outputs {
            try budget.consume(.nodes)
            let layout = try typeLayout(output.type)
            offset = try aligned(offset, to: layout.align)
            members.append(.init(name: output.name, type: output.type, binding: output.binding, offset: offset))
            let (end, overflow) = offset.addingReportingOverflow(layout.size)
            guard !overflow else { throw glslError(token,.invalidType,"Stage output layout overflow") }
            offset = end
            alignment = max(alignment, layout.align)
        }
        let type = try addType(.structure(members: members, span: aligned(offset, to: alignment)))
        let result = try expressionNode(type, .compose(type: type, components: values), at: token)
        try append(.returnValue(result.expression), at: token)
        return .init(type: type)
    }
}
