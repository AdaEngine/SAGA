import SAGAShaderIR

extension GLSLParser {
    func isWritable(_ global:IRGlobalVariable) -> Bool {
        switch global.space { case .uniform:return false; case .storage(let access):return access.contains(.store); default:return true }
    }
    /// std430 scalar/vector/array/structure layout. std140 is rejected, never silently replaced.
    func typeLayout(_ type:IRHandle<IRType>) throws -> (align:UInt32,size:UInt32) {
        try enter(); defer { depth-=1 }
        switch kind(type) {
        case .scalar(let s): return (UInt32(s.width),UInt32(s.width))
        case .vector(let n,let s):return (UInt32(n == .two ? 2:4)*UInt32(s.width),UInt32(n.rawValue)*UInt32(s.width))
        case .array(let base,let size,let stride):
            let align=try typeLayout(base).align
            switch size {
            case .constant(let n): let (bytes,overflow)=stride.multipliedReportingOverflow(by:n); guard !overflow,bytes<=UInt32(Int32.max) else { throw glslError(token,.invalidType,"Array size overflow") }; return (align,bytes)
            case .dynamic:return (align,0)
            default:throw glslError(token,.unsupportedConstruct,"Override array size unsupported")
            }
        case .structure(let members,let span):
            var alignment:UInt32=1
            for member in members { alignment=max(alignment,try typeLayout(member.type).align) }
            return (alignment,span)
        default:throw glslError(token,.unsupportedConstruct,"Layout not implemented for this type")
        }
    }
    func aligned(_ value:UInt32,to alignment:UInt32) throws -> UInt32 {
        guard alignment>0,value<=UInt32(Int32.max)-alignment else { throw glslError(token,.invalidType,"Layout overflow") }
        return ((value+alignment-1)/alignment)*alignment
    }
    func arraySuffix(_ base:IRHandle<IRType>,runtime:Bool=false) throws -> IRHandle<IRType> {
        guard try take("[") else { return base }
        let size:IRArraySize
        if try take("]") { guard runtime else { throw glslError(token,.invalidType,"Runtime array only in storage tail") }; size = .dynamic }
        else {
            let t=try advance(); guard let n=UInt32(t.text),n>0 else { throw glslError(t,.unsupportedConstruct,"Only positive literal array sizes implemented") }; size = .constant(n); try expect("]")
        }
        if text=="[" { throw glslError(token,.unsupportedConstruct,"Multidimensional arrays not implemented") }
        let layout=try typeLayout(base); let stride=try aligned(layout.size,to:layout.align)
        let type=try addType(.array(base:base,size:size,stride:stride)); _ = try typeLayout(type); return type
    }
    func members(runtime:Bool, explicitOffsets:Bool = false) throws -> IRHandle<IRType> {
        try expect("{"); var fields:[IRStructMember]=[]; var names=Set<String>(); var offset:UInt32=0; var alignment:UInt32=1; var dynamic=false
        while text != "}" {
            try budget.consume(.work)
            guard !text.isEmpty else { throw glslError(token,.unexpectedToken,"Unclosed structure") }
            if dynamic { throw glslError(token,.invalidType,"Runtime array must be last") }
            var explicitOffset: UInt32?
            if try take("layout") {
                guard explicitOffsets else { throw glslError(token,.invalidType,"Member offset requires a buffer block") }
                try expect("(")
                try expect("offset")
                try expect("=")
                let value = try advance()
                guard let parsed = UInt32(value.text) else {
                    throw glslError(value,.invalidType,"Member offset requires a literal uint")
                }
                explicitOffset = parsed
                try expect(")")
            }
            guard let base=try parseType() else { throw glslError(token,.invalidType,"Void member") }
            let name=try identifier(); guard names.insert(name.text).inserted else { throw glslError(name,.duplicateName,"Duplicate member") }
            let type=try arraySuffix(base,runtime:runtime)
            let layout=try typeLayout(type); alignment=max(alignment,layout.align)
            if let explicitOffset {
                guard explicitOffset >= offset, explicitOffset % layout.align == 0 else {
                    throw glslError(name,.invalidType,"Overlapping or misaligned member offset")
                }
                offset = explicitOffset
            } else {
                offset = try aligned(offset,to:layout.align)
            }
            try budget.consume(.nodes)
            fields.append(.init(name:name.text,type:type,offset:offset,location:try name.location))
            let (end,overflow)=offset.addingReportingOverflow(layout.size); guard !overflow,end<=UInt32(Int32.max) else { throw glslError(name,.invalidType,"Member layout overflow") }; offset=end
            if case .array(_, .dynamic, _)=kind(type) { dynamic=true }
            try expect(";")
        }
        try expect("}")
        guard !fields.isEmpty else { throw glslError(token,.invalidType,"Empty structure") }
        // A runtime tail contributes one stride to the nominal structure span.
        if dynamic,let last=fields.last,case .array(_, .dynamic, let stride)=kind(last.type) {
            let (end,overflow)=offset.addingReportingOverflow(stride); guard !overflow,end<=UInt32(Int32.max) else { throw glslError(token,.invalidType,"Runtime stride overflow") }; offset=end
        }
        return try addType(.structure(members:fields,span:aligned(offset,to:alignment)))
    }
    func externalDeclaration() throws {
        var layout:[String:UInt32]=[:]; var flags=Set<String>(); var qualifier=""
        var memory = Set<String>()
        func memoryQualifiers() throws {
            while ["readonly", "writeonly", "coherent"].contains(text) {
                let qualifier = try advance()
                guard memory.insert(qualifier.text).inserted else {
                    throw glslError(qualifier,.duplicateName,"Duplicate memory qualifier")
                }
            }
        }
        if try take("layout") {
            try expect("(")
            repeat {
                let key=try identifier(); guard layout[key.text]==nil,!flags.contains(key.text) else { throw glslError(key,.duplicateName,"Duplicate layout key") }
                if try take("=") { let t=try advance(); guard let n=UInt32(t.text) else { throw glslError(t,.unsupportedConstruct,"Layout requires literal uint") }; layout[key.text]=n }
                else { flags.insert(key.text) }
            } while try take(",")
            try expect(")")
        }
        try memoryQualifiers()
        let sampling: IRSampling?
        if try take("centroid") { sampling = .centroid }
        else if try take("sample") { sampling = .sample }
        else { sampling = nil }
        if ["in","out","buffer","uniform","shared"].contains(text) { qualifier=try advance().text }
        let varying = (stage == .vertex && qualifier == "out") || (stage == .fragment && qualifier == "in")
        guard sampling == nil || varying else {
            throw glslError(token,.invalidBinding,"Interpolation sampling requires an interstage varying")
        }
        try memoryQualifiers()
        if qualifier=="in",text==";" {
            guard stage == .compute,!hasLocalSize,flags.isEmpty,!layout.isEmpty,layout.keys.allSatisfy({["local_size_x","local_size_y","local_size_z"].contains($0)}),layout.values.allSatisfy({$0>0}),memory.isEmpty else { throw glslError(token,.unsupportedConstruct,"Invalid compute workgroup declaration") }
            workgroup = .init(x:layout["local_size_x"] ?? 1,y:layout["local_size_y"] ?? 1,z:layout["local_size_z"] ?? 1);hasLocalSize=true;try expect(";");return
        }
        if qualifier=="buffer" || qualifier=="uniform" {
            let name=try identifier()
            let uniform = qualifier == "uniform"
            let readOnly = memory.contains("readonly")
            let writeOnly = memory.contains("writeonly")
            guard flags == (uniform ? ["std140"] : ["std430"]),
                  layout.keys.allSatisfy({["set","binding"].contains($0)}), let binding=layout["binding"],
                  !(readOnly && writeOnly), !uniform || memory.isEmpty else {
                throw glslError(name,.unsupportedConstruct,"Expected a std140 uniform or std430 storage block with explicit binding")
            }
            let type=try members(runtime:!uniform,explicitOffsets:true); let instance=try identifier();try expect(";")
            guard globalNames[instance.text]==nil,functionNames[instance.text]==nil,!instance.text.hasPrefix("gl_") else { throw glslError(instance,.duplicateName,"Duplicate storage instance") }
            let access:IRStorageAccess=readOnly ? [.load]:writeOnly ? [.store]:[.load,.store]
            let rb=IRResourceBinding(group:layout["set"] ?? 0,binding:binding)
            guard !globals.contains(where:{$0.binding==rb}) else { throw glslError(instance,.invalidBinding,"Duplicate resource binding") }
            try budget.consume(.nodes);let h=try IRHandle<IRGlobalVariable>(index:globals.count)
            let decorations: IRMemoryDecorations = memory.contains("coherent") ? .coherent : []
            globals.append(.init(name:instance.text,space:uniform ? .uniform : .storage(access:access),binding:rb,type:type,
                                 memoryDecorations:decorations,location:try instance.location));globalNames[instance.text]=h;return
        }
        if try take("struct") {
            guard qualifier.isEmpty,layout.isEmpty,flags.isEmpty,memory.isEmpty else { throw glslError(token,.unsupportedConstruct,"Qualified structure unsupported") }
            let name=try identifier();let type=try members(runtime:false);try expect(";")
            guard namedTypes[name.text]==nil else { throw glslError(name,.duplicateName,"Duplicate type") };namedTypes[name.text]=type;return
        }
        guard flags.isEmpty,memory.isEmpty else { throw glslError(token,.unsupportedConstruct,"Unsupported memory/layout qualifier") }
        let type=try parseType();let name=try identifier()
        if qualifier.isEmpty,text=="(" { guard layout.isEmpty else { throw glslError(name,.unsupportedConstruct,"Function layout") }; try function(name,result:type);return }
        guard let base=type else { throw glslError(name,.invalidType,"Void variable") }
        let finalType=try arraySuffix(base)
        guard globalNames[name.text]==nil,functionNames[name.text]==nil,!name.text.hasPrefix("gl_") else { throw glslError(name,.duplicateName,"Duplicate/reserved global") }
        try expect(";")
        if qualifier=="in" || qualifier=="out" {
            let vertexInput = stage == .vertex && qualifier == "in"
            guard stage != .compute,layout.count==1,let loc=layout["location"],let lane = scalar(finalType),
                  lane.kind == .float || (vertexInput && (lane.kind == .uint || lane.kind == .sint)) else {
                throw glslError(name,.unsupportedConstruct,"Unsupported explicit location interface type")
            }
            let binding=IRBinding.location(index:loc,interpolation:varying ? .perspective : nil,
                                           sampling:varying ? (sampling ?? .center) : nil,
                                           blendSource:nil,perPrimitive:false)
            guard !interfaces.contains(where: { io in
                guard io.input == (qualifier == "in"),
                      case .location(let previous, _, _, _, _) = io.binding else { return false }
                return loc == previous
            }) else { throw glslError(name,.invalidBinding,"Duplicate stage location") }
            try registerInterface(name.text,type:finalType,binding:binding,input:qualifier=="in")
        } else {
            guard layout.isEmpty,qualifier.isEmpty || qualifier=="shared" else { throw glslError(name,.unsupportedConstruct,"Unsupported global qualifier") }
            if qualifier=="shared",stage != .compute { throw glslError(name,.invalidAddressSpace,"Shared variable outside compute") }
            try budget.consume(.nodes);let h=try IRHandle<IRGlobalVariable>(index:globals.count)
            globals.append(.init(name:name.text,space:qualifier=="shared" ? .workgroup:.private,type:finalType,location:try name.location));globalNames[name.text]=h
        }
    }
    func forLoop(at t:GLSLToken) throws {
        let c=try requireContext();c.scopes.append([:]);defer { c.scopes.removeLast() };try expect("(")
        // Declaration or ordinary assignment in initializer; both lower before loop.
        if !(try take(";")) {
            if try builtinType(text) != nil { _ = try statement() } else { try simpleStatement(end:";") }
        }
        let outer=c.current;c.current=[]
        let condition:GLSLValue
        if text==";" { condition=try expressionNode(addType(.scalar(.bool)),.literal(.bool(true)),at:t,emit:false) }
        else { condition=try rvalue(expression(),at:t);try requireBool(condition,at:t) }
        try expect(";");let checks=c.current;c.current=[]
        if !(try take(")")) { try simpleStatement(end:")") }
        let continuing=try block(c.current);c.current=outer
        c.loopDepth+=1;let (body,_)=try childStatement();c.loopDepth-=1
        let stop=try block([try statementNode(.break,at:t)]);let empty=try block([])
        let branch=try statementNode(.ifCondition(condition:condition.expression,accept:empty,reject:stop),at:t)
        let child=try statementNode(.block(body),at:t)
            let loopBody=try block(checks+[branch,child])
        try append(.loop(body:loopBody,continuing:continuing,breakIf:nil),at:t)
    }
    func access(_ base:GLSLValue) throws -> GLSLValue {
        let op=try advance();let type=base.pointer ? try pointee(base.type,at:op):base.type
        let memberType:IRHandle<IRType>;let operation:IRExpressionKind
        if op.text=="." {
            let name=try identifier()
            if name.text == "length", try take("(") {
                try expect(")")
                let int = try addType(.scalar(.i32))
                switch kind(type) {
                case .array(_, .constant(let count), _):
                    guard let length = Int32(exactly: count) else {
                        throw glslError(name,.invalidType,"Array length exceeds int")
                    }
                    return try expressionNode(int,.literal(.i32(length)),at:name,emit:false)
                case .array(_, .dynamic, _) where base.pointer:
                    let uint = try addType(.scalar(.u32))
                    let length = try expressionNode(uint,.arrayLength(base.expression),at:name)
                    return try expressionNode(int,.cast(expression:length.expression,target:.i32),at:name)
                default:
                    throw glslError(name,.typeMismatch,"length() requires an array")
                }
            }
            if case .structure(let members,_)=kind(type) {
                guard let n=members.firstIndex(where:{$0.name==name.text}) else { throw glslError(name,.unresolvedName,"Unknown structure member") }
                memberType=members[n].type;operation = .accessIndex(base:base.expression,index:UInt32(n))
            } else {
                guard case .vector(let size,let s)=kind(type) else { throw glslError(name,.typeMismatch,"Field access requires vector/structure") }
                let groups=[Array("xyzw"),Array("rgba"),Array("stpq")];let chars=Array(name.text)
                guard !chars.isEmpty,chars.count<=4,let group=groups.first(where:{chars.allSatisfy($0.contains)}) else { throw glslError(name,.invalidType,"Invalid swizzle") }
                let indices=chars.compactMap {group.firstIndex(of:$0)}
                guard indices.allSatisfy({$0<Int(size.rawValue)}) else { throw glslError(name,.invalidType,"Swizzle out of range") }
                if indices.count==1 {
                    memberType=try addType(.scalar(s))
                    // Preserve lvalue access without emitting a load of the original vector.
                    operation = .accessIndex(base:base.expression,index:UInt32(indices[0]))
                } else {
                    let components:[IRSwizzleComponent]=[.x,.y,.z,.w];let sw=indices.map {components[$0]};let padded=sw+Array(repeating:.x,count:4-sw.count)
                    guard let n=IRVectorSize(rawValue:UInt8(sw.count)) else { throw glslError(name,.invalidType,"Swizzle size") }
                    let v=try rvalue(base,at:op)
                    return try expressionNode(addType(.vector(size:n,scalar:s)),.swizzle(size:n,vector:v.expression,pattern:.init(x:padded[0],y:padded[1],z:padded[2],w:padded[3])),at:name)
                }
            }
        } else {
            let index=try rvalue(expression(),at:op);try expect("]")
            guard case .scalar(let s)=kind(index.type),[.sint,.uint].contains(s.kind) else { throw glslError(op,.typeMismatch,"Index requires scalar integer") }
            switch kind(type) {
            case .array(let child,_,_):memberType=child
            case .vector(_,let s):memberType=try addType(.scalar(s))
            default:throw glslError(op,.typeMismatch,"Type not indexable")
            }
            operation = .access(base:base.expression,index:index.expression)
        }
        if base.pointer {
            guard case .pointer(_,let space)=kind(base.type) else { throw glslError(op,.invalidType,"Pointer expected") }
            return try expressionNode(addType(.pointer(base:memberType,space:space)),operation,at:op,pointer:true,writable:base.writable)
        }
        return try expressionNode(memberType,operation,at:op)
    }
}
