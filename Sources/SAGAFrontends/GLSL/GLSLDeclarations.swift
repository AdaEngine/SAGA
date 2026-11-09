import SAGAShaderIR

enum GLSLBlockLayout {
    case std140
    case std430

    func aggregateAlignment(_ alignment: UInt32) -> UInt32 {
        self == .std140 ? max(16, alignment) : alignment
    }
}

extension GLSLParser {
    func isWritable(_ global:IRGlobalVariable) -> Bool {
        switch global.space { case .uniform, .immediate:return false; case .storage(let access):return access.contains(.store); default:return true }
    }
    func uninstancedBlockMember(_ token: GLSLToken) throws -> GLSLValue? {
        guard let member = uninstancedBlockMembers[token.text] else { return nil }
        let global = globals[member.block.index]
        let writable = isWritable(global)
        let root = try expressionNode(addType(.pointer(base: global.type, space: global.space)),
                                      .globalVariable(member.block), at: token, emit: false, pointer: true, writable: writable)
        return try expressionNode(addType(.pointer(base: member.type, space: global.space)),
                                  .accessIndex(base: root.expression, index: member.index), at: token,
                                  pointer: true, writable: writable)
    }

    func typeLayout(_ type:IRHandle<IRType>, standard: GLSLBlockLayout = .std430) throws -> (align:UInt32,size:UInt32) {
        try enter(); defer { depth-=1 }
        switch kind(type) {
        case .scalar(let s): return (UInt32(s.width),UInt32(s.width))
        case .vector(let n,let s):return (UInt32(n == .two ? 2:4)*UInt32(s.width),UInt32(n.rawValue)*UInt32(s.width))
        case .matrix(let columns, let rows, let scalar):
            let stride = UInt32(rows == .two ? 2 : 4) * UInt32(scalar.width)
            guard standard == .std430 || stride >= 16 else {
                throw glslError(token, .unsupportedConstruct, "Padded std140 matrix column stride is not representable")
            }
            return (stride, stride * UInt32(columns.rawValue))
        case .array(let base,let size,let stride):
            let align = try standard.aggregateAlignment(typeLayout(base, standard: standard).align)
            switch size {
            case .constant(let n): let (bytes,overflow)=stride.multipliedReportingOverflow(by:n); guard !overflow,bytes<=UInt32(Int32.max) else { throw glslError(token,.invalidType,"Array size overflow") }; return (align,bytes)
            case .dynamic:return (align,0)
            default:throw glslError(token,.unsupportedConstruct,"Override array size unsupported")
            }
        case .structure(let members,let span):
            var alignment:UInt32=1
            for member in members { alignment=max(alignment,try typeLayout(member.type, standard: standard).align) }
            return (standard.aggregateAlignment(alignment),span)
        default:throw glslError(token,.unsupportedConstruct,"Layout not implemented for this type")
        }
    }
    func aligned(_ value:UInt32,to alignment:UInt32) throws -> UInt32 {
        guard alignment>0,value<=UInt32(Int32.max)-alignment else { throw glslError(token,.invalidType,"Layout overflow") }
        return ((value+alignment-1)/alignment)*alignment
    }
    func arraySuffix(
        _ base:IRHandle<IRType>,runtime:Bool=false, standard: GLSLBlockLayout = .std430, globalLength: Bool = false
    ) throws -> IRHandle<IRType> {
        guard try take("[") else { return base }
        let size:IRArraySize
        if try take("]") { guard runtime else { throw glslError(token,.invalidType,"Runtime array only in storage tail") }; size = .dynamic }
        else {
            let length = try advance()
            let count: UInt32?
            if let literal = UInt32(length.text) {
                count = literal
            } else if context == nil || globalLength {
                // In a global declaration no local can shadow the constant.
                switch constantNames[length.text].flatMap(globalConstantLiteral) {
                case .i32(let value): count = UInt32(exactly: value)
                case .u32(let value): count = value
                default: count = nil
                }
            } else {
                count = nil
            }
            guard let count, count > 0 else {
                throw glslError(length, .unsupportedConstruct, "Expected a positive fixed array length")
            }
            size = .constant(count)
            try expect("]")
        }
        if text=="[" { throw glslError(token,.unsupportedConstruct,"Multidimensional arrays not implemented") }
        return try layoutArray(base, size: size, standard: standard)
    }

    private func layoutArray(
        _ base: IRHandle<IRType>, size: IRArraySize, standard: GLSLBlockLayout
    ) throws -> IRHandle<IRType> {
        let layout = try typeLayout(base, standard: standard)
        let stride = try aligned(layout.size, to: standard.aggregateAlignment(layout.align))
        let type = try addType(.array(base: base, size: size, stride: stride))
        _ = try typeLayout(type, standard: standard)
        return type
    }

    private func std140MemberType(_ type: IRHandle<IRType>) throws -> IRHandle<IRType> {
        try enter(); defer { depth -= 1 }
        switch kind(type) {
        case .array(let base, let size, _):
            return try layoutArray(std140MemberType(base), size: size, standard: .std140)
        case .structure(let members, _):
            var fields: [IRStructMember] = []
            var offset: UInt32 = 0
            var alignment: UInt32 = 16
            for member in members {
                try budget.consume(.work)
                let memberType = try std140MemberType(member.type)
                let layout = try typeLayout(memberType, standard: .std140)
                alignment = max(alignment, layout.align)
                offset = try aligned(offset, to: layout.align)
                fields.append(.init(name: member.name, type: memberType, binding: member.binding,
                                    offset: offset, location: member.location))
                let (end, overflow) = offset.addingReportingOverflow(layout.size)
                guard !overflow, end <= UInt32(Int32.max) else {
                    throw glslError(token, .invalidType, "Member layout overflow")
                }
                offset = end
            }
            return try addType(.structure(members: fields, span: aligned(offset, to: alignment)))
        default:
            return type
        }
    }

    func members(
        runtime:Bool, explicitOffsets:Bool = false, standard: GLSLBlockLayout = .std430
    ) throws -> IRHandle<IRType> {
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
            guard let parsed=try parseType() else { throw glslError(token,.invalidType,"Void member") }
            let base = try standard == .std140 ? std140MemberType(parsed) : parsed
            let prefix = try explicitOffsets ? arraySuffix(base, runtime: runtime, standard: standard) : base
            let name=try identifier(); guard names.insert(name.text).inserted else { throw glslError(name,.duplicateName,"Duplicate member") }
            guard prefix == base || text != "[" else {
                throw glslError(token, .unsupportedConstruct, "Multidimensional arrays not implemented")
            }
            let type = try arraySuffix(prefix, runtime: runtime, standard: standard)
            let layout = try typeLayout(type, standard: standard); alignment=max(alignment,layout.align)
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
        return try addType(.structure(members:fields,span:aligned(offset,to:standard.aggregateAlignment(alignment))))
    }
    func defaultPrecisionDeclaration() throws {
        let qualifier = try advance()
        guard ["lowp", "mediump", "highp"].contains(qualifier.text) else {
            throw glslError(qualifier, .unexpectedToken, "Expected a GLSL precision qualifier")
        }
        let type = try advance()
        guard type.text == "float" || type.text == "int" else {
            throw glslError(type, .invalidType, "Default numeric precision requires float or int")
        }
        try expect(";")
        // Full precision satisfies the precision guarantees of these desktop GLSL qualifiers.
    }

    func externalDeclaration() throws {
        if try take("precision") { try defaultPrecisionDeclaration(); return }
        var layout:[String:UInt32]=[:]; var flags=Set<String>(); var qualifier=""
        var memory = Set<String>()
        var interpolation: IRInterpolation?
        var sampling: IRSampling?
        qualifiers: while true {
            let start = token
            switch text {
            case "layout":
                _ = try advance()
                try expect("(")
                repeat {
                    let key = try identifier()
                    guard layout[key.text] == nil, !flags.contains(key.text) else {
                        throw glslError(key, .duplicateName, "Duplicate layout key")
                    }
                    if try take("=") {
                        let value = try advance()
                        guard let number = UInt32(value.text) else {
                            throw glslError(value, .unsupportedConstruct, "Layout requires literal uint")
                        }
                        layout[key.text] = number
                    } else {
                        flags.insert(key.text)
                    }
                } while try take(",")
                try expect(")")
            case "readonly", "writeonly", "coherent", "volatile":
                guard memory.insert(try advance().text).inserted else {
                    throw glslError(start, .duplicateName, "Duplicate memory qualifier")
                }
            case "in", "out", "buffer", "uniform", "shared":
                guard qualifier.isEmpty else {
                    throw glslError(start, .invalidBinding, "Multiple storage qualifiers")
                }
                qualifier = try advance().text
            case "flat", "noperspective", "smooth":
                guard interpolation == nil else {
                    throw glslError(start, .invalidBinding, "Multiple interpolation qualifiers")
                }
                switch try advance().text {
                case "flat": interpolation = .flat
                case "noperspective": interpolation = .linear
                default: interpolation = .perspective
                }
            case "centroid", "sample":
                guard sampling == nil else {
                    throw glslError(start, .invalidBinding, "Multiple sampling qualifiers")
                }
                sampling = try advance().text == "centroid" ? .centroid : .sample
            default:
                break qualifiers
            }
        }
        let varying = (stage == .vertex && qualifier == "out") || (stage == .fragment && qualifier == "in")
        guard varying || (interpolation == nil && sampling == nil) else {
            throw glslError(token, .invalidBinding, "Interpolation requires an interstage varying")
        }
        guard interpolation != .flat || sampling == nil else {
            throw glslError(token, .invalidBinding, "Flat interpolation cannot have a sampling qualifier")
        }
        if qualifier=="in",text==";" {
            guard stage == .compute,!hasLocalSize,flags.isEmpty,!layout.isEmpty,layout.keys.allSatisfy({["local_size_x","local_size_y","local_size_z"].contains($0)}),layout.values.allSatisfy({$0>0}),memory.isEmpty else { throw glslError(token,.unsupportedConstruct,"Invalid compute workgroup declaration") }
            workgroup = .init(x:layout["local_size_x"] ?? 1,y:layout["local_size_y"] ?? 1,z:layout["local_size_z"] ?? 1);hasLocalSize=true;try expect(";");return
        }
        if qualifier=="buffer" || qualifier=="uniform" {
            if qualifier == "uniform", let resource = glslOpaqueResourceKind(text) {
                let type = try advance()
                try declareGLSLOpaqueResource(resource, layout: layout, flags: flags, memory: memory, at: type)
                return
            }
            if qualifier == "uniform", isGLSLCombinedTextureType(text) {
                throw glslError(token, .unsupportedConstruct, "Combined sampler bindings require separate texture and sampler declarations")
            }
            let name=try identifier()
            let uniform = qualifier == "uniform"
            let pushConstant = flags.remove("push_constant") != nil
            let readOnly = memory.contains("readonly")
            let writeOnly = memory.contains("writeonly")
            guard flags.isEmpty || flags == ["std140"] || ((!uniform || pushConstant) && flags == ["std430"]),
                  !(readOnly && writeOnly), !uniform || memory.isEmpty else {
                throw glslError(name,.unsupportedConstruct,"Expected a supported uniform or storage block layout with explicit binding")
            }
            let rb: IRResourceBinding?
            if pushConstant {
                guard uniform, layout.isEmpty else {
                    throw glslError(name, .invalidBinding, "Push constants require a uniform block without descriptor bindings")
                }
                rb = nil
            } else {
                guard layout.keys.allSatisfy({ ["set", "binding"].contains($0) }), let binding = layout["binding"] else {
                    throw glslError(name, .invalidBinding, "Resource block requires an explicit binding")
                }
                rb = .init(group: layout["set"] ?? 0, binding: binding)
            }
            let standard: GLSLBlockLayout = (uniform && !pushConstant) || flags.contains("std140") ? .std140 : .std430
            let type = try members(runtime: !uniform, explicitOffsets: true, standard: standard)
            let instance = text == ";" ? nil : try identifier()
            try expect(";")
            if let instance {
                guard !hasGlobalName(instance.text), !instance.text.hasPrefix("gl_") else {
                    throw glslError(instance,.duplicateName,"Duplicate/reserved block instance")
                }
            }
            let exported: [IRStructMember]
            if instance == nil, case .structure(let fields, _) = kind(type) {
                exported = fields
            } else {
                exported = []
            }
            for field in exported {
                try budget.consume(.work)
                guard let member = field.name, !hasGlobalName(member), !member.hasPrefix("gl_") else {
                    throw glslError(name, .duplicateName, "Duplicate/reserved uninstanced block member")
                }
            }
            let access:IRStorageAccess=readOnly ? [.load]:writeOnly ? [.store]:[.load,.store]
            let declared = instance ?? name
            if let rb, globals.contains(where: { $0.binding == rb }) {
                throw glslError(declared, .invalidBinding, "Duplicate resource binding")
            }
            try budget.consume(.nodes);let h=try IRHandle<IRGlobalVariable>(index:globals.count)
            var decorations: IRMemoryDecorations = memory.contains("coherent") ? .coherent : []
            if memory.contains("volatile") { decorations.formUnion([.coherent, .volatile]) }
            let space: IRAddressSpace = pushConstant ? .immediate : uniform ? .uniform : .storage(access: access)
            globals.append(.init(name:declared.text,space:space,binding:rb,type:type,
                                 memoryDecorations: decorations,location:try declared.location))
            if let instance { globalNames[instance.text] = h }
            for (index, field) in exported.enumerated() {
                try budget.consume(.work)
                if let member = field.name {
                    uninstancedBlockMembers[member] = (h, UInt32(index), field.type)
                }
            }
            return
        }
        if try take("struct") {
            guard qualifier.isEmpty,layout.isEmpty,flags.isEmpty,memory.isEmpty else { throw glslError(token,.unsupportedConstruct,"Qualified structure unsupported") }
            let name=try identifier();let type=try members(runtime:false);try expect(";")
            guard namedTypes[name.text]==nil, !hasGlobalName(name.text) else { throw glslError(name,.duplicateName,"Duplicate type") };namedTypes[name.text]=type;return
        }
        guard flags.isEmpty,memory.isEmpty else { throw glslError(token,.unsupportedConstruct,"Unsupported memory/layout qualifier") }
        if try take("const") {
            guard qualifier.isEmpty, layout.isEmpty, let type = try parseType() else {
                throw glslError(token, .unsupportedConstruct, "Expected a global constant type")
            }
            try globalConstant(type)
            return
        }
        let type=try parseType();let name=try identifier()
        if qualifier.isEmpty,text=="(" { guard layout.isEmpty else { throw glslError(name,.unsupportedConstruct,"Function layout") }; try function(name,result:type);return }
        guard let base=type else { throw glslError(name,.invalidType,"Void variable") }
        let finalType=try arraySuffix(base)
        guard !hasGlobalName(name.text), !name.text.hasPrefix("gl_") else { throw glslError(name,.duplicateName,"Duplicate/reserved global") }
        try expect(";")
        if qualifier=="in" || qualifier=="out" {
            let input = qualifier == "in"
            let colorIndex = layout["index"]
            guard stage != .compute, layout.keys.allSatisfy({ $0 == "location" || $0 == "index" }),
                  let location = layout["location"], let shape = IRBitcastShape(kind(finalType)),
                  shape.scalar.kind != .bool else {
                throw glslError(name,.unsupportedConstruct,"Unsupported explicit location interface type")
            }
            guard colorIndex == nil || (stage == .fragment && !input && (colorIndex == 0 || colorIndex == 1)) else {
                throw glslError(name, .invalidBinding, "Color index must be 0 or 1 on a fragment output")
            }
            guard colorIndex != 1 || location == 0 else {
                throw glslError(name, .unsupportedConstruct, "Dual-source fragment colors are supported only at location 0")
            }
            guard !varying || shape.scalar.kind == .float || interpolation == .flat else {
                throw glslError(name, .invalidBinding, "Integer interstage values require flat interpolation")
            }
            let blendSource: UInt32? = colorIndex == 1 ? 1 : nil
            let binding = IRBinding.location(index: location, interpolation: varying ? interpolation ?? .perspective : nil,
                                             sampling: varying && interpolation != .flat ? sampling ?? .center : nil,
                                             blendSource: blendSource, perPrimitive: false)
            guard try !interfaces.contains(where: { io in
                try budget.consume(.work)
                guard io.input == input,
                      case .location(let previous, _, _, let previousSource, _) = io.binding else { return false }
                return location == previous && blendSource == previousSource
            }) else { throw glslError(name,.invalidBinding,"Duplicate stage location") }
            try registerInterface(name.text, type: finalType, binding: binding, input: input)
        } else {
            guard layout.isEmpty,qualifier.isEmpty || qualifier=="shared" else { throw glslError(name,.unsupportedConstruct,"Unsupported global qualifier") }
            if qualifier=="shared",stage != .compute { throw glslError(name,.invalidAddressSpace,"Shared variable outside compute") }
            try budget.consume(.nodes);let h=try IRHandle<IRGlobalVariable>(index:globals.count)
            globals.append(.init(name:name.text,space:qualifier=="shared" ? .workgroup:.private,type:finalType,location:try name.location));globalNames[name.text]=h
        }
    }

    func finalizeGLSLColorBindings() throws {
        guard stage == .fragment else { return }
        var primary: Int?
        var secondary: Int?
        var colorCount = 0
        for index in interfaces.indices {
            try budget.consume(.work)
            let field = interfaces[index]
            guard !field.input, case .location(let location, _, _, let source, _) = field.binding else { continue }
            colorCount += 1
            if location == 0 {
                if source == 1 { secondary = index }
                else { primary = index }
            }
        }
        guard let secondary else { return }
        guard colorCount == 2, let primary, kind(interfaces[primary].type) == kind(interfaces[secondary].type) else {
            throw glslError(token, .invalidBinding, "Dual-source outputs require two matching color types at location 0")
        }
        // GLSL defaults an omitted color index to zero. Mark source zero in IR only for a dual-source pair.
        interfaces[primary].binding = .location(index: 0, interpolation: nil, sampling: nil,
                                                blendSource: 0, perPrimitive: false)
    }

    func access(_ input:GLSLValue) throws -> GLSLValue {
        let op = try advance()
        let base = input.swizzle == nil ? input : try rvalue(input, at: op)
        let type = base.pointer ? try pointee(base.type, at: op) : base.type
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
                    guard let size = IRVectorSize(rawValue: UInt8(indices.count)) else {
                        throw glslError(name, .invalidType, "Swizzle size")
                    }
                    let type = try addType(.vector(size: size, scalar: s))
                    let lanes = indices.map(UInt32.init)
                    if base.pointer && base.writable {
                        return .init(expression: base.expression, type: type, pointer: false, writable: true,
                                     swizzle: .init(pointerType: base.type, indices: lanes))
                    }
                    return try swizzleValue(rvalue(base, at: op), type: type, indices: lanes, at: name)
                }
            }
        } else {
            let index=try rvalue(expression(),at:op);try expect("]")
            guard case .scalar(let s)=kind(index.type),[.sint,.uint].contains(s.kind) else { throw glslError(op,.typeMismatch,"Index requires scalar integer") }
            switch kind(type) {
            case .array(let child,_,_):memberType=child
            case .vector(_,let s):memberType=try addType(.scalar(s))
            case .matrix(_, let rows, let scalar): memberType = try addType(.vector(size: rows, scalar: scalar))
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

    func swizzleValue(_ base: GLSLValue, type: IRHandle<IRType>, indices: [UInt32],
                      at token: GLSLToken) throws -> GLSLValue {
        guard case .vector(let size, _) = kind(type) else {
            throw glslError(token, .invalidType, "Expected swizzle vector type")
        }
        let components = IRSwizzleComponent.allCases
        let lanes = indices.map { components[Int($0)] } + Array(repeating: .x, count: 4 - indices.count)
        let pattern = IRSwizzle(x: lanes[0], y: lanes[1], z: lanes[2], w: lanes[3])
        return try expressionNode(type, .swizzle(size: size, vector: base.expression, pattern: pattern), at: token)
    }
}
