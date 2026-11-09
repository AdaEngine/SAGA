import SAGAShaderIR

struct GLSLInterfaceAccess {
    let index: UInt32
    let type: IRHandle<IRType>
}

struct GLSLStageLocationGroup {
    let fields: Range<Int>
    let location: UInt32?
    let declaration: GLSLToken
}

private struct GLSLStageLocation: Hashable {
    let index: UInt32
    let input: Bool
    let blendSource: UInt32?
}

extension GLSLParser {
    func stageInterface(
        input: Bool, layout: [String: UInt32], interpolation: IRInterpolation?, sampling: IRSampling?
    ) throws {
        let declaration = token
        guard stage != .compute, layout.keys.allSatisfy({ $0 == "location" || $0 == "index" }) else {
            throw glslError(declaration, .unsupportedConstruct, "Unsupported stage interface declaration")
        }
        let type: IRHandle<IRType>
        let instance: GLSLToken?
        if i + 1 < tokens.count, tokens[i + 1].text == "{" {
            let block = try identifier()
            let blockType = try members(runtime: false)
            instance = text == ";" ? nil : try identifier()
            type = instance == nil ? blockType : try arraySuffix(blockType)
            if instance == nil, case .structure(let members, _) = kind(type) {
                for member in members {
                    try budget.consume(.work)
                    guard let name = member.name, !hasGlobalName(name), !name.hasPrefix("gl_") else {
                        throw glslError(block, .duplicateName, "Duplicate/reserved uninstanced block member")
                    }
                }
            }
        } else {
            guard let base = try parseType() else { throw glslError(token, .invalidType, "Void variable") }
            instance = try identifier()
            type = try arraySuffix(base)
        }
        try expect(";")
        if let instance, hasGlobalName(instance.text) || instance.text.hasPrefix("gl_") {
            throw glslError(instance, .duplicateName, "Duplicate/reserved stage interface")
        }

        let location = layout["location"]
        let colorIndex = layout["index"]
        if let colorIndex {
            guard stage == .fragment, !input, colorIndex <= 1, location != nil,
                  IRBitcastShape(kind(type)) != nil else {
                throw glslError(declaration, .invalidBinding, "Color index requires a scalar or vector fragment output with an explicit location")
            }
            guard colorIndex != 1 || location == 0 else {
                throw glslError(declaration, .unsupportedConstruct, "Dual-source fragment colors are supported only at location 0")
            }
        }
        let varying = (stage == .vertex && !input) || (stage == .fragment && input)
        let binding = IRBinding.location(index: 0, interpolation: varying ? interpolation ?? .perspective : nil,
                                         sampling: varying && interpolation != .flat ? sampling ?? .center : nil,
                                         blendSource: colorIndex == 1 ? 1 : nil, perPrimitive: false)
        try budget.consume(.nodes)
        let handle = try IRHandle<IRGlobalVariable>(index: globals.count)
        let name = instance ?? declaration
        globals.append(.init(name: name.text, space: .private, type: type, location: try name.location))
        if input { stageInputGlobals.insert(handle) }
        if let instance {
            globalNames[instance.text] = handle
        } else if case .structure(let members, _) = kind(type) {
            for (index, member) in members.enumerated() {
                try budget.consume(.work)
                if let name = member.name {
                    uninstancedBlockMembers[name] = (handle, UInt32(index), member.type)
                }
            }
        }

        let first = interfaces.count
        try flattenStageInterface(type, global: handle, name: instance?.text ?? "", path: [],
                                  input: input, binding: binding, at: name)
        stageLocationGroups.append(.init(fields: first..<interfaces.count, location: location, declaration: name))
    }

    private func flattenStageInterface(
        _ type: IRHandle<IRType>, global: IRHandle<IRGlobalVariable>, name: String,
        path: [GLSLInterfaceAccess], input: Bool, binding: IRBinding, at declaration: GLSLToken
    ) throws {
        try enter()
        defer { depth -= 1 }
        switch kind(type) {
        case .array(let element, .constant(let count), _):
            for index in 0..<count {
                try budget.consume(.work, amount: name.utf8.count)
                try flattenStageInterface(element, global: global, name: "\(name)_\(index)",
                                          path: path + [.init(index: index, type: element)],
                                          input: input, binding: binding, at: declaration)
            }
        case .structure(let members, _):
            for (index, member) in members.enumerated() {
                let field = member.name ?? String(index)
                try budget.consume(.work, amount: name.utf8.count + field.utf8.count)
                try flattenStageInterface(member.type, global: global, name: name.isEmpty ? field : "\(name)_\(field)",
                                          path: path + [.init(index: UInt32(index), type: member.type)],
                                          input: input, binding: binding, at: declaration)
            }
        default:
            guard let shape = IRBitcastShape(kind(type)), shape.scalar.kind != .bool,
                  case .location(_, let interpolation, _, _, _) = binding else {
                throw glslError(declaration, .unsupportedConstruct, "Stage interfaces require numeric scalar or vector fields")
            }
            guard interpolation == nil || shape.scalar.kind == .float || interpolation == .flat else {
                throw glslError(declaration, .invalidBinding, "Integer interstage values require flat interpolation")
            }
            try budget.consume(.nodes)
            interfaces.append(.init(global: global, type: type, binding: binding, input: input, name: name, path: path))
        }
    }

    func finalizeGLSLStageLocations() throws {
        var occupied = Set<GLSLStageLocation>()
        func location(_ base: UInt32, _ offset: Int, at token: GLSLToken) throws -> UInt32 {
            try budget.consume(.work)
            guard let offset = UInt32(exactly: offset), offset <= UInt32.max - base else {
                throw glslError(token, .invalidBinding, "Stage location range overflows")
            }
            return base + offset
        }
        func assign(_ group: GLSLStageLocationGroup, at base: UInt32) throws {
            for (offset, index) in group.fields.enumerated() {
                let field = interfaces[index]
                guard case .location(_, let interpolation, let sampling, let source, let perPrimitive) = field.binding else {
                    continue
                }
                let slot = try location(base, offset, at: group.declaration)
                guard occupied.insert(.init(index: slot, input: field.input, blendSource: source)).inserted else {
                    throw glslError(group.declaration, .invalidBinding, "Duplicate stage location")
                }
                interfaces[index].binding = .location(index: slot, interpolation: interpolation, sampling: sampling,
                                                      blendSource: source, perPrimitive: perPrimitive)
            }
        }

        // Reserve explicit ranges before assigning implicit declarations in source order.
        for group in stageLocationGroups {
            if let base = group.location { try assign(group, at: base) }
        }
        var nextFree: [Bool: UInt32] = [:]
        for group in stageLocationGroups where group.location == nil {
            let input = interfaces[group.fields.lowerBound].input
            let firstFree = nextFree[input, default: 0]
            var base = firstFree
            while let conflict = try (0..<group.fields.count).first(where: {
                let slot = try location(base, $0, at: group.declaration)
                return occupied.contains(.init(index: slot, input: input, blendSource: nil))
            }) {
                let next = try location(base, conflict + 1, at: group.declaration)
                // A range search can jump over a hole that still fits a smaller declaration.
                if base == nextFree[input, default: 0], conflict == 0 { nextFree[input] = next }
                base = next
            }
            try assign(group, at: base)
            if base == nextFree[input, default: 0] {
                nextFree[input] = try location(base, group.fields.count, at: group.declaration)
            }
        }
    }

    func interfacePointer(_ field: GLSLInterface, at token: GLSLToken) throws -> GLSLValue {
        let type = globals[field.global.index].type
        let root = try addType(.pointer(base: type, space: .private))
        var pointer = try expressionNode(root, .globalVariable(field.global), at: token, emit: false,
                                         pointer: true, writable: true)
        for component in field.path {
            let type = try addType(.pointer(base: component.type, space: .private))
            pointer = try expressionNode(type, .accessIndex(base: pointer.expression, index: component.index),
                                         at: token, pointer: true, writable: true)
        }
        return pointer
    }
}
