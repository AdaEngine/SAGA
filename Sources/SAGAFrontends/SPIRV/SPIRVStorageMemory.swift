import SAGAShaderIR

enum SPVStorageMemoryDecoration {
    static let volatile: UInt32 = 21
    static let coherent: UInt32 = 23

    static func flags(_ decorations: [UInt32: [UInt32]]) -> IRMemoryDecorations {
        var flags: IRMemoryDecorations = []
        if decorations[coherent] != nil { flags.insert(.coherent) }
        if decorations[volatile] != nil { flags.insert(.volatile) }
        return flags
    }
}

extension SPVDecoder {
    mutating func prepareStorageMemoryDecorations(_ instruction: SPVInstruction) throws {
        var byMember: [IRHandle<IRType>: [UInt32: IRMemoryDecorations]] = [:]
        for (id, fields) in members {
            try charge(fields.count + 1)
            guard let type = typeMap[id] else { throw instruction.fail("Unknown member decoration type") }
            for (index, decorations) in fields {
                let flags = SPVStorageMemoryDecoration.flags(decorations)
                if !flags.isEmpty { byMember[type, default: [:]][index] = flags }
            }
        }

        // A flag can move to the global only when every byte that the shader can access is covered.
        var coverage: [IRHandle<IRType>: (any: IRMemoryDecorations, all: IRMemoryDecorations)] = [:]
        if !byMember.isEmpty {
            for (index, type) in types.enumerated() {
                try charge()
                let handle = try IRHandle<IRType>(index: index)
                switch type.kind {
                case .array(let element, _, _), .bindingArray(let element, _):
                    coverage[handle] = coverage[element]
                case .structure(let fields, _) where !fields.isEmpty:
                    try charge(fields.count)
                    var any: IRMemoryDecorations = []
                    var all: IRMemoryDecorations = [.coherent, .volatile]
                    for (member, field) in fields.enumerated() {
                        let direct = byMember[handle]?[UInt32(member)] ?? []
                        let nested = coverage[field.type]
                        any.formUnion(direct.union(nested?.any ?? []))
                        all.formIntersection(direct.union(nested?.all ?? []))
                    }
                    if !any.isEmpty { coverage[handle] = (any, all) }
                default: break
                }
            }
        }
        storageMemoryTypes = Set(coverage.keys)

        for (id, (handle, _)) in globalMap {
            try charge()
            let global = globals[handle.index]
            let direct = SPVStorageMemoryDecoration.flags(decorations[id] ?? [:])
            let nested = coverage[global.type]
            guard !direct.isEmpty || nested != nil else { continue }
            guard case .storage = global.space else {
                throw instruction.fail("Coherent and Volatile require storage buffer memory", code: .unsupportedConstruct)
            }
            let flags = direct.union(nested?.all ?? [])
            guard (nested?.any ?? []).subtracting(flags).isEmpty else {
                throw instruction.fail("Partial storage buffer Coherent or Volatile decoration is not supported",
                                       code: .unsupportedConstruct)
            }
            globals[handle.index] = .init(name: global.name, space: global.space, binding: global.binding, type: global.type,
                                          initializer: global.initializer, memoryDecorations: flags, location: global.location)
        }
    }
}
