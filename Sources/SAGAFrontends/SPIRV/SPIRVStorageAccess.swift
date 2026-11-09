import SAGAShaderIR

struct SPVStorageAccessDecorations {
    static let nonWritable: UInt32 = 24
    static let nonReadable: UInt32 = 25

    var globals: [IRHandle<IRGlobalVariable>: IRStorageAccess] = [:]
    var members: [IRHandle<IRType>: [UInt32: IRStorageAccess]] = [:]
    var descendants: [IRHandle<IRType>: IRStorageAccess] = [:]

    var isEmpty: Bool { globals.isEmpty && members.isEmpty }

    static func prohibited(_ decorations: [UInt32: [UInt32]]) -> IRStorageAccess {
        var access: IRStorageAccess = []
        if decorations[nonWritable] != nil { access.insert(.store) }
        if decorations[nonReadable] != nil { access.insert(.load) }
        return access
    }
}

extension SPVDecoder {
    mutating func prepareStorageAccessDecorations(_ instruction: SPVInstruction) throws {
        for (id, decorations) in self.decorations {
            try charge()
            let prohibited = SPVStorageAccessDecorations.prohibited(decorations)
            if !prohibited.isEmpty, let (global, _) = globalMap[id], globals[global.index].space != .handle {
                storageAccessDecorations.globals[global] = prohibited
            }
        }
        for (id, members) in self.members {
            try charge(members.count + 1)
            guard let type = typeMap[id] else { throw instruction.fail("Unknown member decoration type") }
            for (index, decorations) in members {
                let prohibited = SPVStorageAccessDecorations.prohibited(decorations)
                if !prohibited.isEmpty { storageAccessDecorations.members[type, default: [:]][index] = prohibited }
            }
        }
        guard !storageAccessDecorations.isEmpty else { return }

        // Track any descendant restriction for aggregate accesses and complete write protection for reflection.
        var readOnly: Set<IRHandle<IRType>> = []
        if !storageAccessDecorations.members.isEmpty {
            for (index, entry) in types.enumerated() {
                try charge()
                let type = try IRHandle<IRType>(index: index)
                switch entry.kind {
                case .array(let element, _, _), .bindingArray(let element, _):
                    storageAccessDecorations.descendants[type] = storageAccessDecorations.descendants[element]
                    if readOnly.contains(element) { readOnly.insert(type) }
                case .structure(let fields, _):
                    try charge(fields.count)
                    var descendants: IRStorageAccess = []
                    var allReadOnly = !fields.isEmpty
                    let members = storageAccessDecorations.members[type] ?? [:]
                    for (member, field) in fields.enumerated() {
                        let prohibited = members[UInt32(member)] ?? []
                        descendants.formUnion(prohibited)
                        descendants.formUnion(storageAccessDecorations.descendants[field.type] ?? [])
                        allReadOnly = allReadOnly && (prohibited.contains(.store) || readOnly.contains(field.type))
                    }
                    if !descendants.isEmpty { storageAccessDecorations.descendants[type] = descendants }
                    if allReadOnly { readOnly.insert(type) }
                default: break
                }
            }
        }

        for index in globals.indices {
            try charge()
            let handle = try IRHandle<IRGlobalVariable>(index: index)
            let global = globals[index]
            let prohibited = storageAccessDecorations.globals[handle] ?? []
            guard !prohibited.isEmpty || storageAccessDecorations.descendants[global.type] != nil else { continue }
            guard case .storage = global.space else {
                throw instruction.fail("NonWritable and NonReadable buffer decorations require storage buffers")
            }
            if prohibited.contains(.store) || readOnly.contains(global.type) {
                globals[index] = .init(name: global.name, space: .storage(access: .load), binding: global.binding,
                                       type: global.type, initializer: global.initializer,
                                       memoryDecorations: global.memoryDecorations, location: global.location)
            }
        }
    }
}

extension SPVFunctionLowerer {
    mutating func checkStorageAccess(_ pointer: SPVValue, requiring access: IRStorageAccess, at instruction: SPVInstruction) throws {
        let kind = module.types[expressions[pointer.expression.index].type.index].kind
        guard case .pointer(_, .storage(let declared)) = kind else { return }
        guard declared.contains(access), try prohibitedStorageAccess(pointer, at: instruction).isDisjoint(with: access) else {
            throw instruction.fail("Storage access violates NonWritable or NonReadable decoration")
        }
    }

    mutating func prohibitedStorageAccess(
        _ pointer: SPVValue, at instruction: SPVInstruction
    ) throws -> IRStorageAccess {
        guard !module.storageAccessDecorations.isEmpty,
              case .pointer(_, .storage) = module.types[expressions[pointer.expression.index].type.index].kind else { return [] }
        try work()
        guard let ancestors = storageAccessPaths[pointer.expression], let declared = module.pointers[pointer.type] else {
            throw instruction.fail("Cannot resolve decorated storage access through this pointer", code: .unsupportedConstruct)
        }
        // A canonical buffer can be shared by several SPIR-V views. Only the original
        // view's ancestors and the descendants touched by this access impose restrictions.
        let pointee = try module.type(declared.base, instruction)
        return ancestors.union(module.storageAccessDecorations.descendants[pointee] ?? [])
    }
}
