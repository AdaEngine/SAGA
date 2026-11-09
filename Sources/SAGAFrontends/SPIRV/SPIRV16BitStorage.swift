import SAGAShaderIR

enum SPV16BitStorage {
    static let extensionName = "SPV_KHR_16bit_storage"
    static let storageClassExtension = "SPV_KHR_storage_buffer_storage_class"
    static let storage: UInt32 = 4433
    static let uniform: UInt32 = 4434
    static let pushConstant: UInt32 = 4435
    static let interface: UInt32 = 4436
    static let coreVersion: UInt32 = 0x00010300
}

private struct SPV16BitDomains: OptionSet {
    let rawValue: UInt8
    static let integer = Self(rawValue: 1)
    static let floating = Self(rawValue: 2)
}

extension SPVDecoder {
    var has16BitStorage: Bool {
        capabilities.contains(SPV16BitStorage.storage) || capabilities.contains(SPV16BitStorage.uniform)
            || capabilities.contains(SPV16BitStorage.pushConstant) || capabilities.contains(SPV16BitStorage.interface)
    }

    mutating func prepare16BitStorage(_ header: SPVInstruction) throws {
        if has16BitStorage, version < SPV16BitStorage.coreVersion, !extensions.contains(SPV16BitStorage.extensionName) {
            throw header.fail("16-bit storage requires SPIR-V 1.3 or SPV_KHR_16bit_storage", code: .unsupportedCapability)
        }
        guard has16BitStorage || capabilities.contains(9) || capabilities.contains(22) else { return }
        var restricted: SPV16BitDomains = []
        if !capabilities.contains(22) { restricted.insert(.integer) }
        if !capabilities.contains(9) { restricted.insert(.floating) }
        var domains: [SPV16BitDomains] = []
        for (index, type) in types.enumerated() {
            try charge()
            let contained: SPV16BitDomains
            switch type.kind {
            case .scalar(let lane), .vector(_, let lane), .matrix(_, _, let lane):
                contained = lane.width == 2 ? (lane.kind == .float ? .floating : .integer) : []
            case .array(let base, _, _), .bindingArray(let base, _), .pointer(let base, _):
                contained = domains[base.index]
            case .structure(let members, _):
                try charge(members.count)
                contained = members.reduce(into: SPV16BitDomains()) { $0.formUnion(domains[$1.type.index]) }
            default: contained = []
            }
            domains.append(contained)
            if !contained.isDisjoint(with: restricted) {
                storageOnly16BitTypes.insert(try IRHandle(index: index))
            }
        }

        for (_, (handle, pointer)) in globalMap {
            try charge()
            let global = globals[handle.index]
            guard !domains[global.type.index].isEmpty else { continue }
            func fail(_ message: String) -> ShaderFailure {
                .diagnostics([.init(severity: .error, code: .unsupportedCapability, message: message, location: global.location)])
            }
            let storageClass = pointers[pointer]?.storage
            if storageClass == 1 || storageClass == 3 {
                guard capabilities.contains(SPV16BitStorage.interface) else {
                    throw fail("16-bit stage inputs and outputs require StorageInputOutput16")
                }
                continue
            }
            switch global.space {
            case .uniform:
                guard capabilities.contains(SPV16BitStorage.uniform) else {
                    throw fail("16-bit uniform members require UniformAndStorageBuffer16BitAccess")
                }
            case .immediate:
                guard capabilities.contains(SPV16BitStorage.pushConstant) else {
                    throw fail("16-bit push constant members require StoragePushConstant16")
                }
            case .storage:
                guard capabilities.contains(SPV16BitStorage.storage) || capabilities.contains(SPV16BitStorage.uniform) else {
                    throw fail("16-bit storage members require StorageBuffer16BitAccess")
                }
                if storageClass == 12, version < SPV16BitStorage.coreVersion,
                   !extensions.contains(SPV16BitStorage.storageClassExtension) {
                    throw fail("16-bit StorageBuffer requires SPIR-V 1.3 or SPV_KHR_storage_buffer_storage_class")
                }
            default:
                if storageOnly16BitTypes.contains(global.type) {
                    throw fail("Storage-only 16-bit types require a uniform or storage buffer")
                }
            }
        }
        guard !storageOnly16BitTypes.isEmpty else { return }
        for constant in constants {
            try charge()
            guard !storageOnly16BitTypes.contains(constant.type) else {
                throw header.fail("16-bit constants require the corresponding arithmetic capability", code: .unsupportedCapability)
            }
        }
        for override in overrides {
            try charge()
            guard !storageOnly16BitTypes.contains(override.type) else {
                throw header.fail("16-bit specialization constants require the corresponding arithmetic capability",
                                  code: .unsupportedCapability)
            }
        }
        for (id, pointer) in pointers {
            try charge()
            if isStorageOnly16Bit(id), ![1, 2, 3, 9, 12].contains(pointer.storage) {
                throw header.fail("Storage-only 16-bit pointers require an interface or buffer storage class", code: .unsupportedCapability)
            }
        }
        for signature in functionTypes.values {
            try charge(signature.parameters.count + 1)
            if isStorageOnly16Bit(signature.result) || signature.parameters.contains(where: isStorageOnly16Bit) {
                throw header.fail("16-bit function signatures require the corresponding arithmetic capability", code: .unsupportedCapability)
            }
        }
    }

    private func isStorageOnly16Bit(_ rawType: UInt32) -> Bool {
        typeMap[rawType].map(storageOnly16BitTypes.contains) ?? false
    }

    func validateStorageOnly16BitResult(_ i: SPVInstruction) throws {
        guard !storageOnly16BitTypes.isEmpty, let rawType = i.a.first, isStorageOnly16Bit(rawType) else { return }
        let allowed: Bool
        if pointers[rawType] != nil {
            allowed = i.op == 65 || i.op == 66
        } else {
            allowed = i.op == 83 || (i.op == 61 || (113...115).contains(i.op)) && isStorage16BitValue(rawType)
        }
        guard allowed else {
            throw i.fail("16-bit result for opcode \(i.op) requires the corresponding arithmetic capability", code: .unsupportedCapability)
        }
    }

    func validateStorageOnly16BitOperand(_ id: UInt32, _ i: SPVInstruction) throws {
        guard !storageOnly16BitTypes.isEmpty,
              let rawType = functionValueTypes[id] ?? globalMap[id]?.1 ?? constantMap[id]?.type ?? overrideMap[id]?.type,
              isStorageOnly16Bit(rawType) else { return }
        let allowed: Bool
        if pointers[rawType] != nil {
            allowed = [61, 62, 65, 66, 68].contains(i.op)
        } else {
            allowed = i.op == 83 || (i.op == 62 || (113...115).contains(i.op)) && isStorage16BitValue(rawType)
        }
        guard allowed else {
            throw i.fail("16-bit operand for opcode \(i.op) requires the corresponding arithmetic capability", code: .unsupportedCapability)
        }
    }

    func validateStorageOnly16BitConversion(
        from source: IRHandle<IRType>, to target: IRHandle<IRType>, sourceWidth: UInt8, targetWidth: UInt8, _ i: SPVInstruction
    ) throws {
        guard storageOnly16BitTypes.contains(source) || storageOnly16BitTypes.contains(target) else { return }
        guard (113...115).contains(i.op),
              (sourceWidth == 2 && targetWidth == 4) || (sourceWidth == 4 && targetWidth == 2) else {
            throw i.fail("Storage-only 16-bit conversion requires a corresponding 32-bit type", code: .unsupportedCapability)
        }
    }

    private func isStorage16BitValue(_ rawType: UInt32) -> Bool {
        guard let handle = typeMap[rawType] else { return false }
        switch types[handle.index].kind {
        case .scalar(let scalar), .vector(_, let scalar), .matrix(_, _, let scalar): return scalar.width == 2
        default: return false
        }
    }
}
