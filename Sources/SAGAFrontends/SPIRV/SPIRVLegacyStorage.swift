import SAGAShaderIR

enum SPVLegacyStorage {
    static let uniform: UInt32 = 2
    static let storageBuffer: UInt32 = 12
    static let block: UInt32 = 2
    static let bufferBlock: UInt32 = 3
    static let lastVersion: UInt32 = 0x00010300
}

extension SPVDecoder {
    func isLegacyStoragePointer(_ pointer: SPVPointer) -> Bool {
        let base = descriptorArrayElements[pointer.base] ?? pointer.base
        return pointer.storage == SPVLegacyStorage.uniform && decorations[base]?[SPVLegacyStorage.bufferBlock] != nil
    }

    func effectivePointerSpace(_ pointer: SPVPointer, _ instruction: SPVInstruction) throws -> IRAddressSpace {
        if isLegacyStoragePointer(pointer) { return .storage(access: [.load, .store]) }
        return try storage(pointer.storage, instruction)
    }

    mutating func validateLegacyStorageClasses(_ instruction: SPVInstruction) throws {
        var hasLegacy = false
        var hasModern = false
        for pointer in pointers.values {
            try charge()
            hasLegacy = hasLegacy || isLegacyStoragePointer(pointer)
            hasModern = hasModern || pointer.storage == SPVLegacyStorage.storageBuffer
            if hasLegacy && hasModern {
                throw instruction.fail("BufferBlock Uniform and StorageBuffer classes cannot be mixed")
            }
        }
    }
}
