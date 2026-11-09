import SAGAShaderIR

private enum SPVPointerAccess {
    static let volatile: UInt32 = 1
    static let aligned: UInt32 = 2
    static let nontemporal: UInt32 = 4
    static let makeAvailable: UInt32 = 8
    static let makeVisible: UInt32 = 16
    static let nonPrivate: UInt32 = 32
    static let scoped = makeAvailable | makeVisible
    static let vulkan = scoped | nonPrivate
    static let supportedHints = aligned | nontemporal
}

private enum SPVTexelAccess {
    static let makeAvailable: UInt32 = 0x100
    static let makeVisible: UInt32 = 0x200
    static let nonPrivate: UInt32 = 0x400
    static let volatile: UInt32 = 0x800
    static let nontemporal: UInt32 = 0x4000
    static let scoped = makeAvailable | makeVisible
    static let vulkan = scoped | nonPrivate | volatile
}

extension SPVFunctionLowerer {
    func validatePointerMemoryAccess(_ instruction: SPVInstruction, at start: Int) throws {
        try instruction.minimum(start)
        let operands = instruction.a
        guard operands.count > start else { return }
        let mask = operands[start]
        let known = SPVPointerAccess.supportedHints | SPVPointerAccess.volatile | SPVPointerAccess.vulkan
        guard mask & ~known == 0 else {
            throw instruction.fail("Unsupported Memory Access mask bits", code: .unsupportedConstruct)
        }
        let hasArgument = SPVPointerAccess.aligned | SPVPointerAccess.scoped
        try instruction.count(start + 1 + (mask & hasArgument).nonzeroBitCount)
        let aligned = mask & SPVPointerAccess.aligned != 0
        if aligned, operands[start + 1].nonzeroBitCount != 1 {
            throw instruction.fail("Aligned Memory Access requires a positive power-of-two alignment")
        }
        if mask & SPVPointerAccess.vulkan != 0 {
            guard module.usesVulkanMemoryModel else {
                throw instruction.fail("Pointer memory ordering operands require the Vulkan memory model", code: .unsupportedCapability)
            }
            let store = instruction.op == 62
            let scopeMask = mask & SPVPointerAccess.scoped
            if scopeMask != 0 {
                let expected = store ? SPVPointerAccess.makeAvailable : SPVPointerAccess.makeVisible
                guard scopeMask == expected, mask & SPVPointerAccess.nonPrivate != 0 else {
                    throw instruction.fail("Pointer availability or visibility requires the matching access and NonPrivatePointer")
                }
                let scopeID = operands[start + (aligned ? 2 : 1)]
                guard let scope = module.constantMap[scopeID], let scopeValue = scope.nonnegativeInteger,
                      try [.scalar(.u32), .scalar(.i32)].contains(module.kind(scope.type, instruction)) else {
                    throw instruction.fail("Pointer memory access scope must be a 32-bit integer constant")
                }
                if scopeValue == 1, !module.capabilities.contains(SPVVulkanMemoryModel.deviceScopeCapability) {
                    throw instruction.fail("Device scope requires VulkanMemoryModelDeviceScope", code: .unsupportedCapability)
                }
            }
            throw instruction.fail("NonPrivatePointer and pointer availability or visibility are not supported",
                                   code: .unsupportedConstruct)
        }
        if mask & SPVPointerAccess.volatile != 0 {
            throw instruction.fail("Volatile pointer memory accesses are not supported", code: .unsupportedConstruct)
        }
        // Under logical addressing Aligned is a promise, and Nontemporal is only a cache hint.
    }

    func imageOperandsWithoutMemoryHints(_ instruction: SPVInstruction, at start: Int) throws -> UInt32 {
        let operands = instruction.a
        guard operands.count > start else { return 0 }
        let mask = operands[start]
        if mask & SPVTexelAccess.vulkan != 0 {
            guard module.usesVulkanMemoryModel else {
                throw instruction.fail("Texel memory operands require the Vulkan memory model", code: .unsupportedCapability)
            }
            let scopeMask = mask & SPVTexelAccess.scoped
            if scopeMask != 0 {
                let expected: UInt32 = switch instruction.op {
                case 98: SPVTexelAccess.makeVisible
                case 99: SPVTexelAccess.makeAvailable
                default: 0
                }
                guard scopeMask == expected, mask & SPVTexelAccess.nonPrivate != 0 else {
                    throw instruction.fail("Texel availability or visibility requires the matching image access and NonPrivateTexel")
                }
                try instruction.minimum(start + 2)
            }
            throw instruction.fail("NonPrivateTexel, VolatileTexel and texel availability or visibility are not supported",
                                   code: .unsupportedConstruct)
        }
        if mask & SPVTexelAccess.nontemporal != 0, module.version < 0x00010600 {
            throw instruction.fail("Nontemporal image operands require SPIR-V 1.6", code: .unsupportedCapability)
        }
        return mask & ~SPVTexelAccess.nontemporal
    }
}
